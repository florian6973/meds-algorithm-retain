"""TECO — Transformer-based Encounter-level Clinical Outcome model, over MEDS event streams.

Source: Kim *et al.*, "A deep learning model for clinical outcome prediction using longitudinal inpatient
electronic health records", *JAMIA Open* 8(2) ooaf026 (2025). **No code was published**, so this is a
reimplementation from the paper's Methods; ``IMPLEMENTATION_REPORT.md`` records every element as ported,
adapted or omitted, and lists the assumptions the paper leaves open.

What TECO does: an ICU encounter is aligned onto a **regular grid of 15-minute intervals**, each interval
holding the mean of every time-dependent variable in it (carried forward from the previous interval when
absent) concatenated with the patient's static baseline variables. That feature vector is projected to 512
dimensions, given a relative positional encoding, and read by a 6-layer / 8-head Transformer encoder whose
output feeds a linear softmax classifier. The point of the grid is that one encounter-level model can be
re-scored at every new interval instead of training one model per prediction time.

**The architecture is pinned by the paper's parameter count, not guessed.** The paper reports 18 928 130
trainable parameters. Six ``nn.TransformerEncoderLayer(512, 8, 2048)`` layers account for 18 914 304 and a
``Linear(512, 2)`` head for 1 026, leaving exactly 12 800 = 512 x 25 — a bias-free linear embedding of 25
input features. Nothing is left over, so the positional encoding carries **no parameters**: it is a fixed
sinusoidal encoding of each interval's offset, not a learned table or a learned attention bias. See
:func:`teco_parameter_count` and the doctest on :class:`Model`, which reproduce the published number.

The batch-field test (``docs/PORTING-A-MODEL.md`` step 2): TECO's ``forward`` reads a dense
``[B, n_intervals, n_features]`` matrix of *forward-filled, interval-averaged* values plus a static vector.
It reads no timestamps beyond the grid geometry and no missingness mask — because its preprocessing already
moved that information into the grid. A MEDS batch is the opposite shape: a sparse, irregular measurement
stream. :class:`IntervalGrid` therefore *is* the port of TECO's preprocessing, not a convenience wrapper —
it rebuilds the interval grid from ``code`` / ``numeric_value`` / ``time_delta_days`` so the encoder sees
what the published model saw. Dropping it and feeding raw measurement embeddings would be a different model.

The ``supervised`` DAG calls exactly two hooks here: ``compute_loss`` (from ``supervised_train``) and
``predict_step`` (from ``predict``).
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from meds_model_base.lightning.modules import PAD_INDEX, BaseLightningModule, padding_mask
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

#: Architecture reported in the paper (Methods, "Model development"); see the module docstring for how
#: the parameter count pins every one of these.
DEFAULT_D_MODEL = 512
DEFAULT_NHEAD = 8
DEFAULT_NUM_LAYERS = 6
DEFAULT_DIM_FEEDFORWARD = 2048
DEFAULT_DROPOUT = 0.0

#: Grid geometry: 15-minute alignment intervals, and the 96-hour lookback the paper gives for the 24-hour
#: sub-model — 384 intervals.
DEFAULT_INTERVAL_MINUTES = 15.0
DEFAULT_LOOKBACK_HOURS = 96.0

MINUTES_PER_DAY = 24.0 * 60.0


def teco_parameter_count(
    n_features: int = 25,
    d_model: int = DEFAULT_D_MODEL,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
    n_classes: int = 2,
) -> int:
    """Trainable parameters of the published TECO geometry, derived from its parts.

    Kept as a function because it is the evidence for the architecture: the paper reports one number and
    no layer-by-layer breakdown, and this is the decomposition that reproduces it exactly.

    Per encoder layer: ``4*d^2 + 4*d`` for self-attention (packed QKV + output projection), ``2*d*ff + ff +
    d`` for the feed-forward block, and ``4*d`` for the two LayerNorms. Plus a bias-free
    ``Linear(n_features, d)`` embedding and a ``Linear(d, n_classes)`` head.

    Examples:
        >>> teco_parameter_count()
        18928130
        >>> teco_parameter_count() == Model(vocab_size=25).n_parameters()
        True

        The 25 features are the paper's own input width, recovered from the reported total: it is the only
        value for which the decomposition closes.

        >>> teco_parameter_count(n_features=24) == 18928130
        False
    """
    attention = 4 * d_model * d_model + 4 * d_model
    feedforward = 2 * d_model * dim_feedforward + dim_feedforward + d_model
    norms = 4 * d_model
    encoder = num_layers * (attention + feedforward + norms)
    embedding = n_features * d_model  # bias-free
    head = d_model * n_classes + n_classes
    return encoder + embedding + head


def sinusoidal_encoding(n_intervals: int, d_model: int) -> Tensor:
    """Fixed sinusoidal encoding of each interval's offset *back from the prediction time*.

    Position 0 is the most recent interval, which is what makes the encoding **relative**: the same
    physiological history produces the same encoding whenever it is re-scored, so a model that updates its
    risk every interval is not also seeing the position shift underneath it. That is the property the paper
    asks of it ("relative positional encoding to avoid imputing large amounts of default values for time
    points without measurement") and it is parameter-free, as the parameter count requires.

    Returns a ``[1, n_intervals, d_model]`` tensor ordered oldest-interval-first, so it adds directly to a
    grid whose last column is the prediction time.

    Examples:
        >>> pe = sinusoidal_encoding(4, 6)
        >>> pe.shape
        torch.Size([1, 4, 6])

        The final row is the prediction time itself, i.e. offset 0 — ``sin(0), cos(0)`` repeated:

        >>> pe[0, -1].tolist()
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        Adding intervals to the *front* leaves the recent end untouched, which is the whole point of
        anchoring at the prediction time rather than at admission:

        >>> torch.allclose(sinusoidal_encoding(9, 6)[0, -4:], pe[0])
        True
    """
    if d_model % 2:
        raise ValueError(f"d_model must be even for a sinusoidal encoding; got {d_model}.")
    offsets = torch.arange(n_intervals - 1, -1, -1, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(n_intervals, d_model)
    encoding[:, 0::2] = torch.sin(offsets * scale)
    encoding[:, 1::2] = torch.cos(offsets * scale)
    return encoding.unsqueeze(0)


class IntervalGrid(nn.Module):
    """TECO's input construction: interval-mean alignment, carry-forward, statics, linear embedding.

    This is the port of the paper's preprocessing, rebuilt from a MEDS batch. For each subject the module
    walks back ``lookback_hours`` from the end of the sequence (which, under the task datamodule's
    ``TO_END`` sampling, is the prediction time), lays down ``n_intervals`` bins of ``interval_minutes``,
    averages every measurement that falls in a bin, carries the last observed value forward across empty
    bins, and adds the subject's static measurements to every bin. The result is embedded by a **bias-free**
    linear layer, exactly as the parameter count requires.

    Two adaptations the MEDS format forces, both recorded in ``IMPLEMENTATION_REPORT.md``:

    - **The feature axis is the code vocabulary**, not 11 hand-chosen clinical variables. mSOFA and the
      SF ratio are cohort-specific derived quantities; a MEDS model is handed an arbitrary dataset and
      cannot resolve them.
    - **A measurement with no numeric value contributes 1.0** in its code's column. TECO's variables are
      all numeric; MEDS carries value-less codes too, and 1.0 makes such a code's contribution exactly its
      embedding vector, which is the usual convention for an occurrence indicator.

    The grid's right edge is the subject's **last measurement in the sampled window**, not the label's
    ``prediction_time``. A ``MEDSTorchBatch`` carries inter-measurement gaps but no absolute timestamp, so
    the offset between the two is not recoverable from the batch. Under ``TO_END`` sampling the last
    measurement is the most recent event at or before the prediction time, so the two differ by less than
    the gap to the next observation — under an interval for a densely monitored ICU encounter, more for a
    sparse one.

    Memory is ``O(batch x n_intervals x U)`` where ``U`` is the number of *distinct codes present in the
    batch* — not the vocabulary size. The dense grid over the full vocabulary is never materialized; since
    the embedding is linear and bias-free, restricting both the grid and the embedding matrix to the codes
    actually present is algebraically identical (``tests/test_interval_grid.py`` asserts this against a
    naive dense implementation).

    Args:
        vocab_size: code vocabulary size; injected by the training commands from the datamodule.
        d_model: embedding width.
        interval_minutes: width of one alignment interval (the paper's 15 minutes).
        lookback_hours: how far back the grid reaches from the prediction time (the paper's 96 hours).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_D_MODEL,
        interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
        lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    ) -> None:
        super().__init__()
        if interval_minutes <= 0 or lookback_hours <= 0:
            raise ValueError(
                f"interval_minutes and lookback_hours must be positive; got {interval_minutes} and "
                f"{lookback_hours}."
            )
        self.vocab_size = vocab_size
        self.interval_minutes = float(interval_minutes)
        self.lookback_hours = float(lookback_hours)
        self.n_intervals = max(1, round(lookback_hours * 60.0 / interval_minutes))
        self.embed = nn.Linear(vocab_size, d_model, bias=False)

    def interval_index(self, batch: MEDSTorchBatch) -> tuple[Tensor, Tensor]:
        """Return ``(interval_index, in_window)``, both ``[B, L]``, for the batch's measurements.

        ``time_delta_days`` is the gap to the *next* event, so an exclusive cumulative sum gives each
        measurement's elapsed time from the start of the sampled window, and the largest such value is the
        window's own length. Working backwards from that end — rather than forwards from the start — is
        what makes the grid line up with the prediction time for every subject regardless of how much
        history each one has.

        Zeroing the deltas at padded positions first makes the arithmetic independent of which side the
        collator padded.

        Examples:
            >>> from types import SimpleNamespace
            >>> batch = SimpleNamespace(
            ...     code=torch.tensor([[3, 4, 5, 0]]),
            ...     time_delta_days=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            ... )
            >>> grid = IntervalGrid(vocab_size=8, d_model=4, interval_minutes=1440, lookback_hours=72)
            >>> grid.n_intervals
            3

            Three days of lookback in one-day intervals: the two-day-old measurement lands in interval 0,
            the one-day-old in interval 1, and the most recent in interval 2 (the prediction time).

            >>> index, in_window = grid.interval_index(batch)
            >>> index.tolist()
            [[0, 1, 2, 2]]
            >>> in_window.tolist()
            [[True, True, True, False]]

            Padding is excluded, and so is anything older than the lookback window:

            >>> narrow = IntervalGrid(vocab_size=8, d_model=4, interval_minutes=1440, lookback_hours=24)
            >>> narrow.interval_index(batch)[1].tolist()
            [[False, False, True, False]]
        """
        valid = padding_mask(batch)
        delta = torch.nan_to_num(batch.time_delta_days) * valid
        elapsed = torch.cumsum(delta, dim=1) - delta
        window_end = (elapsed * valid).amax(dim=1, keepdim=True)
        age_minutes = (window_end - elapsed).clamp(min=0.0) * MINUTES_PER_DAY
        offset = torch.floor(age_minutes / self.interval_minutes)
        index = self.n_intervals - 1 - offset
        return index.clamp(0, self.n_intervals - 1).long(), valid & (index >= 0)

    def dynamic_values(self, batch: MEDSTorchBatch) -> Tensor:
        """``[B, L]`` value per measurement: the numeric value, or 1.0 for a value-less code."""
        numeric = torch.nan_to_num(batch.numeric_value)
        return torch.where(batch.numeric_value_mask, numeric, torch.ones_like(numeric))

    def static_embedding(self, batch: MEDSTorchBatch) -> Tensor | None:
        """``[B, d_model]`` contribution of the static (baseline) measurements, or ``None`` if absent.

        TECO holds the baseline variables constant across the whole time range, so this is added to every
        interval. It needs ``static_inclusion_mode: INCLUDE`` on the datamodule, which
        ``configs/profile/default.yaml`` sets; with the template's default ``OMIT`` the batch carries no
        static tensors and this returns ``None``.
        """
        code = getattr(batch, "static_code", None)
        if code is None or code.numel() == 0:
            return None
        numeric = torch.nan_to_num(batch.static_numeric_value)
        mask = batch.static_numeric_value_mask
        value = torch.where(mask, numeric, torch.ones_like(numeric)) * (code != PAD_INDEX)
        return (self.embed.weight[:, code] * value).sum(-1).transpose(0, 1)

    def forward(self, batch: MEDSTorchBatch) -> Tensor:
        """Return the ``[B, n_intervals, d_model]`` embedded interval grid."""
        index, in_window = self.interval_index(batch)
        value = self.dynamic_values(batch)

        rows, positions = in_window.nonzero(as_tuple=True)
        # The feature axis is compacted to the codes present in this batch. With a bias-free linear
        # embedding that is exact, not an approximation: the columns dropped are multiplied by an
        # all-zero row of the grid. An empty batch window leaves a `[B, T, 0]` grid, whose product with
        # the `[0, d_model]` embedding is the correct all-zero encoding.
        present, dense_code = torch.unique(batch.code[rows, positions], return_inverse=True)
        n_codes = present.numel()
        n_cells = batch.batch_size * self.n_intervals * n_codes
        cell = (rows * self.n_intervals + index[rows, positions]) * n_codes + dense_code

        total = value.new_zeros(n_cells).index_add_(0, cell, value[rows, positions])
        count = value.new_zeros(n_cells).index_add_(0, cell, torch.ones_like(cell, dtype=value.dtype))
        shape = (batch.batch_size, self.n_intervals, n_codes)
        observed = (count > 0).view(shape)
        # The interval mean, then carry-forward: `last` is the most recent observed interval at or before
        # each position, and `seen` zeroes the leading stretch where nothing has been observed yet (TECO's
        # "no previous interval to carry" case).
        means = (total / count.clamp(min=1.0)).view(shape)
        intervals = torch.arange(self.n_intervals, device=means.device).view(1, -1, 1).expand(shape)
        last = torch.where(observed, intervals, torch.zeros_like(intervals)).cummax(dim=1).values
        seen = observed.cumsum(dim=1) > 0
        grid = torch.gather(means, 1, last) * seen

        embedded = grid @ self.embed.weight[:, present].t()
        static = self.static_embedding(batch)
        return embedded if static is None else embedded + static.unsqueeze(1)


class Model(BaseLightningModule):
    """TECO: a Transformer encoder over the aligned interval grid, with a binary softmax head.

    Args:
        vocab_size: code vocabulary size; injected by the training commands from the datamodule.
        d_model: embedding width (paper: 512).
        nhead: attention heads per layer (paper: 8).
        num_layers: encoder layers (paper: 6).
        dim_feedforward: feed-forward width (paper: 2048, PyTorch's default).
        dropout: paper: 0 — the hyperparameter search selected no dropout.
        interval_minutes: alignment interval (paper: 15).
        lookback_hours: history the grid spans (paper: 96).
        pooling: which encoder position the head reads. ``last`` is the prediction time itself, which is
            what an encounter-level model re-scored every interval is asking about; ``mean`` averages the
            window. The paper does not say which it used — see ``IMPLEMENTATION_REPORT.md``.
        optimizer: optimizer factory (``params -> Optimizer``), from ``configs/optimizer/sgd.yaml``.
        scheduler: optional scheduler factory (``optimizer -> LRScheduler``).

    Examples:
        The published geometry, reproducing the paper's reported parameter count on its own input width:

        >>> Model(vocab_size=25).n_parameters()
        18928130

        Dropout is 0 by default, as the paper's selected hyperparameters have it:

        >>> Model(vocab_size=25).hparams.dropout
        0.0
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
        lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
        pooling: str = "last",
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        if pooling not in {"last", "mean"}:
            raise ValueError(f"pooling must be 'last' or 'mean'; got {pooling!r}.")
        self.vocab_size = vocab_size
        self.pooling = pooling
        self.grid = IntervalGrid(vocab_size, d_model, interval_minutes, lookback_hours)
        # GELU and PyTorch's post-norm default; `norm=None` on the stack, since a final LayerNorm would
        # put the parameter count 1024 over the published one.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.register_buffer(
            "positional_encoding", sinusoidal_encoding(self.grid.n_intervals, d_model), persistent=False
        )
        self.head = nn.Linear(d_model, 2)

    def n_parameters(self) -> int:
        """Trainable parameter count — the number the paper reports for its own 25-feature input."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, batch: MEDSTorchBatch) -> Tensor:
        """Return the ``[B, d_model]`` encounter representation at the prediction time.

        No key-padding mask is passed to the encoder, and that is not an oversight: TECO's grid is dense by
        construction — every interval holds either an observation or a carried-forward value — so there is
        nothing to mask. The variable-length object was the measurement stream, and the grid is what
        replaced it.
        """
        hidden = self.encoder(self.grid(batch) + self.positional_encoding)
        return hidden.mean(dim=1) if self.pooling == "mean" else hidden[:, -1]

    def logits(self, batch: MEDSTorchBatch) -> Tensor:
        """Return ``[B, 2]`` class logits; the paper's linear layer with a softmax activation."""
        return self.head(self.encode(batch))

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        """Softmax cross-entropy against ``batch.boolean_value``, plus accuracy for the logs.

        Two classes rather than one logit because the paper's head is ``Linear(d_model, 2)`` followed by a
        softmax; that is also what makes the parameter count close.
        """
        logits = self.logits(batch)
        target = batch.boolean_value.long()
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).to(logits.dtype).mean()
        return loss, {"acc": acc}

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        """Mortality probability in dataloader order; ``predict`` schema- and coverage-checks these."""
        return {"predicted_boolean_probability": torch.softmax(self.logits(batch), dim=-1)[:, 1]}

    def configure_optimizers(self):
        """As the base class, except the schedule steps per **epoch**.

        TECO halves the learning rate "every 50 epochs", and ``BaseLightningModule`` hardcodes
        ``interval: "step"`` — which would halve it every 50 *optimizer steps* instead, a schedule orders
        of magnitude faster than the published one. The contract is vendored and must not be edited, so
        the interval is corrected here; ``IMPLEMENTATION_REPORT.md`` records it as an upstream finding.
        """
        parameters = self.parameters()
        optimizer = (
            self._optimizer_factory(parameters)
            if self._optimizer_factory is not None
            else torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
        )
        if self._scheduler_factory is None:
            return optimizer
        scheduler = self._scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
