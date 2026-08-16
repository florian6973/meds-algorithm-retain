"""Batching, the optimisation loop, and scoring.

The loop follows `train_RETAIN` in `mp2893/retain` @ 9fd39c46: length-sorted minibatches whose *order* is
reshuffled every epoch, Adadelta by default, the four explicit L2 penalties added to a binary cross-entropy
guarded by `log_eps`, and the parameters of the epoch with the best validation AUC.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import RetainConfig
from .model import Retain
from .windows import SequenceBatch


def resolve_device(selector: str) -> torch.device:
    if selector == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(selector)


def gpu_metrics(device: torch.device) -> dict[str, Any] | None:
    """SPEC section 6's GPU block, or `None` when no accelerator was used."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "source": "torch.cuda.max_memory_allocated",
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based ROC AUC with tie correction. `None` when only one class is present.

    Equivalent to `sklearn.metrics.roc_auc_score`, which the official code calls, without taking on
    scikit-learn as a runtime dependency for a single function.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    ordered = scores[order]
    start = 0
    while start < ordered.size:
        stop = start + 1
        while stop < ordered.size and ordered[stop] == ordered[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[labels].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


@dataclass
class TensorBatch:
    codes: torch.Tensor
    offsets: torch.Tensor
    slots: torch.Tensor
    lengths: torch.Tensor
    gaps: torch.Tensor | None
    n_rows: int
    n_visits: int


def collate(batch: SequenceBatch, rows: np.ndarray, config: RetainConfig, device: torch.device) -> TensorBatch:
    """Build the `EmbeddingBag` view of `rows` and the padded grid they scatter into.

    A row with no history keeps a length of one so the attention softmax has a position to normalise
    over. Its single visit carries no bag, so `v = 0`, `c = 0`, and the score is the model's learned
    intercept. That is what lets every requested key be scored instead of excluded.
    """
    rows = np.asarray(rows, dtype=np.int64)
    lengths = batch.lengths[rows]
    n_visits = int(max(lengths.max(initial=0), 1))
    gaps_source = batch.visit_gap_days() if config.use_time else None

    code_chunks: list[np.ndarray] = []
    width_chunks: list[np.ndarray] = []
    slot_chunks: list[np.ndarray] = []
    gap_chunks: list[np.ndarray] = []
    for position, row in enumerate(rows.tolist()):
        start, stop = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
        if stop == start:
            continue
        code_chunks.append(batch.codes[int(batch.visit_ptr[start]) : int(batch.visit_ptr[stop])])
        width_chunks.append(batch.visit_ptr[start + 1 : stop + 1] - batch.visit_ptr[start:stop])
        slot_chunks.append(position * n_visits + np.arange(stop - start, dtype=np.int64))
        if gaps_source is not None:
            gap_chunks.append(gaps_source[start:stop])

    if code_chunks:
        codes = np.concatenate(code_chunks).astype(np.int64)
        widths = np.concatenate(width_chunks).astype(np.int64)
        offsets = np.zeros(widths.size + 1, dtype=np.int64)
        np.cumsum(widths, out=offsets[1:])
        slots = np.concatenate(slot_chunks)
    else:
        codes = np.zeros(0, dtype=np.int64)
        offsets = np.zeros(1, dtype=np.int64)
        slots = np.zeros(0, dtype=np.int64)

    gaps = None
    if gaps_source is not None:
        stacked = np.concatenate(gap_chunks) if gap_chunks else np.zeros(0, dtype=np.float32)
        gaps = torch.from_numpy(np.ascontiguousarray(stacked)).to(device)
    return TensorBatch(
        codes=torch.from_numpy(codes).to(device),
        offsets=torch.from_numpy(offsets).to(device),
        slots=torch.from_numpy(slots).to(device),
        lengths=torch.from_numpy(np.maximum(lengths, 1)),
        gaps=gaps,
        n_rows=int(rows.size),
        n_visits=n_visits,
    )


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor, log_eps: float) -> torch.Tensor:
    """`-(y log(y_hat + eps) + (1-y) log(1 - y_hat + eps))`, exactly as in `build_model`.

    Written out rather than delegated to `binary_cross_entropy_with_logits` because upstream adds
    `log_eps` inside both logarithms, which changes the loss surface near saturation.
    """
    probabilities = torch.sigmoid(logits)
    return -(
        targets * torch.log(probabilities + log_eps)
        + (1.0 - targets) * torch.log(1.0 - probabilities + log_eps)
    ).mean()


def _optimizer(model: Retain, config: RetainConfig) -> torch.optim.Optimizer:
    parameters = model.trainable_parameters()
    if config.solver == "adam":
        # `adam()` upstream parameterises b1/b2 as 1 - beta, so b1=0.1, b2=0.001 is the usual
        # (0.9, 0.999), and its fix1/fix2 terms are standard bias correction.
        return torch.optim.Adam(parameters, lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    # The Theano `adadelta()` block is rho=0.95 with eps=1e-6 and an implicit unit learning rate.
    return torch.optim.Adadelta(parameters, lr=1.0, rho=0.95, eps=1e-6)


@dataclass
class TrainingHistory:
    """A commit-safe record of how the fit actually went."""

    epochs_run: int = 0
    best_epoch: int = 0
    best_validation_auc: float | None = None
    train_losses: list[float] = field(default_factory=list)
    validation_aucs: list[float | None] = field(default_factory=list)
    n_train_rows: int = 0
    n_validation_rows: int = 0
    n_positive_train_rows: int = 0
    n_positive_validation_rows: int = 0
    n_empty_train_sequences: int = 0
    n_empty_validation_sequences: int = 0
    optimizer_steps: int = 0
    selected_by: str = "validation_auc"

    def summary(self) -> dict[str, Any]:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_validation_auc": self.best_validation_auc,
            "train_losses": [round(value, 6) for value in self.train_losses],
            "validation_aucs": [None if v is None else round(v, 6) for v in self.validation_aucs],
            "n_train_rows": self.n_train_rows,
            "n_validation_rows": self.n_validation_rows,
            "n_positive_train_rows": self.n_positive_train_rows,
            "n_positive_validation_rows": self.n_positive_validation_rows,
            "n_empty_train_sequences": self.n_empty_train_sequences,
            "n_empty_validation_sequences": self.n_empty_validation_sequences,
            "optimizer_steps": self.optimizer_steps,
            "selected_by": self.selected_by,
        }


def _length_sorted(batch: SequenceBatch) -> np.ndarray:
    """`len_argsort`: order rows by sequence length so a minibatch pads as little as possible."""
    return np.argsort(batch.lengths, kind="stable")


def score(
    model: Retain, batch: SequenceBatch, config: RetainConfig, device: torch.device
) -> np.ndarray:
    """Probabilities for every row of `batch`, in `batch`'s own row order."""
    model.eval()
    scores = np.zeros(batch.n_keys, dtype=np.float64)
    order = _length_sorted(batch)
    with torch.no_grad():
        for start in range(0, order.size, config.batch_size):
            rows = order[start : start + config.batch_size]
            tensors = collate(batch, rows, config, device)
            output = model(
                tensors.codes,
                tensors.offsets,
                tensors.slots,
                tensors.lengths,
                tensors.gaps,
                n_rows=tensors.n_rows,
                n_visits=tensors.n_visits,
            )
            scores[rows] = torch.sigmoid(output.logits).detach().cpu().numpy().astype(np.float64)
    return scores


def train(
    *,
    vocabulary_size: int,
    train_batch: SequenceBatch,
    train_labels: np.ndarray,
    validation_batch: SequenceBatch,
    validation_labels: np.ndarray,
    config: RetainConfig,
    device: torch.device,
) -> tuple[Retain, TrainingHistory]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    shuffler = random.Random(config.seed)

    model = Retain(vocabulary_size, config).to(device)
    optimizer = _optimizer(model, config)
    targets = torch.from_numpy(train_labels.astype(np.float32)).to(device)

    order = _length_sorted(train_batch)
    n_batches = max(1, int(np.ceil(order.size / config.batch_size)))
    history = TrainingHistory(
        n_train_rows=train_batch.n_keys,
        n_validation_rows=validation_batch.n_keys,
        n_positive_train_rows=int(train_labels.sum()),
        n_positive_validation_rows=int(validation_labels.sum()),
        n_empty_train_sequences=train_batch.n_empty,
        n_empty_validation_sequences=validation_batch.n_empty,
    )
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(config.n_epochs):
        model.train()
        losses: list[float] = []
        for index in shuffler.sample(range(n_batches), n_batches):
            rows = order[index * config.batch_size : (index + 1) * config.batch_size]
            if not rows.size:
                continue
            tensors = collate(train_batch, rows, config, device)
            output = model(
                tensors.codes,
                tensors.offsets,
                tensors.slots,
                tensors.lengths,
                tensors.gaps,
                n_rows=tensors.n_rows,
                n_visits=tensors.n_visits,
            )
            loss = _cross_entropy(output.logits, targets[rows], config.log_eps)
            cost = loss + model.regularization()
            optimizer.zero_grad(set_to_none=True)
            cost.backward()
            optimizer.step()
            history.optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
        history.epochs_run = epoch + 1
        history.train_losses.append(float(np.mean(losses)) if losses else float("nan"))

        auc = None
        if validation_batch.n_keys:
            auc = roc_auc(validation_labels, score(model, validation_batch, config, device))
        history.validation_aucs.append(auc)
        if auc is not None and (history.best_validation_auc is None or auc > history.best_validation_auc):
            history.best_validation_auc = auc
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if history.best_validation_auc is None:
        # No usable validation signal: either no tuning rows, or one class only. Upstream would have
        # saved nothing at all here; keeping the final epoch is the closest usable behaviour and is
        # recorded so the report cannot claim a selection that never happened.
        history.selected_by = "final_epoch"
        history.best_epoch = history.epochs_run - 1
    else:
        model.load_state_dict(best_state)
    return model, history
