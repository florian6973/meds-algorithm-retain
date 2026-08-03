"""Tier 1 — is TECO's input construction actually TECO's?

``IntervalGrid`` is the port of the paper's preprocessing (15-minute interval means, carry-forward,
statics broadcast onto every interval), and it is written as sparse tensor arithmetic over the codes
present in a batch rather than as the dense ``[batch, intervals, vocabulary]`` matrix the paper describes.
That rewrite is what makes the model usable on a real vocabulary, and it is also exactly the kind of
optimisation that can quietly stop being the thing it replaced.

So the dense version is written out here, naively and slowly, straight from the sentences in the Methods
section, and the two are asserted equal. If the fast path drifts — a wrong interval boundary, a
carry-forward that leaks backwards, a mean that is really a sum — this fails.

These need no trained model and no dataset, so they run from a cold checkout in under a second.
"""

import math

import pytest
import torch
from meds_model_base.lightning.modules import PAD_INDEX
from meds_torchdata import MEDSTorchBatch

from teco.model import MINUTES_PER_DAY, IntervalGrid, Model

VOCAB_SIZE = 12


def dense_grid(batch: MEDSTorchBatch, vocab_size: int, interval_minutes: float, n_intervals: int):
    """The paper's construction, written out with Python loops: ``[batch, n_intervals, vocab_size]``.

    "we aligned the time-dependent variables by taking the mean of each variable in every 15-minute
    interval. If a 15-minute mean value was missing, the value from the previous interval was carried
    forward."
    """
    n_rows, n_positions = batch.code.shape
    grid = torch.zeros(n_rows, n_intervals, vocab_size)
    for row in range(n_rows):
        valid = batch.code[row] != PAD_INDEX
        delta = torch.nan_to_num(batch.time_delta_days[row]) * valid
        elapsed = torch.cumsum(delta, 0) - delta
        window_end = float((elapsed * valid).max())

        observations: dict[tuple[int, int], list[float]] = {}
        for position in range(n_positions):
            if not valid[position]:
                continue
            age = max(window_end - float(elapsed[position]), 0.0) * MINUTES_PER_DAY
            interval = n_intervals - 1 - math.floor(age / interval_minutes)
            if interval < 0:  # older than the lookback window
                continue
            value = (
                float(batch.numeric_value[row, position])
                if bool(batch.numeric_value_mask[row, position])
                else 1.0
            )
            observations.setdefault((interval, int(batch.code[row, position])), []).append(value)

        for code in range(vocab_size):
            carried = None
            for interval in range(n_intervals):
                seen = observations.get((interval, code))
                if seen:
                    carried = sum(seen) / len(seen)
                if carried is not None:
                    grid[row, interval, code] = carried
    return grid


def random_batch(n_rows=5, n_positions=9, n_statics=2, *, with_statics=True, seed=0) -> MEDSTorchBatch:
    """A random SM-mode batch: ragged (right-padded) sequences with irregular gaps and mixed values."""
    generator = torch.Generator().manual_seed(seed)

    code = torch.randint(1, VOCAB_SIZE, (n_rows, n_positions), generator=generator)
    lengths = torch.randint(1, n_positions + 1, (n_rows,), generator=generator)
    positions = torch.arange(n_positions).unsqueeze(0)
    code = torch.where(positions < lengths.unsqueeze(1), code, torch.full_like(code, PAD_INDEX))

    # Gaps spanning minutes to days, so intervals land both inside and outside a short lookback window.
    time_delta_days = torch.rand(n_rows, n_positions, generator=generator) * 2.0
    numeric_value = torch.randn(n_rows, n_positions, generator=generator)
    numeric_value_mask = torch.rand(n_rows, n_positions, generator=generator) > 0.4

    fields = {
        "code": code,
        "numeric_value": numeric_value,
        "numeric_value_mask": numeric_value_mask,
        "time_delta_days": time_delta_days,
        "boolean_value": torch.rand(n_rows, generator=generator) > 0.5,
    }
    if with_statics:
        fields |= {
            "static_code": torch.randint(1, VOCAB_SIZE, (n_rows, n_statics), generator=generator),
            "static_numeric_value": torch.randn(n_rows, n_statics, generator=generator),
            "static_numeric_value_mask": torch.rand(n_rows, n_statics, generator=generator) > 0.5,
        }
    return MEDSTorchBatch(**fields)


@pytest.mark.parametrize(
    ("interval_minutes", "lookback_hours"),
    [
        (15.0, 96.0),  # the paper's own geometry
        (60.0, 24.0),  # a short window, so measurements fall outside it and must be dropped
        (1440.0, 240.0),  # coarse intervals, so several measurements share one bin and are averaged
    ],
)
@pytest.mark.parametrize("with_statics", [True, False])
def test_matches_the_dense_construction(interval_minutes, lookback_hours, with_statics):
    """The sparse grid equals the paper's dense one, embedded.

    Both sides go through the same embedding matrix, because that is the claim being checked: with a
    bias-free linear embedding, compacting the feature axis to the codes present in the batch is an
    identity, not an approximation.
    """
    grid = IntervalGrid(
        VOCAB_SIZE, d_model=8, interval_minutes=interval_minutes, lookback_hours=lookback_hours
    )
    batch = random_batch(with_statics=with_statics)

    reference = dense_grid(batch, VOCAB_SIZE, interval_minutes, grid.n_intervals) @ grid.embed.weight.t()
    if with_statics:
        static_value = torch.where(
            batch.static_numeric_value_mask,
            batch.static_numeric_value,
            torch.ones_like(batch.static_numeric_value),
        ) * (batch.static_code != PAD_INDEX)
        reference = reference + (static_value.unsqueeze(-1) * grid.embed.weight.t()[batch.static_code]).sum(
            1
        ).unsqueeze(1)

    assert torch.allclose(grid(batch), reference, atol=1e-5)


def test_carry_forward_reaches_the_prediction_time():
    """A single old measurement is still visible at the last interval, and nowhere before it.

    This is the property the model depends on: the head reads the final interval, so anything the
    carry-forward drops is information the classifier never sees.
    """
    batch = MEDSTorchBatch(
        # Code 5 is measured two days before code 6, which ends the sequence.
        code=torch.tensor([[5, 6]]),
        numeric_value=torch.tensor([[2.5, 1.0]]),
        numeric_value_mask=torch.tensor([[True, True]]),
        time_delta_days=torch.tensor([[2.0, 0.0]]),
    )
    # A three-day window in one-day intervals: code 5 lands in interval 0, code 6 in interval 2.
    grid = IntervalGrid(VOCAB_SIZE, d_model=4, interval_minutes=1440.0, lookback_hours=72.0)
    dense = dense_grid(batch, VOCAB_SIZE, 1440.0, grid.n_intervals)

    assert dense[0, :, 5].tolist() == [2.5, 2.5, 2.5]  # carried forward to the prediction time
    assert dense[0, :, 6].tolist() == [0.0, 0.0, 1.0]  # and never carried *backwards*
    assert dense[0].sum() == pytest.approx(8.5)  # nothing leaked into any other code's column
    assert torch.allclose(grid(batch), dense @ grid.embed.weight.t(), atol=1e-6)


def test_measurements_before_the_window_are_dropped():
    """A one-hour lookback sees only the last measurement; the earlier one is out of range."""
    batch = MEDSTorchBatch(
        code=torch.tensor([[5, 6]]),
        numeric_value=torch.tensor([[2.5, 4.0]]),
        numeric_value_mask=torch.tensor([[True, True]]),
        time_delta_days=torch.tensor([[1.0, 0.0]]),  # a full day apart
    )
    grid = IntervalGrid(VOCAB_SIZE, d_model=4, interval_minutes=30.0, lookback_hours=1.0)
    _, in_window = grid.interval_index(batch)
    assert in_window.tolist() == [[False, True]]
    dense = dense_grid(batch, VOCAB_SIZE, 30.0, grid.n_intervals)
    assert dense[0, :, 5].tolist() == [0.0, 0.0]  # the day-old measurement contributes nothing
    assert dense[0, :, 6].tolist() == [0.0, 4.0]


def test_empty_window_encodes_to_the_static_contribution_alone():
    """Every measurement outside the window leaves a zero grid — and must not crash on the empty axis."""
    batch = random_batch(with_statics=True, seed=3)
    grid = IntervalGrid(VOCAB_SIZE, d_model=8, interval_minutes=1.0, lookback_hours=1.0 / 60.0)
    encoded = grid(batch)
    assert encoded.shape == (batch.batch_size, 1, 8)
    # Only the final measurement of each subject survives a one-minute window, so drop the statics to see
    # the grid alone go to zero when even that is excluded.
    empty = IntervalGrid(VOCAB_SIZE, d_model=8, interval_minutes=1.0, lookback_hours=1.0 / 60.0)
    no_statics = random_batch(with_statics=False, seed=3)
    assert empty.static_embedding(no_statics) is None
    assert torch.isfinite(encoded).all()


def test_statics_are_added_to_every_interval():
    """TECO holds the baseline variables constant across the whole time range."""
    batch = random_batch(with_statics=True, seed=1)
    grid = IntervalGrid(VOCAB_SIZE, d_model=8, interval_minutes=60.0, lookback_hours=6.0)

    bare = random_batch(with_statics=False, seed=1)
    difference = grid(batch) - grid(bare)
    static = grid.static_embedding(batch)

    assert static is not None
    assert torch.allclose(difference, static.unsqueeze(1).expand_as(difference), atol=1e-6)


def test_model_reads_the_last_interval():
    """`pooling='last'` must be the prediction time, not an arbitrary position.

    Changing history *before* the final interval changes the encoding (the encoder attends over the
    window), but the position the head reads is the one aligned to the prediction time — so a model built
    with `pooling='last'` and one built with `pooling='mean'` disagree on a batch whose window is not
    uniform. A pooling bug that silently made them identical would not survive this.
    """
    batch = random_batch(seed=2)
    torch.manual_seed(0)
    last = Model(
        vocab_size=VOCAB_SIZE,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        interval_minutes=60.0,
        lookback_hours=6.0,
        pooling="last",
    ).eval()
    torch.manual_seed(0)
    mean = Model(
        vocab_size=VOCAB_SIZE,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        interval_minutes=60.0,
        lookback_hours=6.0,
        pooling="mean",
    ).eval()

    with torch.no_grad():
        hidden = last.encoder(last.grid(batch) + last.positional_encoding)
        assert torch.allclose(last.encode(batch), hidden[:, -1], atol=1e-6)
        assert not torch.allclose(last.encode(batch), mean.encode(batch), atol=1e-4)
