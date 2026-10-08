"""A run must reproduce from the *content* of its inputs, not from how their files are laid out.

SPEC section 5 says row order is never meaningful. The task index is written by ACES through operations
with no ordering guarantee, so the same task can be materialised twice with identical content in a
different row order. Anything that lets that difference reach the fit makes a result depend on a property
the schema says carries no information.

Rebuilding a fixture twice does not test this: a fixture that writes its labels in a canonical order
reproduces its own byte layout and the check passes while the defect is still there. These tests therefore
permute row order directly, holding content fixed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from retain.commands import COMMANDS  # noqa: E402
from retain.labels import read_labels  # noqa: E402

from .fixtures import SMOKE_HYPERPARAMETERS, write_meds_dataset, write_task_labels  # noqa: E402

PROBABILITY = "predicted_boolean_probability"
KEYS = ["subject_id", "prediction_time"]


def _permuted_copy(labels_dir, destination, seed=99):
    """The same label rows, same shards, shuffled within each file."""
    for source in sorted(labels_dir.rglob("*.parquet")):
        target = destination / source.relative_to(labels_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.read_parquet(source).sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(target)
    return destination


def _run(meds_dir, labels_dir, out):
    COMMANDS["preprocess_data"]()(
        {"external_meds_dir": meds_dir, "output_data_dir": out / "data", "do_overwrite": True}
    )
    COMMANDS["supervised_train"]()(
        {
            "input_data_dir": out / "data",
            "external_labels_dir": labels_dir,
            "output_supervised_model_dir": out / "model",
            "do_overwrite": True,
            **SMOKE_HYPERPARAMETERS,
            "device": "cpu",
        }
    )
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": out / "model",
            "external_labels_dir": labels_dir,
            "output_predictions_dir": out / "predictions",
            "do_overwrite": True,
            "device": "cpu",
        }
    )
    return pl.read_parquet(out / "predictions" / "predictions.parquet").sort(KEYS)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("retain-order")
    meds_dir = write_meds_dataset(root / "meds")
    labels_dir = write_task_labels(meds_dir, root / "labels")
    return root, meds_dir, labels_dir


def test_permuting_label_row_order_does_not_change_predictions(cohort):
    """The real check: identical content, different row order, at a fixed seed."""
    root, meds_dir, labels_dir = cohort
    shuffled = _permuted_copy(labels_dir, root / "labels-shuffled")

    original_rows = pl.read_parquet(sorted(labels_dir.rglob("*.parquet"))).sort(KEYS)
    shuffled_rows = pl.read_parquet(sorted(shuffled.rglob("*.parquet"))).sort(KEYS)
    assert original_rows.equals(shuffled_rows), "the permuted copy must differ only in row order"

    first = _run(meds_dir, labels_dir, root / "run-original")
    second = _run(meds_dir, shuffled, root / "run-shuffled")
    assert first.equals(second)


def test_read_labels_imposes_a_canonical_order(cohort):
    """The fix, at the boundary where the ordering enters."""
    root, _, labels_dir = cohort
    shuffled = _permuted_copy(labels_dir, root / "labels-shuffled-read", seed=7)
    assert read_labels(labels_dir).equals(read_labels(shuffled))
    ordered = read_labels(labels_dir)
    assert ordered.equals(ordered.sort(KEYS))


def test_minibatch_composition_does_not_depend_on_row_order(cohort):
    """Equal-length sequences are common, so tie resolution is the path that carries row order in.

    Checked on the batching function itself rather than only end to end, because a fit that happens to
    agree on one fixture would otherwise hide a positional tiebreak.
    """
    from retain.config import RetainConfig
    from retain.training import _length_sorted
    from retain.windows import PatientStore, build_sequences

    root, meds_dir, labels_dir = cohort
    data_dir = root / "batching-data"
    if not data_dir.exists():
        COMMANDS["preprocess_data"]()(
            {"external_meds_dir": meds_dir, "output_data_dir": data_dir, "do_overwrite": True}
        )
    store = PatientStore(data_dir)
    config = RetainConfig()

    keys = read_labels(labels_dir).select(KEYS)
    permuted = keys.sample(fraction=1.0, shuffle=True, seed=5)
    forward = build_sequences(store, keys, config)
    scrambled = build_sequences(store, permuted, config)

    def ordered_keys(batch):
        return batch.keys[_length_sorted(batch)]

    assert np.array_equal(ordered_keys(forward), ordered_keys(scrambled))


def test_rebuilding_the_fixture_reproduces_it_exactly(tmp_path):
    """Weaker than the permutation test above, and kept only to pin fixture determinism itself."""
    first_meds = write_meds_dataset(tmp_path / "a" / "meds")
    second_meds = write_meds_dataset(tmp_path / "b" / "meds")
    first = write_task_labels(first_meds, tmp_path / "a" / "labels")
    second = write_task_labels(second_meds, tmp_path / "b" / "labels")
    assert pl.read_parquet(sorted(first.rglob("*.parquet"))).equals(
        pl.read_parquet(sorted(second.rglob("*.parquet")))
    )


def test_vocabulary_selection_breaks_frequency_ties_on_the_code_string(tmp_path):
    """`max_vocab_size` must cut on content, not on which shard was read first."""
    from datetime import datetime

    from retain.config import RetainConfig
    from retain.preprocess import preprocess
    from retain.windows import PatientStore

    rows = [
        {"subject_id": 1, "time": datetime(2020, 1, 1), "code": code, "numeric_value": None}
        for code in ("B_TIE", "A_TIE", "C_TIE")
    ]
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.Utf8,
        "numeric_value": pl.Float32,
    }
    selected = []
    for index, order in enumerate([rows, list(reversed(rows))]):
        root = tmp_path / f"meds{index}"
        (root / "data" / "train").mkdir(parents=True)
        (root / "metadata").mkdir(parents=True)
        pl.from_dicts(order, schema=schema).write_parquet(root / "data" / "train" / "0.parquet")
        pl.DataFrame({"subject_id": [1], "split": ["train"]}).write_parquet(
            root / "metadata" / "subject_splits.parquet"
        )
        data_dir = tmp_path / f"data{index}"
        data_dir.mkdir()
        preprocess(root, data_dir, RetainConfig(max_vocab_size=2))
        selected.append(PatientStore(data_dir).code_strings())
    # All three codes occur once, so only the tiebreak decides which two survive the cut.
    assert selected[0] == selected[1] == ["A_TIE", "B_TIE"]
