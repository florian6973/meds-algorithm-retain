"""Required model-owned proof that the implementation performs its primary behavior.

RETAIN's primary behavior is to *represent a medical code* and let it drive the prediction, with the
visit-level attention deciding which visit that code mattered in. So proving the behavior needs three
things, not one:

1. **Binding.** The signal code must actually reach the model's input. This port learns a closed,
   frequency-filtered vocabulary from the training split, so a code can be silently dropped by
   `min_code_count`/`max_vocab_size`, by the MEDS-standard-marker exclusion, or by falling outside the
   prediction-time window. The test proves `SIGNAL//POS` survives into the vocabulary *and* into the
   multi-hot input of exactly the positive subjects, before any claim about learning.
2. **Response.** The real `supervised` profile is run and the held-out AUC is asserted.
3. **Negative control.** The identical cohort with shuffled labels must *not* produce that AUC, which is
   what separates "the model learned the code" from "the fixture leaked".

`tests/signal.py` randomises the signal's position and the history length independently of the label, so
a model cannot win by exploiting position or length. That is the same defence this port needed when label
row order was reaching the fit through `len_argsort`.

The cohort is bucketed at `visit_bucket_minutes=60` rather than the port's 1440 default. `signal.py`
spaces its events one hour apart, so a daily bucket would collapse every subject to a *single* visit and
the test would never exercise the visit attention that distinguishes RETAIN from a bag-of-codes model.
The configuration is the model's own, not a change to the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from retain.commands import COMMANDS  # noqa: E402
from retain.config import RetainConfig  # noqa: E402
from retain.labels import read_labels  # noqa: E402
from retain.training import roc_auc  # noqa: E402
from retain.windows import PatientStore, build_sequences  # noqa: E402

from .signal import SIGNAL_CODE, build_signal_dataset  # noqa: E402

#: One visit per hourly event, so the signal sits at a random *visit* index rather than being folded into
#: a single multi-hot bag. Everything else is a small but real RETAIN.
SETTINGS = {
    "visit_bucket_minutes": 60,
    "embed_size": 32,
    "alpha_hidden_dim_size": 32,
    "beta_hidden_dim_size": 32,
    "batch_size": 32,
    "n_epochs": 40,
    "device": "cpu",
}


def _run_profile(root: Path, *, shuffle_labels: bool):
    """Run the real preprocess -> supervised_train -> predict profile on the signal cohort."""
    root = Path(root)
    dataset = build_signal_dataset(root / "meds", shuffle_labels=shuffle_labels)
    labels_dir = Path(dataset) / "task_labels" / "signal_task"
    data_dir, model_dir, predictions_dir = root / "data", root / "model", root / "predictions"

    COMMANDS["preprocess_data"]()(
        {
            "external_meds_dir": dataset,
            "output_data_dir": data_dir,
            "do_overwrite": True,
            "visit_bucket_minutes": SETTINGS["visit_bucket_minutes"],
        }
    )
    COMMANDS["supervised_train"]()(
        {
            "input_data_dir": data_dir,
            "external_labels_dir": labels_dir,
            "output_supervised_model_dir": model_dir,
            "do_overwrite": True,
            **SETTINGS,
        }
    )
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": model_dir,
            "external_labels_dir": labels_dir,
            "output_predictions_dir": predictions_dir,
            "splits": "[held_out]",
            "do_overwrite": True,
        }
    )
    predictions = pl.read_parquet(predictions_dir / "predictions.parquet")
    return {
        "dataset": Path(dataset),
        "labels_dir": labels_dir,
        "store": PatientStore(data_dir),
        "predictions": predictions,
        "auc": roc_auc(
            predictions.get_column("boolean_value").to_numpy(),
            predictions.get_column("predicted_boolean_probability").to_numpy(),
        ),
    }


def _held_out_auc(result) -> float:
    assert result["auc"] is not None, "held-out split carries only one class; AUC is undefined"
    return float(result["auc"])


@pytest.fixture(scope="module")
def positive(tmp_path_factory):
    return _run_profile(tmp_path_factory.mktemp("signal-positive"), shuffle_labels=False)


@pytest.fixture(scope="module")
def shuffled(tmp_path_factory):
    return _run_profile(tmp_path_factory.mktemp("signal-shuffled"), shuffle_labels=True)


def test_the_signal_code_is_bound_to_the_model_input(positive):
    """Binding, at both levels this port can drop a code at.

    Vocabulary membership alone would not be enough: a code can be in the vocabulary and still never
    reach a sequence, because the history is cut at the prediction time and bucketed afterwards.
    """
    store = positive["store"]
    assert SIGNAL_CODE in store.code_strings(), (
        f"{SIGNAL_CODE} was dropped before the model ever saw it; a learning assertion after this point "
        "would be measuring something else"
    )
    signal_index = store.code_strings().index(SIGNAL_CODE)

    labels = read_labels(positive["labels_dir"])
    config = RetainConfig(**json.loads((Path(store.root) / "config.json").read_text()))
    batch = build_sequences(store, labels.select("subject_id", "prediction_time"), config)
    truth = labels.get_column("boolean_value").to_numpy()

    carries_signal = np.zeros(batch.n_keys, dtype=bool)
    for row in range(batch.n_keys):
        start, stop = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
        codes = batch.codes[int(batch.visit_ptr[start]) : int(batch.visit_ptr[stop])]
        carries_signal[row] = bool((codes == signal_index).any())

    # The fixture defines the signal as present exactly for positives, so this is an equality, not a
    # correlation: it proves the windowing carried the code through without adding or losing any.
    assert np.array_equal(carries_signal, truth.astype(bool))
    assert batch.lengths.min() >= 2, "hourly bucketing must produce multi-visit sequences"


def test_the_signal_occupies_a_random_visit_position(positive):
    """The fixture's defence against a position or length leak, verified rather than assumed."""
    store = positive["store"]
    signal_index = store.code_strings().index(SIGNAL_CODE)
    labels = read_labels(positive["labels_dir"])
    config = RetainConfig(**json.loads((Path(store.root) / "config.json").read_text()))
    batch = build_sequences(store, labels.select("subject_id", "prediction_time"), config)
    truth = labels.get_column("boolean_value").to_numpy().astype(bool)

    positions, lengths = [], []
    for row in range(batch.n_keys):
        start, stop = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
        lengths.append(stop - start)
        for visit in range(stop - start):
            index = start + visit
            codes = batch.codes[int(batch.visit_ptr[index]) : int(batch.visit_ptr[index + 1])]
            if (codes == signal_index).any():
                positions.append(visit)
                break
    assert len(set(positions)) > 1, "the signal must not sit at a fixed visit index"
    # Sequence length must not separate the classes, or a length-counting model would pass.
    lengths = np.array(lengths)
    assert abs(lengths[truth].mean() - lengths[~truth].mean()) < 1.0


def test_primary_behavior(positive, shuffled):
    """The required proof: bound signal, real profile, learned response, shuffled-label control.

    The AUC threshold is a property of a synthetic cohort in which one code determines the label by
    construction. It is not a performance claim about RETAIN on clinical data, and nothing in this file
    should be read as one.

    Measured on this cohort at these settings: signal 0.937, shuffled 0.460, gap +0.477. The thresholds
    sit clear of those values rather than against them. The signal AUC saturates below 1.0 for a reason
    that is faithful rather than accidental -- `alpha` is a softmax over visits and the official `* 0.5`
    scaling halves its logits, so attention cannot fully concentrate on the one visit that carries the
    code. Ten epochs already reach 0.907, so the assertion is not sensitive to the epoch budget.
    """
    signal_auc = _held_out_auc(positive)
    control_auc = _held_out_auc(shuffled)

    assert signal_auc > 0.85, (
        f"held-out AUC {signal_auc:.3f} on a cohort where {SIGNAL_CODE} alone determines the label: "
        "the model is not responding to a code it demonstrably received"
    )
    assert control_auc < 0.75, (
        f"shuffled-label control reached AUC {control_auc:.3f}; the cohort leaks something other than "
        "the signal code, so the positive result above does not mean what it appears to"
    )
    assert signal_auc - control_auc > 0.25

    # Coverage still holds on this cohort, so the AUC is computed over the whole requested index.
    requested = read_labels(positive["labels_dir"])
    written = positive["predictions"]
    held_out = requested.join(
        written.select("subject_id", "prediction_time"), on=["subject_id", "prediction_time"], how="semi"
    )
    assert held_out.height == written.height > 0


def test_the_model_attributes_its_score_to_the_signal_code(positive, tmp_path):
    """RETAIN's distinguishing claim: its own Eq. (5) attribution should name the code it learned.

    A model can reach a high AUC while attributing it to the wrong feature. This is the check that the
    interpretability output -- the reason to choose RETAIN over a plain RNN -- points at the real cause.
    """
    predictions_dir = tmp_path / "interpreted"
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": Path(positive["store"].root).parent / "model",
            "external_labels_dir": positive["labels_dir"],
            "output_predictions_dir": predictions_dir,
            "splits": "[held_out]",
            "interpretation_top_k": 1,
            "do_overwrite": True,
        }
    )
    records = [
        json.loads(line) for line in (predictions_dir / "interpretation.ndjson").read_text().splitlines()
    ]
    scored = {
        (row["subject_id"], row["prediction_time"]): row
        for row in positive["predictions"]
        .with_columns(pl.col("prediction_time").dt.strftime("%Y-%m-%dT%H:%M:%S"))
        .to_dicts()
    }
    top_is_signal = 0
    positives = 0
    for record in records:
        label = scored[(record["subject_id"], record["prediction_time"])]["boolean_value"]
        if not label or not record["top_contributions"]:
            continue
        positives += 1
        if record["top_contributions"][0]["code"] == SIGNAL_CODE:
            top_is_signal += 1
    assert positives > 0
    assert top_is_signal / positives > 0.8, (
        f"{SIGNAL_CODE} was the top-attributed code for only {top_is_signal}/{positives} positive keys; "
        "the model reaches the right answer but explains it with the wrong code"
    )
