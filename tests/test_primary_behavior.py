"""Required algorithm-owned proof that the implementation performs its primary behavior.

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

The core signal fixture randomises the signal's position and the history length independently of the
label, so a model cannot win by exploiting position or length.

The cohort is bucketed at `visit_bucket_minutes=60` rather than the port's 1440 default. The fixture
spaces its events one hour apart, so a daily bucket would collapse every subject to a *single* visit and
the test would never exercise the visit attention that distinguishes RETAIN from a bag-of-codes model.
The configuration is the algorithm's own, not a change to the fixture.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from meds_algorithm.conformance.signal import SIGNAL_CODE, build_signal_dataset

from retain_meds._training import roc_auc
from retain_meds._windows import PatientStore, build_sequences

from .e2e import TASK, TEST_CONFIG, run_e2e

#: One visit per hourly event, so the signal sits at a random *visit* index rather than being folded into
#: a single multi-hot bag. Everything else is a small but real RETAIN. The Eq. (5) attribution is written
#: for every scored key so the last test can read it.
CONFIG = replace(
    TEST_CONFIG,
    embed_size=32,
    alpha_hidden_dim_size=32,
    beta_hidden_dim_size=32,
    n_epochs=40,
    interpretation_top_k=1,
)
KEYS = ["subject_id", "prediction_time"]


def _run_profile(root: Path, *, profile: str, shuffle_labels: bool) -> dict:
    """Run the real preprocess -> train -> predict profile on the signal cohort."""
    dataset = build_signal_dataset(Path(root) / "meds", shuffle_labels=shuffle_labels)
    labels_dir = dataset / "task_labels" / TASK
    result = run_e2e(
        Path(root) / "run", profile=profile, external_meds_dir=dataset, external_labels_dir=labels_dir, config=CONFIG
    )
    predictions = pl.read_parquet(result.artifacts["predict_supervised"] / "predictions.parquet")
    return {
        "labels": pl.read_parquet(sorted(labels_dir.glob("*.parquet"))).sort(KEYS),
        "store": PatientStore(result.artifacts["preprocess_data"]),
        "predictions_dir": result.artifacts["predict_supervised"],
        "predictions": predictions,
        "auc": roc_auc(
            predictions["boolean_value"].to_numpy(), predictions["predicted_boolean_probability"].to_numpy()
        ),
    }


@pytest.fixture(scope="module")
def positive(tmp_path_factory, profile):
    return _run_profile(tmp_path_factory.mktemp("signal-positive"), profile=profile, shuffle_labels=False)


@pytest.fixture(scope="module")
def shuffled(tmp_path_factory, profile):
    return _run_profile(tmp_path_factory.mktemp("signal-shuffled"), profile=profile, shuffle_labels=True)


def _signal_visits(batch, signal_index: int) -> list[int | None]:
    """The visit index carrying the signal code for each key, or None."""
    positions: list[int | None] = []
    for row in range(batch.n_keys):
        start, stop = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
        found = None
        for visit in range(start, stop):
            codes = batch.codes[int(batch.visit_ptr[visit]) : int(batch.visit_ptr[visit + 1])]
            if (codes == signal_index).any():
                found = visit - start
                break
        positions.append(found)
    return positions


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
    labels = positive["labels"]
    batch = build_sequences(store, labels.select(KEYS), CONFIG)
    positions = _signal_visits(batch, store.code_strings().index(SIGNAL_CODE))
    carries_signal = np.array([position is not None for position in positions])

    # The fixture defines the signal as present exactly for positives, so this is an equality, not a
    # correlation: it proves the windowing carried the code through without adding or losing any.
    assert np.array_equal(carries_signal, labels["boolean_value"].to_numpy())
    assert batch.lengths.min() >= 2, "hourly bucketing must produce multi-visit sequences"


def test_the_signal_occupies_a_random_visit_position(positive):
    """The fixture's defence against a position or length leak, verified rather than assumed."""
    store = positive["store"]
    labels = positive["labels"]
    batch = build_sequences(store, labels.select(KEYS), CONFIG)
    positions = _signal_visits(batch, store.code_strings().index(SIGNAL_CODE))
    truth = labels["boolean_value"].to_numpy()

    assert len({position for position in positions if position is not None}) > 1, (
        "the signal must not sit at a fixed visit index"
    )
    # Sequence length must not separate the classes, or a length-counting model would pass.
    assert abs(batch.lengths[truth].mean() - batch.lengths[~truth].mean()) < 1.0


def test_primary_behavior(positive, shuffled):
    """The required proof: bound signal, real profile, learned response, shuffled-label control.

    The AUC threshold is a property of a synthetic cohort in which one code determines the label by
    construction. It is not a performance claim about RETAIN on clinical data, and nothing in this file
    should be read as one. The signal AUC saturates below 1.0 for a reason that is faithful rather than
    accidental: `alpha` is a softmax over visits and the official `* 0.5` scaling halves its logits, so
    attention cannot fully concentrate on the one visit that carries the code.
    """
    assert positive["auc"] is not None and shuffled["auc"] is not None, "held-out split carries one class only"
    signal_auc, control_auc = positive["auc"], shuffled["auc"]

    assert signal_auc > 0.85, (
        f"held-out AUC {signal_auc:.3f} on a cohort where {SIGNAL_CODE} alone determines the label: "
        "the model is not responding to a code it demonstrably received"
    )
    assert control_auc < 0.75, (
        f"shuffled-label control reached AUC {control_auc:.3f}; the cohort leaks something other than "
        "the signal code, so the positive result above does not mean what it appears to"
    )
    assert signal_auc - control_auc > 0.25


def test_the_model_attributes_its_score_to_the_signal_code(positive):
    """RETAIN's distinguishing claim: its own Eq. (5) attribution should name the code it learned.

    A model can reach a high AUC while attributing it to the wrong feature. This is the check that the
    interpretability output (the reason to choose RETAIN over a plain RNN) points at the real cause.
    """
    records = [
        json.loads(line)
        for line in (positive["predictions_dir"] / "interpretation.ndjson").read_text().splitlines()
    ]
    scored = {
        (row["subject_id"], row["prediction_time"]): row["boolean_value"]
        for row in positive["predictions"]
        .with_columns(pl.col("prediction_time").dt.strftime("%Y-%m-%dT%H:%M:%S"))
        .to_dicts()
    }
    top_is_signal = positives = 0
    for record in records:
        if not scored[(record["subject_id"], record["prediction_time"])] or not record["top_contributions"]:
            continue
        positives += 1
        top_is_signal += record["top_contributions"][0]["code"] == SIGNAL_CODE
    assert positives > 0
    assert top_is_signal / positives > 0.8, (
        f"{SIGNAL_CODE} was the top-attributed code for only {top_is_signal}/{positives} positive keys; "
        "the model reaches the right answer but explains it with the wrong code"
    )
