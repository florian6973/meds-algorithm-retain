"""Tier 3 — does the model actually learn? (``@slow``)

The flagship correctness check, and the one that cannot be faked: build a synthetic MEDS dataset with a
*designed* signal, run this model's whole DAG over it, and assert the signal is learned — paired with a
**negative control** on shuffled labels so the test cannot pass vacuously. A model that scores well on both
has learned nothing; the pair is the point.

This is DAG-independent: `run_chain` follows whatever `commands.py` registers, so the same test applies to
a supervised model, a fine-tune, a probe and a zero-shot model alike.

Deselected in the fast CI job (`-m "not slow"`); run it before you claim the model works.
"""

import random
from pathlib import Path

import polars as pl
import pytest
from meds_model_base.schemas import code_metadata_filepath
from meds_model_base.testing import (
    assert_learns_signal,
    binary_auroc,
    build_signal_dataset,
    build_workspace,
    skip_if_stub,
)

from teco.model import Model

pytestmark = pytest.mark.slow

#: Baseline (static) variables added to the synthetic cohort — see :func:`add_baseline_variables`. One
#: value-less code and one numeric one, so both branches of ``IntervalGrid.static_embedding`` are covered.
BASELINE_GROUPS = ("BASELINE//GROUP_A", "BASELINE//GROUP_B")
BASELINE_AGE = "BASELINE//AGE"


def add_baseline_variables(root: Path, seed: int) -> Path:
    """Give every synthetic subject the static measurements TECO's baseline variables arrive as.

    ``build_signal_dataset`` writes a cohort with **no static measurements at all**, and TECO reads them:
    ``configs/profile/default.yaml`` sets ``static_inclusion_mode: INCLUDE`` because the paper concatenates
    age, sex, ethnicity and race onto every interval. Those two facts do not compose — meds-torch-data's
    collate fails with ``ValueError: Cannot infer dtype from empty values`` when the static tensors are
    empty for the whole batch, so the DAG never reaches the model.

    Fixing that here rather than dropping the baseline variables is deliberate: ``docs/PORTING-A-MODEL.md``
    step 4 lists "it breaks the test fixture" as an illegitimate reason to omit a source element. The
    crash is recorded as an upstream finding in ``IMPLEMENTATION_REPORT.md``.

    The assignment is drawn from a generator independent of the one that decides the labels, so the
    baseline variables carry no information about the outcome — the designed signal stays the only thing
    there is to learn, and the negative control stays a control.
    """
    rng = random.Random(seed + 97)
    for shard in sorted((root / "data").glob("*/*.parquet")):
        events = pl.read_parquet(shard)
        statics = pl.DataFrame(
            [
                row
                for subject in events["subject_id"].unique().sort().to_list()
                for row in (
                    {
                        "subject_id": subject,
                        "time": None,
                        "code": rng.choice(BASELINE_GROUPS),
                        "numeric_value": None,
                    },
                    {
                        "subject_id": subject,
                        "time": None,
                        "code": BASELINE_AGE,
                        "numeric_value": rng.uniform(20.0, 90.0),
                    },
                )
            ],
            schema=events.schema,
        )
        # MEDS puts static measurements (null time) first within each subject.
        pl.concat([statics, events]).sort("subject_id", "time", nulls_last=False).write_parquet(shard)

    codes_fp = root / code_metadata_filepath
    codes = pl.read_parquet(codes_fp)
    added = [*BASELINE_GROUPS, BASELINE_AGE]
    extra = pl.DataFrame({"code": added, "description": added}, schema=codes.schema)
    pl.concat([codes, extra]).write_parquet(codes_fp)
    return root


def _predictions_for(chain, tmp_path, name: str, *, shuffle_labels: bool, seed: int):
    """Build a synthetic dataset, run the DAG over it, and return (predictions, held-out labels)."""
    raw = build_signal_dataset(tmp_path / f"raw_{name}", seed=seed, shuffle_labels=shuffle_labels)
    add_baseline_variables(raw, seed)
    labels_dir = raw / "task_labels" / "signal_task"
    data_dir = build_workspace(raw, tmp_path / f"data_{name}")
    artifacts = chain(data_dir, labels_dir, tmp_path / f"out_{name}", epochs=25, batch_size=16)
    assert "predictions" in artifacts, "this DAG does not register `predict`, so it cannot be evaluated"
    return (
        pl.read_parquet(artifacts["predictions"] / "predictions.parquet"),
        pl.read_parquet(raw / "task_labels" / "signal_task" / "held_out.parquet"),
    )


def test_learns_designed_signal(chain, tmp_path):
    skip_if_stub(Model)
    preds, held = _predictions_for(chain, tmp_path, "signal", shuffle_labels=False, seed=0)
    # The signal is perfectly separable, so a working model reaches ≈1.0 — a bar much below that would
    # pass a model that learned the signal only partially.
    auroc = assert_learns_signal(preds, held, min_auroc=0.9)
    print(f"\ndesigned signal: held-out AUROC = {auroc:.4f}")
    assert auroc > 0.9


def test_negative_control_does_not_learn(chain, tmp_path):
    """Shuffled labels break the signal-label link, so a sound model must NOT score well here."""
    skip_if_stub(Model)
    preds, held = _predictions_for(chain, tmp_path, "control", shuffle_labels=True, seed=1)
    joined = preds.join(held, on=["subject_id", "prediction_time"], how="inner")
    ctrl = binary_auroc(joined["boolean_value"].to_list(), joined["predicted_boolean_probability"].to_list())
    print(f"\nnegative control: held-out AUROC = {ctrl:.4f}")
    assert ctrl <= 0.70, f"negative control learned a shuffled signal (AUROC={ctrl:.3f}); test is unsound"
