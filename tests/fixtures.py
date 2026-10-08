"""A small synthetic MEDS dataset and task, used by the local and MEDS-DEV gates.

The fixture is deliberately *code-only*: RETAIN consumes a stream of medical codes and this port is
rendered predicate-free, so the fixture contains diagnosis, procedure, laboratory and administrative codes
shaped like MIMIC-IV in MEDS form and nothing else. It contains no real patient data and is generated
deterministically.

Three shapes are planted on purpose because they are the ones that break a sequence model:

* subjects with a single visit, so the attention softmax has to normalise over one position;
* keys whose prediction time precedes every event the subject has, giving an empty history;
* codes that occur only in the held-out split, exercising the closed training vocabulary.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

SPLITS = ("train", "tuning", "held_out")

DIAGNOSIS_CODES = [f"DIAGNOSIS//ICD//10//{code}" for code in ("I509", "E119", "N179", "J189", "I10")]
RISK_CODES = [f"DIAGNOSIS//ICD//10//{code}" for code in ("R579", "J960", "N170")]
PROCEDURE_CODES = [f"PROCEDURE//ICD//10//{code}" for code in ("5A1955Z", "0BH17EZ")]
LAB_CODES = ["LAB//50912//mg/dL", "LAB//50971//mEq/L", "LAB//51301//K/uL"]
RARE_HELD_OUT_CODE = "DIAGNOSIS//ICD//10//Z9989"

ADMISSION_CODE = "HOSPITAL_ADMISSION//EW EMER.//ED"
DISCHARGE_CODE = "HOSPITAL_DISCHARGE//HOME"
DEATH_CODE = "MEDS_DEATH"


def write_meds_dataset(root: Path, n_subjects: int = 60, n_shards: int = 2, seed: int = 19) -> Path:
    """Write a split-sharded MEDS dataset and return its root directory."""
    generator = random.Random(seed)
    root = Path(root)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    assignments: list[tuple[int, str]] = []
    shards: dict[str, list[dict]] = {}
    for index in range(n_subjects):
        subject_id = 2_000 + index
        split = SPLITS[0] if index % 5 < 3 else SPLITS[1] if index % 5 == 3 else SPLITS[2]
        assignments.append((subject_id, split))
        rows = shards.setdefault(f"{split}/{index % n_shards}", [])

        severity = generator.random()
        # Every fifth subject has one visit only. Every eleventh has no record at all before its first
        # admission, which combined with the extra early label row in `write_task_labels` produces a key
        # with an empty history. Both are real cohorts and both are where padding logic fails.
        n_visits = 1 if index % 5 == 0 else generator.randint(2, 6)
        no_prior_record = index % 11 == 0
        first_admission = datetime(2021, 1, 1) + timedelta(days=3 * index)

        rows.append({"subject_id": subject_id, "time": None, "code": "GENDER//" + generator.choice("MF")})
        if not no_prior_record:
            rows.append(
                {"subject_id": subject_id, "time": datetime(1945 + index % 45, 6, 1), "code": "MEDS_BIRTH"}
            )

        admission = first_admission
        for visit in range(n_visits):
            rows.append({"subject_id": subject_id, "time": admission, "code": ADMISSION_CODE})
            for code in generator.sample(DIAGNOSIS_CODES, generator.randint(1, 3)):
                rows.append({"subject_id": subject_id, "time": admission, "code": code})
            if severity > 0.55:
                for code in generator.sample(RISK_CODES, generator.randint(1, len(RISK_CODES))):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "time": admission + timedelta(hours=2),
                            "code": code,
                        }
                    )
            for code in generator.sample(LAB_CODES, generator.randint(1, len(LAB_CODES))):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": admission + timedelta(hours=4),
                        "code": code,
                        "numeric_value": 1.0 + 4.0 * severity,
                    }
                )
            if generator.random() < 0.4:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": admission + timedelta(hours=6),
                        "code": generator.choice(PROCEDURE_CODES),
                    }
                )
            if split == "held_out" and visit == 0:
                rows.append({"subject_id": subject_id, "time": admission, "code": RARE_HELD_OUT_CODE})
            rows.append(
                {"subject_id": subject_id, "time": admission + timedelta(days=1), "code": DISCHARGE_CODE}
            )
            admission = admission + timedelta(days=generator.randint(2, 40))

        outcome = admission + timedelta(days=2)
        if severity > 0.62:
            rows.append({"subject_id": subject_id, "time": outcome, "code": DEATH_CODE})

    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.Utf8,
        "numeric_value": pl.Float32,
    }
    for shard, rows in shards.items():
        path = root / "data" / f"{shard}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.from_dicts([{"numeric_value": None, **row} for row in rows], schema=schema)
        frame.sort("subject_id", "time", nulls_last=False).write_parquet(path)

    pl.DataFrame(assignments, schema=["subject_id", "split"], orient="row").write_parquet(
        root / "metadata" / "subject_splits.parquet"
    )
    return root


def write_task_labels(dataset_dir: Path, labels_dir: Path) -> Path:
    """Derive a mortality-shaped label per subject, laid out like a MEDS-DEV task output.

    The prediction time is 24 hours after the subject's *last* admission and the label is whether a death
    event follows, which reproduces the structure of an ACES task index -- a `prediction_time`, a
    `boolean_value`, sharded by split -- without claiming to be one. Anchoring on the last admission
    rather than the first is what gives the gate multi-visit histories to attend over.

    Subjects with no birth event get a second, earlier key one day *before* their first admission. A task
    may legitimately trigger on a patient with nothing recorded yet, and that key is the one that proves
    an empty history is scored rather than excluded.
    """
    labels_dir = Path(labels_dir)
    events = pl.read_parquet(sorted((Path(dataset_dir) / "data").rglob("*.parquet")))
    admissions = (
        events.filter(pl.col("code") == ADMISSION_CODE)
        .group_by("subject_id")
        .agg(
            pl.col("time").min().alias("first_admission"),
            pl.col("time").max().alias("admission"),
        )
    )
    deaths = (
        events.filter(pl.col("code") == DEATH_CODE)
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("death_time"))
    )
    splits = pl.read_parquet(Path(dataset_dir) / "metadata" / "subject_splits.parquet")
    scored = admissions.join(deaths, on="subject_id", how="left").with_columns(
        pl.col("death_time").is_not_null().alias("boolean_value")
    )
    with_birth_record = (
        events.filter(pl.col("code") == "MEDS_BIRTH").get_column("subject_id").unique().to_list()
    )
    early = (
        scored.filter(~pl.col("subject_id").is_in(with_birth_record))
        .with_columns(
            (pl.col("first_admission") - pl.duration(days=1)).alias("prediction_time"),
            pl.lit(False).alias("boolean_value"),
        )
        .select("subject_id", "prediction_time", "boolean_value")
    )
    labels = (
        pl.concat(
            [
                scored.with_columns(
                    (pl.col("admission") + pl.duration(hours=24)).alias("prediction_time")
                ).select("subject_id", "prediction_time", "boolean_value"),
                early,
            ]
        )
        .join(splits, on="subject_id", how="left")
        .select("subject_id", "prediction_time", "boolean_value", "split")
        .sort("subject_id", "prediction_time")
    )
    for split in labels.get_column("split").unique().to_list():
        path = labels_dir / f"{split}/0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        labels.filter(pl.col("split") == split).drop("split").write_parquet(path)
    return labels_dir


#: Small enough for a gate to run in seconds while exercising every code path, including the Eq. (5)
#: interpretation pass. Everything not named here keeps the official `retain.py` default.
SMOKE_HYPERPARAMETERS = {
    "embed_size": 16,
    "alpha_hidden_dim_size": 16,
    "beta_hidden_dim_size": 16,
    "batch_size": 8,
    "n_epochs": 3,
    "interpretation_top_k": 3,
}
