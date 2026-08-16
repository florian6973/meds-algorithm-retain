"""Turn a MEDS dataset into RETAIN's tokenised code stream.

The official pipeline (`process_mimic.py`) starts from ICD-9 strings, maps each distinct string to an
integer, and emits a list-of-list-of-list of visits. This module does the same thing from the generic MEDS
`code` column: no dataset predicates, no variable map, no charted-vital semantics, nothing beyond "a code
observed at a time for a subject".

Two properties matter and are enforced here:

* The vocabulary is counted over the **training split only**. A vocabulary fitted on held-out subjects
  would leak information about the evaluation cohort into the model's input space.
* Events are stored per `(subject_id, time)`, not per visit. Bucketing into visits happens later, after
  the prediction-time cut, so that no visit can straddle a prediction time and carry future events into
  its own history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .config import RetainConfig

PAYLOAD_FORMAT = "retain-patients/v1"

TRAIN_SPLIT = "train"
TUNING_SPLIT = "tuning"
HELD_OUT_SPLIT = "held_out"

EVENT_COLUMNS = ("subject_id", "time", "code")


def data_files(meds_dir: Path) -> list[Path]:
    root = Path(meds_dir) / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"{meds_dir} is not a MEDS dataset: no data/ directory")
    files = sorted(path for path in root.rglob("*.parquet") if not path.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"{root} contains no parquet shards")
    return files


def read_split_map(meds_dir: Path) -> pl.DataFrame:
    """`(subject_id, split)` from MEDS metadata, falling back to the split-sharded directory layout."""
    meds_dir = Path(meds_dir)
    metadata = meds_dir / "metadata" / "subject_splits.parquet"
    if metadata.is_file():
        frame = pl.read_parquet(metadata)
        if {"subject_id", "split"} <= set(frame.columns):
            return frame.select(
                pl.col("subject_id").cast(pl.Int64), pl.col("split").cast(pl.Utf8)
            ).unique(subset=["subject_id"], keep="first")
    root = meds_dir / "data"
    frames = []
    for path in data_files(meds_dir):
        parts = path.relative_to(root).parts
        split = parts[0] if len(parts) > 1 else TRAIN_SPLIT
        subjects = pl.read_parquet(path, columns=["subject_id"]).unique()
        frames.append(subjects.with_columns(pl.lit(split).alias("split")))
    return (
        pl.concat(frames)
        .select(pl.col("subject_id").cast(pl.Int64), pl.col("split").cast(pl.Utf8))
        .unique(subset=["subject_id"], keep="first")
    )


def _build_vocabulary(files: list[Path], train_subjects: pl.DataFrame, config: RetainConfig) -> pl.DataFrame:
    """Count codes over training subjects and assign integer indices, most frequent first."""
    counts: pl.DataFrame | None = None
    for path in files:
        shard = (
            pl.scan_parquet(path)
            .select("subject_id", "time", "code")
            .filter(pl.col("time").is_not_null())
            .with_columns(pl.col("subject_id").cast(pl.Int64))
            .join(train_subjects.lazy(), on="subject_id", how="semi")
            .group_by("code")
            .agg(pl.len().alias("count"))
            .collect()
        )
        counts = shard if counts is None else pl.concat([counts, shard])
    if counts is None or counts.is_empty():
        raise ValueError(
            "No timestamped events belong to the training split, so RETAIN has no vocabulary to learn. "
            "Check that the MEDS dataset declares a 'train' split."
        )
    counts = counts.group_by("code").agg(pl.col("count").sum())
    counts = counts.filter(pl.col("count") >= config.min_code_count)
    # Deterministic order: frequency first, then the code string, so the same dataset always produces
    # the same indices regardless of shard iteration order.
    counts = counts.sort(["count", "code"], descending=[True, False])
    if config.max_vocab_size is not None:
        counts = counts.head(config.max_vocab_size)
    if counts.is_empty():
        raise ValueError(
            f"min_code_count={config.min_code_count} removed every code from the training split"
        )
    return counts.with_row_index("code_index").select(
        pl.col("code"), pl.col("code_index").cast(pl.UInt32), pl.col("count").cast(pl.Int64)
    )


def preprocess(meds_dir: Path, staging: Path, config: RetainConfig) -> dict[str, Any]:
    """Write the patients payload into `staging` and return its work units."""
    meds_dir = Path(meds_dir)
    files = data_files(meds_dir)
    splits = read_split_map(meds_dir)
    train_subjects = splits.filter(pl.col("split") == TRAIN_SPLIT).select("subject_id")
    if train_subjects.is_empty():
        raise ValueError(
            f"{meds_dir} declares no '{TRAIN_SPLIT}' split; RETAIN fits its vocabulary and its "
            "parameters on the training split only."
        )

    vocabulary = _build_vocabulary(files, train_subjects, config)
    vocabulary.write_parquet(staging / "vocabulary.parquet")
    splits.write_parquet(staging / "subject_splits.parquet")
    lookup = vocabulary.select("code", "code_index")

    sequences = staging / "sequences"
    totals = {"events": 0, "static_events": 0, "out_of_vocabulary_events": 0, "kept_events": 0}
    subjects: set[int] = set()
    event_groups = 0
    for path in files:
        frame = pl.read_parquet(path, columns=list(EVENT_COLUMNS)).with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("time").cast(pl.Datetime("us")),
            pl.col("code").cast(pl.Utf8),
        )
        totals["events"] += frame.height
        timestamped = frame.filter(pl.col("time").is_not_null())
        totals["static_events"] += frame.height - timestamped.height
        mapped = timestamped.join(lookup, on="code", how="inner")
        totals["out_of_vocabulary_events"] += timestamped.height - mapped.height
        # One row per (subject, timestamp, code): RETAIN's input is binary, so a code repeated at the
        # same instant must contribute once, exactly as `xvec[subseq] = 1.` does upstream.
        events = (
            mapped.select("subject_id", "time", "code_index")
            .unique(subset=["subject_id", "time", "code_index"])
            .sort(["subject_id", "time", "code_index"])
        )
        totals["kept_events"] += events.height
        subjects.update(events.get_column("subject_id").unique().to_list())
        event_groups += events.select("subject_id", "time").unique().height
        destination = sequences / path.relative_to(meds_dir / "data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        events.write_parquet(destination)

    (staging / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return {
        "subjects": len(subjects),
        "subjects_in_split_map": splits.height,
        "shards": len(files),
        "vocabulary_size": vocabulary.height,
        "event_groups": event_groups,
        **totals,
    }
