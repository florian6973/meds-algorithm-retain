"""Task index handling.

SPEC section 2.2 says a task input is an index of `(subject_id, prediction_time)` plus, for training and
evaluation only, a value column, and that prediction code must operate on the index without reading ground
truth. `prediction_keys` is therefore the only frame allowed to reach sequence construction, and it carries
no value column at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

KEY_COLUMNS = ("subject_id", "prediction_time")
VALUE_COLUMNS = ("boolean_value", "integer_value", "float_value", "categorical_value")


def read_labels(labels_dir: Path | str) -> pl.DataFrame:
    """Read every parquet under `labels_dir` into one frame with unique keys."""
    directory = Path(labels_dir)
    files = sorted(path for path in directory.rglob("*.parquet") if not path.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"{directory} contains no label parquet files")
    frames = [pl.read_parquet(path, missing_columns="insert") for path in files]
    labels = pl.concat(frames, how="vertical_relaxed")
    missing = [column for column in KEY_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"Labels in {directory} lack required column(s) {missing}")
    labels = labels.with_columns(
        pl.col("subject_id").cast(pl.Int64),
        pl.col("prediction_time").cast(pl.Datetime("us")),
    )
    return labels.unique(subset=list(KEY_COLUMNS), keep="first", maintain_order=True)


def prediction_keys(labels: pl.DataFrame) -> pl.DataFrame:
    """The `(subject_id, prediction_time)` index alone, with all ground truth removed."""
    return labels.select(*KEY_COLUMNS)


def boolean_targets(labels: pl.DataFrame) -> pl.Series:
    """Training targets. Raises when the task carries no usable boolean value column."""
    if "boolean_value" not in labels.columns:
        raise ValueError("supervised_train needs a boolean_value column in the task labels")
    values = labels.get_column("boolean_value")
    if values.null_count():
        raise ValueError(f"{values.null_count()} label rows have a null boolean_value")
    return values.cast(pl.Boolean)


def requested_splits(value: object) -> set[str] | None:
    """Parse the `splits` argument. `None` means 'every split present in the label index'.

    MEDS-DEV expands its command template into `splits=[held_out]`, which is not JSON, so the bare-string
    form has to be understood as well as a real list.
    """
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        value = [part for part in value.strip("[]").split(",") if part.strip()]
    return {str(item).strip() for item in value}
