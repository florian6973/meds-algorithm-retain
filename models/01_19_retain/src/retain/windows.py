"""Turn `(subject_id, prediction_time)` keys into RETAIN visit sequences.

Three invariants:

1. **No label leakage.** Nothing here sees a value column; `build_sequences` rejects any frame carrying
   one, and the history is cut at the prediction time.
2. **Keys, never row order.** Every sequence carries its own `(subject_id, prediction_time)` key and is
   associated with a score through that key alone.
3. **Every requested key produces a sequence.** A key whose subject has no history yields an empty
   sequence rather than being dropped, so the published index can cover the requested index exactly.

Sequences are held in a CSR-style ragged layout — `row_ptr` into visits, `visit_ptr` into codes — because
the padded dense form is `n_keys x max_visits x |C|` and is unusable at MIMIC-IV scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .config import RetainConfig
from .labels import KEY_COLUMNS, VALUE_COLUMNS
from .preprocess import PAYLOAD_FORMAT

MICROSECONDS_PER_MINUTE = 60_000_000
MICROSECONDS_PER_DAY = 86_400_000_000


@dataclass(frozen=True)
class SequenceBatch:
    """Visit sequences for a set of keys, in ragged CSR layout."""

    #: `(n_keys, 2)` of `(subject_id, prediction_time in microseconds)`.
    keys: np.ndarray
    #: `(n_keys + 1,)` index into the visit arrays.
    row_ptr: np.ndarray
    #: `(n_visits + 1,)` index into `codes`.
    visit_ptr: np.ndarray
    #: `(n_visits,)` bucket start time in microseconds.
    visit_time: np.ndarray
    #: `(n_codes,)` vocabulary indices.
    codes: np.ndarray

    @property
    def n_keys(self) -> int:
        return int(self.keys.shape[0])

    @property
    def lengths(self) -> np.ndarray:
        return (self.row_ptr[1:] - self.row_ptr[:-1]).astype(np.int64)

    @property
    def n_empty(self) -> int:
        return int((self.lengths == 0).sum())

    def visit_gap_days(self) -> np.ndarray:
        """Days between consecutive visits, zero at each first visit.

        This is the `--time_file` quantity of the official repository, whose README defines it as "the
        duration between consecutive visits" with "the duration for the first visit set to zero".
        """
        gaps = np.zeros(self.visit_time.size, dtype=np.float32)
        if self.visit_time.size:
            deltas = np.diff(self.visit_time) / MICROSECONDS_PER_DAY
            gaps[1:] = deltas
            gaps[self.row_ptr[:-1][self.lengths > 0]] = 0.0
        return gaps


class PatientStore:
    """Read access to a published `retain-patients/v1` artifact."""

    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir)
        self.vocabulary = pl.read_parquet(self.root / "vocabulary.parquet")
        self.splits = pl.read_parquet(self.root / "subject_splits.parquet")
        self.config = json.loads((self.root / "config.json").read_text())
        self.shards = sorted(
            path for path in (self.root / "sequences").rglob("*.parquet") if not path.name.startswith(".")
        )
        if not self.shards:
            raise FileNotFoundError(f"{self.root} declares {PAYLOAD_FORMAT} but holds no sequence shards")

    @property
    def vocabulary_size(self) -> int:
        return self.vocabulary.height

    def split_of(self, subject_ids: pl.Series) -> pl.Series:
        """Split name per subject, `None` for a subject the dataset does not assign."""
        frame = pl.DataFrame({"subject_id": subject_ids.cast(pl.Int64)})
        joined = frame.join(self.splits, on="subject_id", how="left")
        return joined.get_column("split")

    def code_strings(self) -> list[str]:
        ordered = self.vocabulary.sort("code_index")
        return ordered.get_column("code").to_list()


def build_sequences(store: PatientStore, keys: pl.DataFrame, config: RetainConfig) -> SequenceBatch:
    """Build one visit sequence per row of `keys`, which must hold the index columns only."""
    extra = set(keys.columns) - set(KEY_COLUMNS)
    if extra & set(VALUE_COLUMNS) or extra:
        raise ValueError(f"build_sequences must never see label columns; got extra column(s) {sorted(extra)}")

    keys = keys.with_columns(
        pl.col("subject_id").cast(pl.Int64), pl.col("prediction_time").cast(pl.Datetime("us"))
    )
    n_rows = keys.height
    subject_of_row = keys.get_column("subject_id").to_numpy()
    prediction_us = keys.get_column("prediction_time").to_physical().to_numpy().astype(np.int64)

    rows_by_subject: dict[int, list[int]] = {}
    for row, subject in enumerate(subject_of_row.tolist()):
        rows_by_subject.setdefault(int(subject), []).append(row)

    bucket_us = int(config.visit_bucket_minutes) * MICROSECONDS_PER_MINUTE
    lookback_us = (
        None if config.lookback_days is None else int(round(config.lookback_days * MICROSECONDS_PER_DAY))
    )
    vocabulary_size = max(store.vocabulary_size, 1)
    side = "right" if config.include_events_at_prediction_time else "left"

    per_row_visits: list[np.ndarray | None] = [None] * n_rows
    per_row_codes: list[np.ndarray | None] = [None] * n_rows
    seen_subjects: set[int] = set()

    for shard in store.shards:
        frame = pl.read_parquet(shard, columns=["subject_id", "time", "code_index"])
        if frame.is_empty():
            continue
        shard_subjects = frame.get_column("subject_id").to_numpy()
        shard_times = frame.get_column("time").to_physical().to_numpy().astype(np.int64)
        shard_codes = frame.get_column("code_index").to_numpy().astype(np.int64)
        # The shard is written sorted by (subject_id, time, code_index), so each subject occupies one
        # contiguous range and its events are already in chronological order.
        boundaries = np.flatnonzero(np.diff(shard_subjects)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [shard_subjects.size]))
        for start, stop in zip(starts.tolist(), stops.tolist()):
            subject = int(shard_subjects[start])
            rows = rows_by_subject.get(subject)
            if not rows:
                continue
            if subject in seen_subjects:
                # MEDS keeps a subject's events in one shard. If that ever stops holding, a later shard
                # would silently replace an earlier partial history instead of extending it.
                raise ValueError(
                    f"Subject {subject} appears in more than one shard of {store.root}; RETAIN builds "
                    "each history from a single contiguous shard range"
                )
            seen_subjects.add(subject)
            times = shard_times[start:stop]
            codes = shard_codes[start:stop]
            for row in rows:
                prediction = int(prediction_us[row])
                high = int(np.searchsorted(times, prediction, side=side))
                low = 0
                if lookback_us is not None:
                    low = int(np.searchsorted(times, prediction - lookback_us, side="left"))
                if high <= low:
                    continue
                window_times = times[low:high]
                buckets = window_times if bucket_us == 0 else (window_times // bucket_us) * bucket_us
                visit_time, inverse = np.unique(buckets, return_inverse=True)
                if config.max_visits is not None and visit_time.size > config.max_visits:
                    # A reverse-time model attends to the most recent visits, so truncation drops the
                    # oldest ones, mirroring the paper's bounded observation window.
                    first_kept = visit_time.size - config.max_visits
                    keep = inverse >= first_kept
                    inverse = inverse[keep] - first_kept
                    visit_time = visit_time[first_kept:]
                    window_codes = codes[low:high][keep]
                else:
                    window_codes = codes[low:high]
                # `x` is binary, so a code seen twice inside one bucket must contribute once. Packing
                # (visit, code) into a single key deduplicates and orders both axes in one pass.
                packed = np.unique(inverse.astype(np.int64) * vocabulary_size + window_codes)
                per_row_visits[row] = visit_time
                per_row_codes[row] = packed

    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    widths: list[np.ndarray] = []
    times_out: list[np.ndarray] = []
    codes_out: list[np.ndarray] = []
    for row in range(n_rows):
        visit_time = per_row_visits[row]
        packed = per_row_codes[row]
        if visit_time is None or packed is None or visit_time.size == 0:
            row_ptr[row + 1] = row_ptr[row]
            continue
        row_ptr[row + 1] = row_ptr[row] + visit_time.size
        visit_index = packed // vocabulary_size
        widths.append(np.bincount(visit_index, minlength=visit_time.size).astype(np.int64))
        times_out.append(visit_time.astype(np.int64))
        codes_out.append((packed % vocabulary_size).astype(np.int32))

    width = np.concatenate(widths) if widths else np.zeros(0, dtype=np.int64)
    visit_ptr = np.zeros(width.size + 1, dtype=np.int64)
    np.cumsum(width, out=visit_ptr[1:])
    return SequenceBatch(
        keys=np.column_stack((subject_of_row.astype(np.int64), prediction_us)),
        row_ptr=row_ptr,
        visit_ptr=visit_ptr,
        visit_time=np.concatenate(times_out) if times_out else np.zeros(0, dtype=np.int64),
        codes=np.concatenate(codes_out) if codes_out else np.zeros(0, dtype=np.int32),
    )
