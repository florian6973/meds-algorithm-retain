"""The MEDS-to-visits adaptation layer, which has no counterpart in the official repository.

`process_mimic.py` hands RETAIN an already-tokenised list of visits. Everything between a MEDS event
stream and that list is this port's own work, so it is the part that needs its own tests rather than a
fidelity comparison.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from retain.config import RetainConfig
from retain.preprocess import preprocess, read_split_map
from retain.windows import PatientStore, build_sequences

DAY = 86_400_000_000


def _write_dataset(root, rows, splits):
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.Utf8,
        "numeric_value": pl.Float32,
    }
    frame = pl.from_dicts([{"numeric_value": None, **row} for row in rows], schema=schema)
    for split in {split for _, split in splits}:
        members = [subject for subject, assigned in splits if assigned == split]
        path = root / "data" / split / "0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.filter(pl.col("subject_id").is_in(members)).sort(
            "subject_id", "time", nulls_last=False
        ).write_parquet(path)
    pl.DataFrame(splits, schema=["subject_id", "split"], orient="row").write_parquet(
        root / "metadata" / "subject_splits.parquet"
    )
    return root


ROWS = [
    {"subject_id": 1, "time": None, "code": "GENDER//F"},
    {"subject_id": 1, "time": datetime(2020, 1, 1, 1), "code": "A"},
    {"subject_id": 1, "time": datetime(2020, 1, 1, 5), "code": "A"},
    {"subject_id": 1, "time": datetime(2020, 1, 1, 5), "code": "B"},
    {"subject_id": 1, "time": datetime(2020, 1, 3, 9), "code": "C"},
    {"subject_id": 1, "time": datetime(2020, 1, 5, 0), "code": "D"},
    {"subject_id": 2, "time": datetime(2020, 1, 1, 2), "code": "A"},
    {"subject_id": 2, "time": datetime(2020, 1, 2, 2), "code": "B"},
    {"subject_id": 3, "time": datetime(2020, 1, 4, 2), "code": "ONLY_HELD_OUT"},
    {"subject_id": 3, "time": datetime(2020, 1, 4, 3), "code": "A"},
]
SPLITS = [(1, "train"), (2, "train"), (3, "held_out")]


@pytest.fixture
def store(tmp_path):
    dataset = _write_dataset(tmp_path / "meds", ROWS, SPLITS)
    data_dir = tmp_path / "data"
    record = preprocess(dataset, _staged(data_dir), RetainConfig())
    assert record["static_events"] == 1
    return PatientStore(data_dir), record


def _staged(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _visits(batch, row=0):
    """The row's visits as `(bucket time, sorted code indices)` pairs."""
    start, stop = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
    return [
        (
            int(batch.visit_time[index]),
            sorted(batch.codes[int(batch.visit_ptr[index]) : int(batch.visit_ptr[index + 1])].tolist()),
        )
        for index in range(start, stop)
    ]


def _keys(subject, moment):
    return pl.DataFrame(
        {"subject_id": [subject], "prediction_time": [moment]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    )


def test_static_events_are_excluded_and_counted(store):
    _, record = store
    assert record["static_events"] == 1
    assert record["kept_events"] == 8


def test_the_vocabulary_is_learned_from_the_training_split_only(store):
    patients, _ = store
    codes = set(patients.code_strings())
    assert codes == {"A", "B", "C", "D"}
    assert "ONLY_HELD_OUT" not in codes


def test_held_out_codes_outside_the_vocabulary_are_dropped_not_mapped(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(3, datetime(2020, 1, 5)), RetainConfig())
    # Subject 3's visit keeps `A` and silently loses the unseen code rather than folding it into a
    # shared bucket, which is what a closed vocabulary means.
    assert _visits(batch) == [(int(np.datetime64(datetime(2020, 1, 4), "us").astype(np.int64)), [0])]


def test_daily_bucketing_merges_a_day_and_deduplicates_repeated_codes(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 6)), RetainConfig())
    visits = _visits(batch)
    assert [time for time, _ in visits] == [
        int(np.datetime64(datetime(2020, 1, day), "us").astype(np.int64)) for day in (1, 3, 5)
    ]
    # `A` occurs twice on 1 January; `x` is binary, so the visit must hold it once.
    assert len(visits[0][1]) == 2


def test_zero_bucket_width_keeps_every_distinct_timestamp(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 6)), RetainConfig(visit_bucket_minutes=0))
    assert [time for time, _ in _visits(batch)] == [
        int(np.datetime64(datetime(2020, 1, 1, hour), "us").astype(np.int64)) for hour in (1, 5)
    ] + [
        int(np.datetime64(datetime(2020, 1, 3, 9), "us").astype(np.int64)),
        int(np.datetime64(datetime(2020, 1, 5, 0), "us").astype(np.int64)),
    ]


def test_the_prediction_time_instant_is_included_by_default(store):
    patients, _ = store
    at_event = build_sequences(patients, _keys(1, datetime(2020, 1, 5, 0)), RetainConfig())
    assert len(_visits(at_event)) == 3
    excluded = build_sequences(
        patients, _keys(1, datetime(2020, 1, 5, 0)), RetainConfig(include_events_at_prediction_time=False)
    )
    assert len(_visits(excluded)) == 2


def test_nothing_after_the_prediction_time_enters_the_history(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 2)), RetainConfig())
    visits = _visits(batch)
    assert len(visits) == 1
    assert visits[0][0] < int(np.datetime64(datetime(2020, 1, 2), "us").astype(np.int64))


def test_a_bucket_is_cut_at_the_prediction_time_rather_than_straddling_it(store):
    """Bucketing happens after the cut, so a partial final day carries no later events."""
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 1, 3)), RetainConfig())
    visits = _visits(batch)
    assert len(visits) == 1
    assert visits[0][1] == [patients.code_strings().index("A")]


def test_lookback_drops_older_visits(store):
    patients, _ = store
    batch = build_sequences(
        patients, _keys(1, datetime(2020, 1, 6)), RetainConfig(lookback_days=3.0)
    )
    assert [time for time, _ in _visits(batch)] == [
        int(np.datetime64(datetime(2020, 1, day), "us").astype(np.int64)) for day in (3, 5)
    ]


def test_max_visits_keeps_the_most_recent(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 6)), RetainConfig(max_visits=2))
    assert [time for time, _ in _visits(batch)] == [
        int(np.datetime64(datetime(2020, 1, day), "us").astype(np.int64)) for day in (3, 5)
    ]


def test_a_key_before_every_event_yields_an_empty_sequence_not_a_dropped_row(store):
    patients, _ = store
    batch = build_sequences(patients, _keys(1, datetime(2019, 1, 1)), RetainConfig())
    assert batch.n_keys == 1
    assert batch.lengths.tolist() == [0]
    assert batch.n_empty == 1


def test_visit_gaps_are_days_since_the_previous_visit_and_zero_at_each_first_visit(store):
    patients, _ = store
    keys = pl.concat([_keys(1, datetime(2020, 1, 6)), _keys(2, datetime(2020, 1, 6))])
    batch = build_sequences(patients, keys, RetainConfig(use_time=True))
    np.testing.assert_allclose(batch.visit_gap_days(), [0.0, 2.0, 2.0, 0.0, 1.0])


def test_split_assignment_comes_from_metadata(store):
    patients, _ = store
    assert patients.split_of(pl.Series([1, 2, 3, 999])).to_list() == [
        "train",
        "train",
        "held_out",
        None,
    ]


def test_split_map_falls_back_to_the_sharded_directory_layout(tmp_path):
    dataset = _write_dataset(tmp_path / "meds", ROWS, SPLITS)
    (dataset / "metadata" / "subject_splits.parquet").unlink()
    derived = read_split_map(dataset).sort("subject_id")
    assert derived.rows() == [(1, "train"), (2, "train"), (3, "held_out")]


def test_a_dataset_without_a_training_split_is_refused(tmp_path):
    dataset = _write_dataset(tmp_path / "meds", ROWS, [(1, "held_out"), (2, "held_out"), (3, "held_out")])
    with pytest.raises(ValueError, match="declares no 'train' split"):
        preprocess(dataset, _staged(tmp_path / "data"), RetainConfig())


def test_min_code_count_prunes_the_vocabulary(tmp_path):
    dataset = _write_dataset(tmp_path / "meds", ROWS, SPLITS)
    record = preprocess(dataset, _staged(tmp_path / "data"), RetainConfig(min_code_count=2))
    patients = PatientStore(tmp_path / "data")
    # In the training split `A` occurs three times and `B` twice; `C` and `D` occur once each.
    assert record["vocabulary_size"] == 2
    assert patients.code_strings() == ["A", "B"]


def test_max_vocab_size_keeps_the_most_frequent_codes(tmp_path):
    dataset = _write_dataset(tmp_path / "meds", ROWS, SPLITS)
    preprocess(dataset, _staged(tmp_path / "data"), RetainConfig(max_vocab_size=2))
    assert PatientStore(tmp_path / "data").code_strings()[0] == "A"


def test_meds_standard_markers_are_excluded_from_the_visit_sequence(tmp_path):
    """The paper and the official code both keep demographics and outcome markers out of a visit.

    Paper section 4.1 builds visits from "diagnosis, medication and procedure codes"; age is a cohort
    restriction, not a feature. `process_mimic.py` builds sequences from `DIAGNOSES_ICD` alone and reads
    `PATIENTS.csv` only to derive the mortality label. `MEDS_BIRTH` would otherwise be the first visit of
    essentially every sequence, decades before the rest, holding one of RETAIN's attention slots;
    `MEDS_DEATH` would put the canonical outcome marker in the input space.
    """
    rows = list(ROWS) + [
        {"subject_id": 1, "time": datetime(1950, 5, 5), "code": "MEDS_BIRTH"},
        {"subject_id": 1, "time": datetime(2020, 1, 6), "code": "MEDS_DEATH"},
        {"subject_id": 2, "time": datetime(1955, 5, 5), "code": "MEDS_BIRTH"},
    ]
    dataset = _write_dataset(tmp_path / "meds", rows, SPLITS)
    record = preprocess(dataset, _staged(tmp_path / "data"), RetainConfig())
    patients = PatientStore(tmp_path / "data")
    assert "MEDS_BIRTH" not in patients.code_strings()
    assert "MEDS_DEATH" not in patients.code_strings()
    assert record["meds_standard_marker_events"] == 3

    batch = build_sequences(patients, _keys(1, datetime(2020, 1, 7)), RetainConfig())
    # Without the exclusion the 1950 birth would be visit 0, seventy years before the rest.
    assert _visits(batch)[0][0] == int(np.datetime64(datetime(2020, 1, 1), "us").astype(np.int64))


def test_the_marker_exclusion_can_be_turned_off(tmp_path):
    rows = list(ROWS) + [{"subject_id": 1, "time": datetime(1950, 5, 5), "code": "MEDS_BIRTH"}]
    dataset = _write_dataset(tmp_path / "meds", rows, SPLITS)
    preprocess(dataset, _staged(tmp_path / "data"), RetainConfig(exclude_meds_standard_markers=False))
    assert "MEDS_BIRTH" in PatientStore(tmp_path / "data").code_strings()


def test_a_task_anchored_on_a_subjects_first_event_is_not_empty_by_default(store):
    """The inclusive boundary is what keeps a first-event-anchored task from being all-empty.

    Measured on the MIMIC-IV demo cohort: with a prediction time exactly at each subject's first modelled
    event, the default `<=` boundary gives 0/100 empty histories and the strict `<` boundary gives
    **100/100**. Empty history otherwise occurs only for a prediction time preceding the subject's entire
    modelled record.

    This matters beyond the demo. The MIMIC-IV demo task anchors on ICU admission, which guarantees prior
    history; a task of the form "does code X occur within N days" carries no such anchor, and if it
    triggers on a subject's first record the strict cutoff would send every one of those keys to the
    learned intercept. The strict setting looks like the conservative choice and is the dangerous one
    here, so this pins the direction.
    """
    patients, _ = store
    first_event = datetime(2020, 1, 1, 1)  # subject 1's earliest event in ROWS
    inclusive = build_sequences(patients, _keys(1, first_event), RetainConfig())
    assert inclusive.n_empty == 0
    assert inclusive.lengths.tolist() == [1]

    strict = build_sequences(
        patients, _keys(1, first_event), RetainConfig(include_events_at_prediction_time=False)
    )
    assert strict.n_empty == 1, "the strict boundary empties a first-event-anchored key"


def test_a_subject_listed_twice_in_the_split_map_collapses_to_one_row(tmp_path):
    """The dedup in `read_split_map` is what keeps split assignment aligned to label rows.

    Verified by mutation that removing it passed all 114 tests: its only control was a fixture with
    no duplicate to catch, which is a control blind to the condition it protects against.
    """
    dataset = _write_dataset(tmp_path / "meds", ROWS, SPLITS + [(1, "train")])
    derived = read_split_map(dataset)
    assert derived.height == 3
    assert derived.get_column("subject_id").is_duplicated().sum() == 0


def test_split_of_refuses_a_split_map_that_would_misalign_rather_than_misassigning(store):
    """The consequence the dedup prevents, asserted on the consumer rather than assumed.

    An expanded join returns more splits than subjects asked for; nothing downstream compares the
    two lengths, so the failure would surface as held-out rows in the fit and not as an error.
    """
    patients, _ = store
    patients.splits = pl.concat([patients.splits, patients.splits.filter(pl.col("subject_id") == 3)])
    with pytest.raises(ValueError, match="more than once"):
        patients.split_of(pl.Series([1, 2, 3]))


def test_the_split_map_is_authoritative_when_shard_placement_disagrees(tmp_path):
    """Which of the two split sources wins, established by making them disagree.

    `_write_dataset` writes `data/<split>/` directories *and* `subject_splits.parquet` from one
    assignment, so any fixture built from it is unable to discriminate: consistent sources agree
    whichever one production actually reads. Only divergence answers it.
    """
    placement = [(1, "train"), (2, "held_out"), (3, "train")]
    declared = [(1, "train"), (2, "train"), (3, "held_out")]
    dataset = _write_dataset(tmp_path / "meds", ROWS, placement)
    pl.DataFrame(declared, schema=["subject_id", "split"], orient="row").write_parquet(
        dataset / "metadata" / "subject_splits.parquet"
    )

    # Subject 2's events sit in data/held_out/ and subject 3's in data/train/; the map says the
    # opposite for both. The map wins, so placement cannot silently reassign a subject.
    assert read_split_map(dataset).sort("subject_id").rows() == declared

    # And placement governs only when there is no map to consult.
    (dataset / "metadata" / "subject_splits.parquet").unlink()
    assert read_split_map(dataset).sort("subject_id").rows() == sorted(placement)


def test_shard_placement_is_inert_inside_the_published_artifact(tmp_path):
    """`preprocess` mirrors the input directory tree into `sequences/`, so the artifact carries a
    second apparent split source that nothing consults.

    `PatientStore` rglobs every shard and takes membership from the copied map, so a subject whose
    sequences live under `sequences/held_out/` is still `train` if the map says so. Pinned because
    filtering shards by directory is a plausible future optimisation that would silently reassign.
    """
    placement = [(1, "train"), (2, "held_out"), (3, "train")]
    declared = [(1, "train"), (2, "train"), (3, "held_out")]
    dataset = _write_dataset(tmp_path / "meds", ROWS, placement)
    pl.DataFrame(declared, schema=["subject_id", "split"], orient="row").write_parquet(
        dataset / "metadata" / "subject_splits.parquet"
    )
    out = tmp_path / "data"
    preprocess(dataset, _staged(out), RetainConfig())
    patients = PatientStore(out)

    assert (out / "sequences" / "held_out").is_dir(), "the fixture must actually diverge the two"
    assert patients.split_of(pl.Series([1, 2, 3])).to_list() == ["train", "train", "held_out"]


def test_a_flat_extract_without_a_split_map_is_refused_rather_than_called_train(tmp_path):
    """The trapdoor in the directory fallback: `parts[0] if len(parts) > 1 else TRAIN_SPLIT`.

    A flat layout has no split component, so every subject defaulted to `train` -- measured as
    `[(1, 'train'), (2, 'train'), (3, 'train')]` before this refusal. That is a transductive
    vocabulary and a fit over held-out patients, reported by nothing. Reachable only when the
    dataset also ships no `subject_splits.parquet`, which is why no fixture had produced it.
    """
    dataset = tmp_path / "meds"
    (dataset / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "time": [datetime(2020, 1, 1)] * 3,
            "code": ["A", "B", "C"],
            "numeric_value": [None, None, None],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.Utf8,
            "numeric_value": pl.Float32,
        },
    ).write_parquet(dataset / "data" / "0.parquet")

    with pytest.raises(ValueError, match="flat data/ layout"):
        read_split_map(dataset)
