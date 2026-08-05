"""`infer` must attach each subject's embedding to *that subject's* key.

This is not a hypothetical. `meds_model_base.commands._runtime.run_predict_step` pairs
`dataset.schema_df` (index order) with `trainer.predict(...)` outputs (iteration order), and
meds-torch-data's `Datamodule.train_dataloader()` shuffles — so on the **train** split the default
`infer` writes every subject a different subject's embedding, silently and differently on each run. See
`porter.commands.OrderStableInferCommand`, which is why this repository does not use the default.

The check is model-free and deliberately so: it compares the code sequence in each batch row against the
subject `schema_df` claims is at that position, which is the invariant `infer` relies on and nothing else
in the suite tests. A model would only obscure it.
"""

from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from meds_model_base.commands._runtime import KEYS, SPLIT_ATTRS
from meds_model_base.lightning import register_structured_configs
from meds_model_base.schemas import train_split
from meds_model_base.tasks import materialize_labels
from meds_model_base.testing import build_workspace
from meds_model_base.testing.synthetic import build_signal_dataset

import porter
from porter.commands import ordered_split_loader

#: Enough train subjects that a shuffle cannot leave the order intact by luck.
N_TRAIN = 32

CONFIG_DIR = str(Path(porter.__file__).parent / "configs")


@pytest.fixture(scope="module")
def aligned_workspace(tmp_path_factory) -> tuple[Path, Path]:
    """A synthetic MEDS workspace plus its labels directory. No model, no training."""
    root = tmp_path_factory.mktemp("alignment")
    raw = build_signal_dataset(root / "raw", n_train=N_TRAIN, n_tuning=8, n_held_out=8, seed=0)
    return build_workspace(raw, root / "data"), raw / "task_labels" / "signal_task"


def _infer_cfg(data_dir: Path, labels_dir: Path, batch_size: int = 8):
    """The real `infer.yaml`, composed exactly as the dispatcher composes it.

    Going through Hydra rather than hand-building a dict matters here: `datamodule.config` is a registered
    *structured* config, and only that registration turns `seq_sampling_strategy: TO_END` from a string
    into the enum meds-torch-data validates against.
    """
    register_structured_configs()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="infer",
            overrides=[
                f"input_data_dir={data_dir}",
                f"external_labels_dir={labels_dir}",
                f"batch_size={batch_size}",
                "num_workers=0",
            ],
        )
    # Set at runtime by the command, from labels materialized into its own work directory.
    cfg.datamodule.config.task_labels_dir = str(labels_dir)
    return cfg


@pytest.mark.parametrize("split", sorted(SPLIT_ATTRS))
def test_batch_rows_follow_schema_df_order(aligned_workspace, tmp_path, split):
    """Batch row *i* must hold the timeline of the subject `schema_df` lists at row *i* — every split."""
    data_dir, labels_src = aligned_workspace
    labels_dir, _ = materialize_labels(labels_src, data_dir / "patients", tmp_path / f"labels_{split}")
    cfg = _infer_cfg(data_dir, labels_dir)

    dataset, dataloader = ordered_split_loader(cfg, split)
    subjects = dataset.schema_df["subject_id"].to_list()
    assert subjects, f"the {split} split is empty, so this test would pass vacuously"

    codes = pl.read_parquet(data_dir / "patients" / "metadata" / "codes.parquet")
    index_to_code = dict(
        zip(codes["code/vocab_index"].cast(pl.Int64).to_list(), codes["code"].to_list(), strict=True)
    )
    raw = pl.concat(
        [pl.read_parquet(p) for p in sorted((data_dir.parent / "raw" / "data").rglob("*.parquet"))]
    )
    timelines = {
        subject: frame.sort("time")["code"].to_list() for (subject,), frame in raw.group_by("subject_id")
    }

    position = 0
    for batch in dataloader:
        for row in range(batch.code.shape[0]):
            expected = timelines[subjects[position]]
            got = [index_to_code[c] for c in batch.code[row].tolist() if c != 0]
            assert got == expected, (
                f"{split} split, position {position}: the batch holds a timeline that is not subject "
                f"{subjects[position]}'s. `infer` writes this row against that subject's key, so the "
                "published embedding belongs to someone else."
            )
            position += 1
    assert position == len(subjects), f"iterated {position} rows for {len(subjects)} index rows"


def test_train_split_is_the_one_that_can_regress(aligned_workspace, tmp_path):
    """Guard the loader itself: the train split must not be shuffled, twice in a row.

    `train_dataloader()` shuffles by construction; this asserts the loader `infer` actually uses does not,
    which is the property `_runtime`'s `SPLIT_ATTRS` comment claims and does not hold for.
    """
    data_dir, labels_src = aligned_workspace
    labels_dir, _ = materialize_labels(labels_src, data_dir / "patients", tmp_path / "labels")
    cfg = _infer_cfg(data_dir, labels_dir)

    orders = []
    for _ in range(2):
        _, dataloader = ordered_split_loader(cfg, train_split)
        orders.append([row for batch in dataloader for row in batch.code.tolist()])
    assert orders[0] == orders[1], "the train-split loader used by `infer` is not order-stable"


def test_index_keys_are_unique(aligned_workspace, tmp_path):
    """A duplicated key would make the alignment above untestable and the artifact join ambiguous."""
    data_dir, labels_src = aligned_workspace
    labels_dir, _ = materialize_labels(labels_src, data_dir / "patients", tmp_path / "labels")
    for split in SPLIT_ATTRS:
        dataset, _ = ordered_split_loader(_infer_cfg(data_dir, labels_dir), split)
        keys = dataset.schema_df.select(KEYS)
        assert len(keys) == len(keys.unique()), f"{split} index has duplicate (subject_id, prediction_time)"
