"""Algorithm-owned profile execution and fitted-feature inspection adapters."""

from collections.abc import Mapping
from pathlib import Path

import polars as pl
from meds_algorithm.conformance.signal import add_split_canaries, build_signal_dataset
from meds_algorithm.conformance.types import E2EResult, MEDSDevRun

from retain_meds import RETAINConfig
from retain_meds.retain_meds import create_repository

TASK = "signal_task"

#: A small but real RETAIN for offline tests: the published architecture at reduced width, on CPU so two
#: runs are bit-identical. Visits are hourly rather than daily because the signal fixture and the null world
#: space their events hours apart; a daily bucket would fold most histories into one or two visits and
#: never exercise the visit attention.
TEST_CONFIG = RETAINConfig(
    visit_bucket_minutes=60,
    embed_size=16,
    alpha_hidden_dim_size=16,
    beta_hidden_dim_size=16,
    batch_size=32,
    n_epochs=3,
    device="cpu",
)


def build_fixture(root: Path, *, seed: int = 0, shuffle_labels: bool = False, **signal) -> tuple[Path, dict]:
    """Core signal dataset plus the split canaries of the transductive-fitting check."""
    build_signal_dataset(root, seed=seed, shuffle_labels=shuffle_labels, **signal)
    return root, add_split_canaries(root)


def run_e2e(
    tmp_path,
    *,
    profile: str,
    seed: int = 0,
    external_meds_dir=None,
    external_labels_dir=None,
    external_index_dir=None,
    external_evaluation_labels_dir=None,
    external_predicates_file=None,
    splits=("held_out",),
    config: RETAINConfig = TEST_CONFIG,
):
    """Run preprocess -> train -> predict through the core runner, each with a fresh algorithm instance.

    RETAIN consumes no predicates; a supplied predicates file is recorded as an input only. The seed is
    passed to every step and is the only source of randomness (initialisation, dropout, batch order).
    """
    tmp_path = Path(tmp_path)
    canaries = {}
    if external_meds_dir is None:
        external_meds_dir, canaries = build_fixture(tmp_path / "fixture", seed=seed)
        external_labels_dir = external_labels_dir or external_meds_dir / "task_labels" / TASK
    if external_labels_dir is None:
        raise ValueError("external_labels_dir is required with an external MEDS dataset")
    external_index_dir = external_index_dir or external_labels_dir
    external_evaluation_labels_dir = external_evaluation_labels_dir or external_index_dir
    tmp_path.mkdir(parents=True, exist_ok=True)
    common = {"seed": seed, "config_file": str(config.save(tmp_path / "config.json"))}

    repository = create_repository()
    artifacts = {
        "preprocess_data": tmp_path / "data",
        "train_supervised": tmp_path / "algorithm",
        "predict_supervised": tmp_path / "predictions",
    }
    repository.run(
        "preprocess_data",
        {"external_meds_dir": str(external_meds_dir), "output_data_dir": str(artifacts["preprocess_data"]), **common},
        profile=profile,
    )
    repository.run(
        "train_supervised",
        {
            "input_data_dir": str(artifacts["preprocess_data"]),
            "external_labels_dir": str(external_labels_dir),
            "output_supervised_algorithm_dir": str(artifacts["train_supervised"]),
            **common,
        },
        profile=profile,
    )
    repository.run(
        "predict_supervised",
        {
            "input_data_dir": str(artifacts["preprocess_data"]),
            "input_supervised_algorithm_dir": str(artifacts["train_supervised"]),
            "external_index_dir": str(external_index_dir),
            "splits": list(splits),
            "output_predictions_dir": str(artifacts["predict_supervised"]),
            "external_evaluation_labels_dir": str(external_evaluation_labels_dir),
            **common,
        },
        profile=profile,
    )
    inputs = {
        "seed": seed,
        "external_meds_dir": Path(external_meds_dir),
        "external_labels_dir": Path(external_labels_dir),
        "external_index_dir": Path(external_index_dir),
        "external_evaluation_labels_dir": Path(external_evaluation_labels_dir),
        "config": config,
    }
    if external_predicates_file is not None:
        inputs["external_predicates_file"] = Path(external_predicates_file)
    return E2EResult(
        artifacts=artifacts,
        labels_dir=Path(external_index_dir),
        splits=list(splits),
        inputs=inputs,
        canaries=canaries,
    )


def fitted_codes(artifacts: Mapping[str, Path]) -> set[str]:
    """RETAIN's only fitted population statistic: the vocabulary learned by preprocess_data."""
    vocabulary = pl.read_parquet(Path(artifacts["preprocess_data"]) / "vocabulary.parquet")
    return set(vocabulary["code"].to_list())


def prepare_meds_dev_run(meds_dev, tmp_path, *, profile: str):
    """Train on the synthetic fixture and predict its held-out split through MEDS-DEV's full mode."""
    dataset, _ = build_fixture(Path(tmp_path) / "fixture")
    labels_dir = dataset / "task_labels" / TASK
    output_dir = Path(tmp_path) / "meds-dev-run"
    args = [
        "mode=full",
        "dataset_type=supervised",
        f"dataset_dir={dataset}",
        f"labels_dir={labels_dir}",
        "dataset_name=synthetic_signal",
        f"task_name={TASK}",
        f"output_dir={output_dir}",
    ]
    predictions_dir = output_dir / "synthetic_signal" / TASK / "predict" / "predictions"
    return MEDSDevRun(args=args, predictions_dir=predictions_dir, labels_dir=labels_dir, splits=["held_out"])
