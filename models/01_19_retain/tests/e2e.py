"""Model-owned DAG adapters plus shared MEDS-DEV MIMIC demo provisioning."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from retain.commands import COMMANDS

from .fixtures import SMOKE_HYPERPARAMETERS, write_meds_dataset, write_task_labels


@dataclass(frozen=True)
class E2EResult:
    """Artifacts plus the requested evaluation index used to verify prediction keys."""

    artifacts: Mapping[str, Path]
    labels_dir: Path
    splits: Sequence[str]


@dataclass(frozen=True)
class MEDSDevRun:
    """MEDS-DEV invocation plus the evaluation index its predictions must cover."""

    args: Sequence[str]
    predictions_dir: Path
    labels_dir: Path
    splits: Sequence[str]


def prepare_fixture(tmp_path, **inputs):
    """Return `(meds_dir, labels_dir)`, generating the synthetic fixture if needed."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    meds_dir = inputs.get("external_meds_dir")
    labels_dir = inputs.get("external_labels_dir")
    if meds_dir is None:
        meds_dir = write_meds_dataset(tmp_path / "meds")
    if labels_dir is None:
        labels_dir = write_task_labels(meds_dir, tmp_path / "labels")
    return Path(meds_dir), Path(labels_dir)


def run_e2e(tmp_path, **inputs):
    """Execute the real `supervised` DAG and return an `E2EResult`.

    The result carries the label index and the evaluated splits alongside the artifacts, because SPEC
    section 8 requires the generic conformance test to compare the actual prediction keys against the
    requested evaluation index rather than trust a self-reported count.

    With no keyword arguments this runs on the synthetic fixture at smoke-sized hyper-parameters. The
    MIMIC-IV demo test passes real directories, `splits`, and its own hyper-parameters instead. This port
    is predicate-free, so `external_predicates_file` is accepted by the demo test's calling convention and
    deliberately never forwarded to a command -- the commands reject it.
    """
    tmp_path = Path(tmp_path)
    meds_dir, labels_dir = prepare_fixture(tmp_path, **inputs)
    settings = dict(inputs.get("hyperparameters") or SMOKE_HYPERPARAMETERS)
    # `None` means "score every requested key"; a list restricts both the run and the key comparison.
    splits = inputs.get("splits")

    data_dir = tmp_path / "artifacts" / "data"
    model_dir = tmp_path / "artifacts" / "model"
    predictions_dir = tmp_path / "artifacts" / "predictions"

    vocabulary_settings = {
        key: settings[key] for key in ("min_code_count", "max_vocab_size") if key in settings
    }
    COMMANDS["preprocess_data"]()(
        {
            "external_meds_dir": meds_dir,
            "output_data_dir": data_dir,
            "do_overwrite": True,
            **vocabulary_settings,
        }
    )
    COMMANDS["supervised_train"]()(
        {
            "input_data_dir": data_dir,
            "external_labels_dir": labels_dir,
            "output_supervised_model_dir": model_dir,
            "do_overwrite": True,
            **settings,
        }
    )
    predict_arguments = {
        "input_supervised_model_dir": model_dir,
        "external_labels_dir": labels_dir,
        "output_predictions_dir": predictions_dir,
        "do_overwrite": True,
    }
    if splits is not None:
        predict_arguments["splits"] = list(splits)
    COMMANDS["predict"]()(predict_arguments)

    return E2EResult(
        artifacts={
            "preprocess_data": data_dir,
            "supervised_train": model_dir,
            "predict": predictions_dir,
        },
        labels_dir=labels_dir,
        # With no restriction every label file is in scope, which `assert_prediction_keys` expresses as an
        # empty split selection.
        splits=list(splits) if splits is not None else [],
    )


def prepare_meds_dev_run(meds_dev, tmp_path):
    """Return ``MEDSDevRun(args, predictions_dir, labels_dir, splits)`` for this implementation.

    MEDS-DEV's own execution path is what is under test here -- descriptor formatting, isolated
    installation, and the real command slots -- so this runs on the synthetic fixture. Real data has its
    own gate and its own human approval. ``meds_dev`` is the provisioned checkout; nothing is read from it
    because the run is driven entirely through ``meds-dev-model``.

    No ``predicates_path`` is passed: this port is predicate-free and its ``model.yaml`` contains no
    ``{predicates_path}`` slot to populate.
    """
    tmp_path = Path(tmp_path)
    meds_dir, labels_dir = prepare_fixture(tmp_path / "fixture")
    output_dir = tmp_path / "meds-dev-run"
    del meds_dev

    arguments = [
        f"dataset_dir={meds_dir}",
        f"labels_dir={labels_dir}",
        f"output_dir={output_dir}",
        "dataset_type=supervised",
        "mode=full",
        "dataset_name=retain-fixture",
        "task_name=mortality_fixture",
        "do_overwrite=true",
    ]
    predictions_dir = output_dir / "retain-fixture" / "mortality_fixture" / "predict" / "predictions"
    # `mode=full` scores the held-out split, so that is the index the predictions must cover exactly.
    return MEDSDevRun(
        args=arguments,
        predictions_dir=predictions_dir,
        labels_dir=labels_dir,
        splits=["held_out"],
    )


def prepare_mimic_demo(meds_dev, tmp_path):
    """Return MEDS-DEV's ``(MIMIC demo, task labels, MIMIC predicates)`` paths.

    Use ``MEDS_DEMO_DIR`` when set; otherwise invoke MEDS-DEV's own demo builder. Pin an existing
    demo-compatible MEDS-DEV task rather than defining a task in this repository.
    """
    from .integration import RUN_TIMEOUT, run, venv_bin

    configured = os.environ.get("MEDS_DEMO_DIR")
    dataset_dir = Path(configured).resolve() if configured else tmp_path / "mimic-iv-demo"
    if not configured:
        run(
            [
                venv_bin(meds_dev) / "meds-dev-dataset",
                "dataset=MIMIC-IV",
                f"output_dir={dataset_dir}",
                "demo=true",
            ],
            timeout=RUN_TIMEOUT,
        )
    labels_dir = tmp_path / "mortality-in-icu-first-24h"
    run(
        [
            venv_bin(meds_dev) / "meds-dev-task",
            "task=mortality/in_icu/first_24h",
            "dataset=MIMIC-IV",
            f"dataset_dir={dataset_dir}",
            f"output_dir={labels_dir}",
        ],
        timeout=RUN_TIMEOUT,
    )
    predicates_file = meds_dev / "src" / "MEDS_DEV" / "datasets" / "MIMIC-IV" / "predicates.yaml"
    return dataset_dir, labels_dir, predicates_file
