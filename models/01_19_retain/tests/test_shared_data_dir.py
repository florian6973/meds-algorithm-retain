"""`RETAIN_DATA_DIR` shares one preprocess artifact without sharing a run directory.

The eICU wave is 22 tasks per port against a 2.67B-event extract, and `preprocess_data` here is
task-independent, so rebuilding it per task is 22x the work every other port does once. The obvious
shortcut -- point `RUN_DIR` at a shared run -- is worse than the cost it avoids: `model/` and
`predict/` hang off `RUN_DIR` too and every stage sets `do_overwrite=true`, so 22 tasks would
overwrite one another in silence and one task's predictions would stand in for all of them. Nothing
would error. These tests pin the split that makes sharing safe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "MEDS_ROOT": "/meds",
    "LABELS_DIR": "/labels",
    "RUN_DIR": "/runs/task-a/artifacts",
}


def _stage_args(command: str, **overrides: str) -> dict[str, str]:
    """The key=value arguments `config.sh` emits for one stage."""
    environment = {**os.environ, **BASE, **overrides}
    environment.pop("RETAIN_DATA_DIR", None)
    environment.update({k: v for k, v in overrides.items()})
    result = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/slurm/config.sh"; stage_args {command}'],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def test_the_default_keeps_the_data_artifact_inside_the_run():
    """Unset, a run stays self-contained -- no shared state to reason about."""
    assert _stage_args("preprocess_data")["output_data_dir"] == f"{BASE['RUN_DIR']}/data"
    assert _stage_args("supervised_train")["input_data_dir"] == f"{BASE['RUN_DIR']}/data"


def test_only_the_data_artifact_is_shared():
    """`RETAIN_DATA_DIR` moves the data artifact and *nothing else*.

    Asserted as a negative on `model/` and `predict/` rather than only as a positive on the data
    paths: a change that routed the outputs through the shared directory too would still satisfy
    "the data dir is shared" while reintroducing exactly the silent overwrite this exists to prevent.
    """
    shared = "/shared/eicu/data"
    preprocess = _stage_args("preprocess_data", RETAIN_DATA_DIR=shared)
    train = _stage_args("supervised_train", RETAIN_DATA_DIR=shared)
    predict = _stage_args("predict", RETAIN_DATA_DIR=shared)

    assert preprocess["output_data_dir"] == shared
    assert train["input_data_dir"] == shared

    assert train["output_supervised_model_dir"] == f"{BASE['RUN_DIR']}/model"
    assert predict["input_supervised_model_dir"] == f"{BASE['RUN_DIR']}/model"
    assert predict["output_predictions_dir"] == f"{BASE['RUN_DIR']}/predict"
    for key in ("output_supervised_model_dir", "input_supervised_model_dir", "output_predictions_dir"):
        value = {**train, **predict}[key]
        assert not value.startswith(shared), f"{key}={value} would be shared across tasks"


def test_two_tasks_sharing_one_data_artifact_keep_separate_outputs():
    """The property that actually prevents the silent overwrite, stated over two tasks.

    One shared `data/`, two run directories, and no output path in common. A single-task check cannot
    observe this: with one run every path is trivially distinct from every other task's, of which
    there are none.
    """
    shared = "/shared/eicu/data"
    a = _stage_args("predict", RETAIN_DATA_DIR=shared, RUN_DIR="/runs/task-a/artifacts")
    b = _stage_args("predict", RETAIN_DATA_DIR=shared, RUN_DIR="/runs/task-b/artifacts")
    train_a = _stage_args("supervised_train", RETAIN_DATA_DIR=shared, RUN_DIR="/runs/task-a/artifacts")
    train_b = _stage_args("supervised_train", RETAIN_DATA_DIR=shared, RUN_DIR="/runs/task-b/artifacts")

    assert train_a["input_data_dir"] == train_b["input_data_dir"] == shared
    assert train_a["output_supervised_model_dir"] != train_b["output_supervised_model_dir"]
    assert a["output_predictions_dir"] != b["output_predictions_dir"]
