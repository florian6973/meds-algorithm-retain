"""Fixtures over a tiny real MEDS dataset.

The ``simple_static_MEDS_dataset_with_task`` fixture comes from ``meds_testing_helpers`` (a pytest plugin)
— no import needed.

The machinery for actually driving this model's DAG lives in ``meds_model_base.testing`` rather than here,
so any test module can import it. A ``conftest.py`` is not importable from a sibling test module: ``tests/``
is not a package, and ``from .conftest import ...`` fails at collection time — before ``-m "not slow"`` can
deselect anything.
"""

from pathlib import Path

import pytest
from meds_model_base.testing import build_workspace, harness, run_chain, run_cli

from teco.commands import COMMANDS

#: What the conformance runs train instead of the published geometry, and why.
#:
#: `configs/model/default.yaml` ships TECO's own architecture: 6 layers, d_model 512, and a 384-interval
#: grid — 18.9M parameters attending over 384 positions. `run_chain` pins `trainer.accelerator=cpu`, where
#: one optimizer step of that model costs on the order of half a teraflop; the learnability tier would run
#: for hours per test. These overrides shrink the *fixture's* model, and nothing else: they are passed as
#: command-line overrides to `supervised_train`, exactly as a user would pass them, and they change no
#: default that `meds-model` uses outside this test session.
#:
#: This is the sanctioned move — `docs/PORTING-A-MODEL.md` step 4 rules out shrinking the *model* to suit
#: the fixture, and rules in adjusting the fixture. Nothing about TECO's published configuration is
#: weakened here; `pytest src/teco/model.py` still builds and checks the full-size model.
CONFORMANCE_MODEL = [
    "model.d_model=64",
    "model.nhead=4",
    "model.num_layers=2",
    "model.dim_feedforward=128",
    "model.interval_minutes=240.0",
]


@pytest.fixture
def cli():
    """Run ``meds-model <args>``, raising with captured output on failure."""
    return run_cli


@pytest.fixture
def chain(monkeypatch):
    """Run every command this model registers over a prepared ``data_dir``, at conformance size.

    ``run_chain`` builds its own argument list and takes no hook for extra overrides, so the size arguments
    are appended by wrapping the ``run_cli`` it calls. Wrapping rather than reimplementing keeps the
    property that makes ``run_chain`` worth using: it follows ``COMMANDS`` and each command's declared
    ``supported_sources``, so a DAG whose wiring and implementation disagree still fails here.
    """
    inner = harness.run_cli

    def sized(args: list[str]):
        return inner([*args, *CONFORMANCE_MODEL] if args and args[0] == "supervised_train" else args)

    monkeypatch.setattr(harness, "run_cli", sized)

    def _chain(data_dir: Path, labels_dir: Path, out_dir: Path, **kwargs):
        return run_chain(COMMANDS, data_dir, labels_dir, out_dir, **kwargs)

    return _chain


@pytest.fixture(scope="session")
def meds_root(simple_static_MEDS_dataset_with_task) -> Path:
    """A tiny MEDS root (with a binary task) shipped by meds-testing-helpers."""
    return Path(simple_static_MEDS_dataset_with_task)


@pytest.fixture(scope="session")
def labels_dir(meds_root) -> Path:
    """A MEDS labels directory inside the fixture (subject_id, prediction_time, boolean_value)."""
    task_root = meds_root / "task_labels"
    tasks = sorted(p for p in task_root.iterdir() if p.is_dir())
    assert tasks, f"No task_labels/* directory under {meds_root}"
    return tasks[0]


@pytest.fixture(scope="session")
def data_dir(meds_root, tmp_path_factory) -> Path:
    """A shared workspace with patients/, built once for the session. Labels are not preprocessed."""
    workspace = tmp_path_factory.mktemp("workspace") / "data"
    return build_workspace(meds_root, workspace)
