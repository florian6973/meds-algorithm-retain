"""Fixtures over a tiny real MEDS dataset.

The ``simple_static_MEDS_dataset_with_task`` fixture comes from ``meds_testing_helpers`` (a pytest plugin)
— no import needed.

The machinery for actually driving this model's DAG lives in ``meds_model_base.testing`` rather than here,
so any test module can import it. A ``conftest.py`` is not importable from a sibling test module: ``tests/``
is not a package, and ``from .conftest import ...`` fails at collection time — before ``-m "not slow"`` can
deselect anything.
"""

import os
from pathlib import Path

import pytest
from meds_model_base.testing import build_workspace, run_chain, run_cli

from porter.commands import COMMANDS
from porter.model import SMALL_BACKBONE

# ----------------------------------------------------------------------------------------------------
# Run the whole suite against a small model.
#
# `configs/model/default.yaml` is PORTER at the published scale: 28 x 768, 115.6 M parameters. That is
# the right default — it is what the paper describes and what a real run should use — but it cannot be
# trained on CPU inside a test.
#
# The obvious fix, passing `model=small` on the command line, is not available: the conformance tests
# drive the DAG through `meds_model_base.testing.run_chain`, which builds its own argv from the COMMANDS
# registry and takes only `epochs` and `batch_size`. So the size is injected through the environment
# instead — `configs/model/default.yaml` reads these with `oc.env`, and `run_cli` spawns `meds-model` as
# a subprocess, which inherits them.
#
# This is the porting procedure's sanctioned move ("fix the fixture — env vars, a larger synthetic
# dataset, adjusted thresholds"), and it is the right way round: the *default* stays faithful and the
# *test* is what scales down. `setdefault` so an explicit value from the caller still wins.
# ----------------------------------------------------------------------------------------------------
for _key, _value in {
    "PORTER_HIDDEN_SIZE": SMALL_BACKBONE["hidden_size"],
    "PORTER_NUM_LAYERS": SMALL_BACKBONE["num_hidden_layers"],
    "PORTER_NUM_HEADS": SMALL_BACKBONE["num_attention_heads"],
    "PORTER_INTERMEDIATE_SIZE": SMALL_BACKBONE["intermediate_size"],
    # Separate reason from the four above. The datamodules default to PREPEND, which is how the paper
    # carries demographics; meds-torch-data raises on a cohort with *no* static measurements, and the
    # template's synthetic learnability fixture (`build_signal_dataset`) has none. That fixture lives in
    # the vendored `meds_model_base.testing`, so it cannot be fixed here — reported upstream as
    # MEDS_model_template#16. The real fixture (`simple_static_MEDS_dataset_with_task`) does have
    # statics; `tests/test_default_is_the_paper.py` asserts the shipped default is still PREPEND.
    "PORTER_STATIC_INCLUSION": "OMIT",
}.items():
    os.environ.setdefault(_key, str(_value))


@pytest.fixture
def cli():
    """Run ``meds-model <args>``, raising with captured output on failure."""
    return run_cli


@pytest.fixture
def chain():
    """Run every command this model registers over a prepared ``data_dir``."""

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
