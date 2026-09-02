"""Fleet conformance spec for `STAGES` in `slurm/submit.sh`.

Six cases. This file is written to TRAVEL: it derives the declared command list from
`meds-model commands` rather than naming this model's stages, so it can be copied to another port
unchanged and will still say what that port's filter has to preserve.

Why a spec and not just an implementation: five ports had a `STAGES` filter and they did not agree.
Three iterate the *request* and append, so `STAGES=predict,supervised_train` submits `predict` first;
one iterates the declared list and filters, so it does not. Submissions chain with `afterok`, and
against a run directory where `$RUN_DIR/model` already exists -- the artifact-reuse case a stage
filter exists for -- caller order lets `predict` run first against a stale model, then
`supervised_train` overwrites the model beneath predictions already written from the previous one,
with no error anywhere. An implementation gets copied; a test says what copying it has to preserve.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = ROOT / ".venv" / "bin"


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        **os.environ,
        "PATH": f"{VENV_BIN}{os.pathsep}{os.environ['PATH']}",
        "MEDS_ROOT": "/meds",
        "LABELS_DIR": "/labels",
    }
    environment.pop("STAGES", None)
    environment.update(overrides)
    return environment


def _declared() -> list[str]:
    """The model's own command list -- never hardcoded, so this file ports unchanged."""
    result = subprocess.run(
        ["meds-model", "commands"], capture_output=True, text=True, env=_environment()
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def _submitted(**overrides: str) -> tuple[int, list[str], str]:
    """The stages `submit.sh --dry-run` would submit, in submission order."""
    result = subprocess.run(
        ["bash", str(ROOT / "slurm" / "submit.sh"), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_environment(**overrides),
    )
    # `submit.sh --dry-run` prints the sbatch argv through `printf %q`, which escapes the commas
    # inside --export, so the value must be matched rather than split out.
    stages = re.findall(r"MEDS_MODEL_COMMAND=([A-Za-z_][A-Za-z0-9_]*)", result.stdout)
    return result.returncode, stages, result.stderr


@pytest.fixture(scope="module")
def declared() -> list[str]:
    stages = _declared()
    assert len(stages) >= 2, "the spec needs at least two declared stages to be meaningful"
    return stages


def test_1_unset_submits_the_whole_declared_profile(declared):
    """A single-task run wants the full DAG, so the filter must be opt-in."""
    code, stages, _ = _submitted()
    assert code == 0
    assert stages == declared


def test_2_a_subset_submits_exactly_that_subset(declared):
    code, stages, _ = _submitted(STAGES=",".join(declared[-2:]))
    assert code == 0
    assert stages == declared[-2:]


def test_3_the_declared_order_survives_a_reversed_request(declared):
    """The case that inverted the fleet decision.

    Submissions chain with `afterok`, so honouring the caller's order queues the later stage first.
    Loud against a clean run directory; silent when `$RUN_DIR/model` already exists.
    """
    code, stages, _ = _submitted(STAGES=",".join(reversed(declared[-2:])))
    assert code == 0
    assert stages == declared[-2:], "STAGES order must not reach the submission order"


def test_4_an_undeclared_stage_fails_loudly_and_says_what_is_declared(declared):
    """A typo must not submit nothing and exit 0 -- an absent result reading as a passed check."""
    code, stages, stderr = _submitted(STAGES=declared[-1] + "x")
    assert code == 2, f"expected exit 2, got {code}"
    assert stages == []
    assert declared[-1] + "x" in stderr, "the message must name the offending value"
    for stage in declared:
        assert stage in stderr, "the message must list what IS declared, not only what is not"


def test_5_stages_accepts_spaces_as_well_as_commas(declared):
    """The permissive separator, the one agreed change to the reference implementation.

    A comma-only parser yields ONE element here -- the whole string -- which then fails the declared
    check and exits 2. That divergence was live across five ports.
    """
    code, stages, _ = _submitted(STAGES=" ".join(declared[-2:]))
    assert code == 0
    assert stages == declared[-2:]


def test_6_a_repeated_stage_is_submitted_once(declared):
    """`STAGES=predict,predict` must not chain two identical jobs.

    Both would exit 0, the second overwriting the first under `do_overwrite=true`, so nothing
    downstream would flag the wasted job.
    """
    code, stages, _ = _submitted(STAGES=f"{declared[-1]},{declared[-1]}")
    assert code == 0
    assert stages == [declared[-1]]
