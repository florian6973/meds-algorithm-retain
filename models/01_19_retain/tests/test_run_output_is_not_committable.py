"""Run output lives inside the checkout, so `.gitignore` is the thing standing between a real submit and
committed dataset paths.

`slurm/config.sh` defaults `RUN_ROOT` to the relative path `runs`, and `scripts/github-sync.sh` publishes
from `runs/<id>/` by design, so the run directory cannot simply be excluded wholesale. That makes the
ignore rules load-bearing rather than housekeeping: `submit.sh` writes `runs/<id>/command-args/<cmd>.args`
on every real submit, and this model's `stage_args` puts `external_meds_dir` and `external_labels_dir` in
them. A default-configured submit followed by `git add -A` would otherwise commit the dataset and task
paths of a real run.

These tests assert both directions, because an ignore rule that is too broad silently breaks result
publication and an ignore rule that is too narrow silently leaks. The allowlist is derived from what
`github-sync.sh` actually commits rather than restated here, so the two cannot drift apart.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RUN_ID = "20260101T000000Z-conformance"


def _is_ignored(path: str) -> bool:
    """`git check-ignore` exit status: 0 means the path is ignored."""
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def _published_paths() -> list[str]:
    """The paths `scripts/github-sync.sh` stages, read from the script itself."""
    script = (ROOT / "scripts" / "github-sync.sh").read_text()
    line = next(l for l in script.splitlines() if l.strip().startswith("git add"))
    return [
        re.sub(r'^"|"$', "", token).replace("$result_dir", f"runs/{RUN_ID}")
        for token in line.split()[2:]
    ]


@pytest.mark.parametrize(
    "relative",
    [
        "command-args/preprocess_data.args",
        "command-args/supervised_train.args",
        "command-args/predict.args",
        "artifacts/data/manifest.yaml",
        "artifacts/model/model.pt",
        "logs/driver.log",
        "checkpoints/epoch1.pt",
        "predictions/predictions.parquet",
        "notes.txt",
        "a-directory-nobody-has-invented-yet/whatever",
    ],
)
def test_run_output_is_ignored(relative):
    """Everything under a run directory is excluded unless it is deliberately allowlisted.

    `command-args/*.args` is the case that matters and the one a list of known-bad subdirectories missed:
    it is written on every real submit and contains the run's dataset and task paths. The last entry
    checks the default rather than the enumeration -- something added under `runs/` later must arrive
    ignored, not trackable.
    """
    assert _is_ignored(f"runs/{RUN_ID}/{relative}")


def test_github_sync_can_still_stage_its_result(tmp_path):
    """The other direction, exercised through the real `git add` rather than through `check-ignore`.

    This is deliberately not a `check-ignore` assertion. `github-sync.sh` stages `manifests` as a
    *directory*, and an ignored parent silently reduces `git add` to staging nothing -- a warning and an
    empty index, not an error. An ignore rule broad enough to close the `.args` leak can therefore break
    result publication quietly, so the check has to be the command the script actually runs.
    """
    repository = tmp_path / "checkout"
    (repository / "runs" / RUN_ID / "manifests").mkdir(parents=True)
    (repository / "runs" / RUN_ID / "command-args").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    run = repository / "runs" / RUN_ID
    (run / "spec.yaml").write_text("dataset: {}\n")
    (run / "result.yaml").write_text("status: complete\n")
    (run / "manifests" / "predict.yaml").write_text("contract_version: '2'\n")
    (run / "command-args" / "predict.args").write_text("external_meds_dir=/data/MIMIC_MEDS\n")

    published = _published_paths()
    assert published, "could not read the staged paths out of scripts/github-sync.sh"
    subprocess.run(["git", "add", *published], cwd=repository, check=True, capture_output=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repository, capture_output=True, text=True
    ).stdout.split()
    assert sorted(staged) == [
        f"runs/{RUN_ID}/manifests/predict.yaml",
        f"runs/{RUN_ID}/result.yaml",
        f"runs/{RUN_ID}/spec.yaml",
    ], f"github-sync.sh would have staged {staged}"


def test_a_careless_add_all_cannot_pick_up_run_inputs(tmp_path):
    """`git add -A` in a worktree holding a completed run must not stage the dataset paths."""
    repository = tmp_path / "checkout"
    run = repository / "runs" / RUN_ID
    (run / "command-args").mkdir(parents=True)
    (run / "artifacts" / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (run / "command-args" / "predict.args").write_text("external_meds_dir=/data/MIMIC_MEDS\n")
    (run / "artifacts" / "data" / "shard.parquet").write_bytes(b"\x00")

    subprocess.run(["git", "add", "-A"], cwd=repository, check=True, capture_output=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repository, capture_output=True, text=True
    ).stdout.split()
    assert staged == [".gitignore"], f"a careless add staged run output: {staged}"


def test_slurm_job_logs_are_ignored():
    """Nothing sets `--output`, so SLURM writes these into the submit working directory."""
    assert _is_ignored("slurm-12345.out")
    assert _is_ignored("slurm-12345.err")


def test_the_args_files_would_have_carried_real_paths():
    """Why the `command-args` case is the serious one, pinned rather than asserted in a comment."""
    config = (ROOT / "slurm" / "config.sh").read_text()
    assert "external_meds_dir=${MEDS_ROOT" in config
    assert "external_labels_dir=${LABELS_DIR" in config
    assert 'RUN_ROOT="${RUN_ROOT:-runs}"' in config, (
        "RUN_ROOT defaults to a relative path, which is why run output lands in the checkout at all"
    )
