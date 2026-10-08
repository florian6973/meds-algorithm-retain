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

The invariant relied on is **the staged-path line**, not the file. `scripts/github-sync.sh` is a per-model
Copier render and differs across ports in its branch name and commit message -- five distinct hashes
across this pilot -- while its `git add` line is character-identical in all of them. A port rendered for a
different profile could legitimately stage different evidence, which is exactly why the list is read from
the script instead of restated.
"""

from __future__ import annotations

import os
import re
import shutil
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


def _profile_commands() -> list[str]:
    """The commands this repository's selected profile runs, hence its manifest set."""
    from meds_model_contract import MODEL_PROFILES

    return [command.value for command in MODEL_PROFILES["supervised"]]


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
    for command in _profile_commands():
        (run / "manifests" / f"{command}.yaml").write_text("contract_version: '2'\n")
    (run / "command-args" / "predict.args").write_text("external_meds_dir=/data/MIMIC_MEDS\n")

    published = _published_paths()
    assert published, "could not read the staged paths out of scripts/github-sync.sh"
    subprocess.run(["git", "add", *published], cwd=repository, check=True, capture_output=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repository, capture_output=True, text=True
    ).stdout.split()
    expected = sorted(
        [f"runs/{RUN_ID}/manifests/{command}.yaml" for command in _profile_commands()]
        + [f"runs/{RUN_ID}/result.yaml", f"runs/{RUN_ID}/spec.yaml"]
    )
    # A real `supervised` run publishes one manifest per command plus spec and result -- five files, not
    # three. The count is derived from the profile rather than written down, because a fixture that
    # publishes one manifest would still pass every rule here (`manifests/**` is allowlisted wholesale)
    # while a move to per-file exceptions would silently lose the other two.
    assert sorted(staged) == expected, f"github-sync.sh would have staged {staged}, expected {expected}"


def test_a_careless_add_all_cannot_pick_up_run_inputs(tmp_path):
    """`git add -A` in a worktree holding a *complete* run must not stage inputs or host paths.

    Asserted as a negative property rather than as an exact staged set. The publishable files are
    trackable by design, so whether they appear depends only on whether they happen to be committed
    already -- an exact-set assertion would encode this repository's commit history and go red on a fresh
    checkout holding a completed-but-unpublished run. That is a false red of the kind usually "fixed" by
    loosening the assertion until it stops meaning anything.

    The property that matters is that nothing carrying run *inputs* is staged, and that no host dataset
    path reaches the index in any file's content.
    """
    host_path = "/groups/example/data/MIMIC_MEDS"
    repository = tmp_path / "checkout"
    run = repository / "runs" / RUN_ID
    for directory in ("command-args", "artifacts/data", "logs", "checkpoints", "manifests"):
        (run / directory).mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    # A finished run: publishable evidence alongside inputs, artifacts, logs and checkpoints.
    (run / "spec.yaml").write_text("dataset: {}\n")
    (run / "result.yaml").write_text("status: complete\n")
    for command in _profile_commands():
        (run / "manifests" / f"{command}.yaml").write_text("contract_version: '2'\n")
    for command in _profile_commands():
        (run / "command-args" / f"{command}.args").write_text(f"external_meds_dir={host_path}\n")
    (run / "artifacts" / "data" / "shard.parquet").write_bytes(b"\x00")
    (run / "logs" / "driver.log").write_text(f"reading {host_path}\n")
    (run / "checkpoints" / "model.pt").write_bytes(b"\x00")

    subprocess.run(["git", "add", "-A"], cwd=repository, check=True, capture_output=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repository, capture_output=True, text=True
    ).stdout.split()
    for forbidden in ("command-args", "artifacts", "logs", "checkpoints"):
        assert not any(forbidden in path for path in staged), f"{forbidden} was staged: {staged}"

    # Content, not just paths: the dataset location must not reach the index inside any staged file.
    content = subprocess.run(
        ["git", "diff", "--cached"], cwd=repository, capture_output=True, text=True
    ).stdout
    assert host_path not in content, "a host dataset path reached the index"


def test_slurm_job_logs_are_ignored():
    """Nothing sets `--output`, so SLURM writes these into the submit working directory."""
    assert _is_ignored("slurm-12345.out")
    assert _is_ignored("slurm-12345.err")


def test_the_args_files_would_have_carried_real_paths():
    """Why the `command-args` case is the serious one, pinned rather than asserted in a comment."""
    config = (ROOT / "slurm" / "config.sh").read_text()
    assert "external_meds_dir=${MEDS_ROOT" in config
    assert "external_labels_dir=${LABELS_DIR" in config
    assert 'RUN_ROOT="${RUN_ROOT:-$_retain_model_root/runs}"' in config, (
        "RUN_ROOT must anchor to the model directory; a relative default sends output to the caller's "
        "working directory, which this model's .gitignore cannot reach"
    )


def test_the_ignore_rules_the_publisher_and_the_submitter_agree_on_one_path():
    """The security property is anchored on a literal path that three files must agree about.

    `.gitignore` excludes `runs/**`; `slurm/config.sh` defaults `RUN_ROOT` to `runs`; and
    `scripts/github-sync.sh` uses `result_dir="runs/$run_id"` as the in-tree *publication* path. The
    publisher now reads `RUN_ROOT` to find where a run actually lives and copies it in, so the three
    still have to agree about this one literal -- it is where the copy lands and what the ignore rules
    allowlist -- but a run is no longer required to live there.

    Two things break if they diverge. Pointing `RUN_ROOT` at another *relative* path puts run output back
    inside the checkout at a location these ignore rules do not cover, and the `command-args` leak
    returns. Pointing it at an absolute scratch path is safe for the leak but makes the publisher search
    `./runs/` on a complete run, find nothing, and exit reporting "refusing to publish an incomplete run"
    -- an error naming the run when the cause is the configuration.

    This asserts the agreement so a change to any one of the three is a failing test rather than a quiet
    change in what can be committed.
    """
    publisher = (ROOT / "scripts" / "github-sync.sh").read_text()
    config = (ROOT / "slurm" / "config.sh").read_text()
    ignore = (ROOT / ".gitignore").read_text()

    published_root = re.search(r'result_dir="([^/"]+)/', publisher)
    assert published_root, "could not read the run root out of github-sync.sh"
    configured_root = re.search(r'RUN_ROOT="\$\{RUN_ROOT:-\$_retain_model_root/([^}]+)\}"', config)
    assert configured_root, (
        "could not read an anchored RUN_ROOT default out of slurm/config.sh; it must resolve against the "
        "model directory rather than the caller's working directory"
    )

    assert published_root.group(1) == configured_root.group(1), (
        f"github-sync.sh publishes from {published_root.group(1)!r} but submit.sh writes to "
        f"{configured_root.group(1)!r}; a complete run would fail to publish"
    )
    assert f"{published_root.group(1)}/**" in ignore, (
        f"the ignore rules do not cover {published_root.group(1)!r}, which is where run output lands"
    )


def test_run_output_lands_inside_the_model_whatever_the_working_directory(tmp_path):
    """Defect #9: the template's relative `RUN_ROOT` follows the caller's cwd, not the model.

    Invoked from the repository or worktree root, `submit.sh` with a relative default writes
    `<worktree>/runs/` -- a sibling of this model, which this model's `.gitignore` cannot reach. Measured
    before the fix: `git check-ignore runs/probe/command-args/x.args` from the worktree root reported it
    TRACKABLE while the in-model path was IGNORED, so the dataset paths in those files were committable
    with no misconfiguration at all -- no `RUN_ROOT` override, no cluster, just the wrong directory.

    A `.gitignore` at `models/<name>/` cannot cover a sibling path, so this is fixed where it originates
    by anchoring `RUN_ROOT` to the model directory rather than by widening the ignore rules.
    """
    result = subprocess.run(
        ["bash", str(ROOT / "slurm" / "submit.sh"), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{ROOT / '.venv' / 'bin'}{os.pathsep}{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    written = re.findall(r"MEDS_MODEL_ARGS_FILE=([^,\\ ]+)", result.stdout)
    assert written, f"no args paths in dry-run output: {result.stdout[:400]}"
    for path in written:
        assert path.startswith(str(ROOT / "runs")), (
            f"run output would land at {path}, outside {ROOT}/runs where the ignore rules apply"
        )


def test_a_run_stored_off_tree_is_published_from_where_it_actually_lives(tmp_path):
    """The defect this replaced: run output had to live inside the git work tree to be publishable.

    `scripts/github-sync.sh` hardcoded `runs/<id>`, so `slurm/config.sh` refused an off-tree `RUN_ROOT`
    (exit 5) to avoid producing a run that completes and then cannot be published. The cost was paid on
    the cluster: completed results sat inside a checkout, where a routine `git clean -fdx` deletes them
    without warning and collecting them meant reaching into a git worktree. Reported by the cluster
    session against this port and BEHRT; the three ports that write off-tree have the mirror image of
    the same coupling -- results safe, publisher unable to find them.

    This runs the publisher end to end against a run that lives outside the repository entirely and
    asserts the evidence arrives committed. `git push` has no remote here and fails by design; the
    commit is what matters and it precedes the push.
    """
    repository = tmp_path / "checkout"
    (repository / "scripts").mkdir(parents=True)
    (repository / "slurm").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    for name, value in (("user.email", "t@example.com"), ("user.name", "T")):
        subprocess.run(["git", "-C", str(repository), "config", name, value], check=True)
    shutil.copy(ROOT / "scripts" / "github-sync.sh", repository / "scripts" / "github-sync.sh")
    shutil.copy(ROOT / "slurm" / "config.sh", repository / "slurm" / "config.sh")
    shutil.copy(ROOT / ".gitignore", repository / ".gitignore")
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "base"], check=True, capture_output=True
    )

    off_tree = tmp_path / "cluster-scratch" / "runs"
    run = off_tree / RUN_ID
    (run / "manifests").mkdir(parents=True)
    (run / "spec.yaml").write_text("dataset: {}\n")
    (run / "result.yaml").write_text("status: complete\n")
    for command in _profile_commands():
        (run / "manifests" / f"{command}.yaml").write_text("contract_version: '2'\n")
    # One manifest is a SYMLINK into the cluster filesystem. Without this the `120000` assertion below
    # has nothing to catch and only looks like coverage -- a control blind to the condition it guards.
    # `cp -R` would preserve it and commit a host path as blob content; `cp -RL` must dereference.
    elsewhere = tmp_path / "elsewhere" / "linked.yaml"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("contract_version: '2'\n")
    linked = run / "manifests" / f"{_profile_commands()[0]}.yaml"
    linked.unlink()
    linked.symlink_to(elsewhere)
    # The leak-carrying file must exist in the source and must NOT be among what is published.
    (run / "command-args").mkdir()
    (run / "command-args" / "predict.args").write_text("external_meds_dir=/data/MIMIC_MEDS\n")

    result = subprocess.run(
        ["bash", "scripts/github-sync.sh", RUN_ID],
        cwd=repository,
        capture_output=True,
        text=True,
        env={**os.environ, "RUN_ROOT": str(off_tree)},
    )
    assert "refusing to publish an incomplete run" not in result.stderr, (
        f"the publisher could not find a complete run stored at {off_tree}: {result.stderr}"
    )

    committed = subprocess.run(
        ["git", "-C", str(repository), "show", "--name-only", "--format=", f"runs/retain/{RUN_ID}"],
        capture_output=True,
        text=True,
    ).stdout.split()
    expected = sorted(
        [f"runs/{RUN_ID}/manifests/{command}.yaml" for command in _profile_commands()]
        + [f"runs/{RUN_ID}/result.yaml", f"runs/{RUN_ID}/spec.yaml"]
    )
    assert sorted(committed) == expected, (
        f"published {committed} from an off-tree run root, expected {expected}"
    )
    assert not any("command-args" in path for path in committed), (
        "the args files carry external_meds_dir and must never be published, off-tree or not"
    )
    # Copied, not linked: a symlink would be committed as a 120000 blob holding the host path.
    modes = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", "-r", f"runs/retain/{RUN_ID}"],
        capture_output=True,
        text=True,
    ).stdout
    assert "120000" not in modes, f"published evidence contains a symlink rather than content:\n{modes}"


def test_the_publisher_copies_rather_than_links_published_evidence(tmp_path):
    """`.gitignore` cannot defend against publication by symlink, so the publisher is checked instead.

    Measured: `git add` refuses a path outside the repository -- absolute or relative, exit 128, "is
    outside repository" -- which is what anchors these rules to the checkout's literal `runs/`. But that
    refusal is not total. A symlink *inside* the checkout pointing at an off-checkout run is staged
    successfully (exit 0) and its blob content is the **target path**: staging
    `runs/<id>/spec.yaml -> /scratch/<id>/spec.yaml` puts the host path into the index, and it is
    allowlisted precisely because a real `spec.yaml` there must be publishable.

    No ignore rule can separate those two cases, so the property is enforced one level up: publication
    must copy. This is a tripwire rather than a fix -- it fires if `github-sync.sh` ever gains symlink
    publication, which is a plausible implementation of making it follow a relocated `RUN_ROOT`.
    """
    publisher = (ROOT / "scripts" / "github-sync.sh").read_text()
    assert not re.search(r"\bln\s+-s", publisher), (
        "github-sync.sh appears to publish by symlink; a symlink under runs/ is staged with the host "
        "path as its content, and no .gitignore rule can distinguish it from the real file"
    )

    # And demonstrate the hazard concretely, so the tripwire's reason cannot be mistaken for theory.
    repository = tmp_path / "checkout"
    (repository / "runs" / RUN_ID).mkdir(parents=True)
    outside = tmp_path / "scratch" / RUN_ID
    outside.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (outside / "spec.yaml").write_text("dataset: /groups/example/MIMIC_MEDS\n")
    (repository / "runs" / RUN_ID / "spec.yaml").symlink_to(outside / "spec.yaml")

    subprocess.run(["git", "add", "-A"], cwd=repository, check=True, capture_output=True)
    staged_content = subprocess.run(
        ["git", "diff", "--cached"], cwd=repository, capture_output=True, text=True
    ).stdout
    assert str(outside) in staged_content, (
        "the symlink hazard this test guards against no longer reproduces; re-derive the guard before "
        "relaxing it"
    )
    # git records the distinction the ignore rules cannot: a symlink is mode 120000, a real file 100644.
    modes = _staged_modes(repository)
    assert modes[f"runs/{RUN_ID}/spec.yaml"] == "120000", modes


def _staged_modes(repository: Path) -> dict[str, str]:
    """`path -> index mode` for everything staged under `runs/`."""
    listing = subprocess.run(
        ["git", "ls-files", "-s", "--", "runs/"], cwd=repository, capture_output=True, text=True
    ).stdout.splitlines()
    return {line.split("\t", 1)[1]: line.split()[0] for line in listing if line.strip()}


def test_nothing_published_is_a_symlink(tmp_path):
    """Assert the outcome, not the implementation: no staged run file may be a symlink.

    The source tripwire above asserts that today's `github-sync.sh` does not link. This asserts the
    property that matters regardless of how a link could arrive -- `cp -s`, `rsync -l`, a helper the
    script shells out to, or a reimplementation of the publisher. Git records the distinction that no
    `.gitignore` rule can express: mode `120000` for a symlink against `100644` for a real file, so the
    check is exact rather than heuristic.

    Note that only the symlinked-*file* case needs this. A symlinked *directory* is refused by git itself
    -- `fatal: ... is beyond a symbolic link`, exit 128 -- so it fails loudly and cannot reach the index.
    """
    repository = tmp_path / "checkout"
    run = repository / "runs" / RUN_ID
    (run / "manifests").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (run / "spec.yaml").write_text("dataset: {}\n")
    (run / "result.yaml").write_text("status: complete\n")
    for command in _profile_commands():
        (run / "manifests" / f"{command}.yaml").write_text("contract_version: '2'\n")

    subprocess.run(["git", "add", *_published_paths()], cwd=repository, check=True, capture_output=True)
    modes = _staged_modes(repository)
    assert modes, "nothing staged, so this assertion proves nothing"
    symlinked = [path for path, mode in modes.items() if mode == "120000"]
    assert not symlinked, (
        f"published as symlinks: {symlinked}. A symlink stages with the target path as its blob content, "
        "so an off-checkout run root would put host paths in the index; publication must copy."
    )


def _dry_run_args_paths(cwd, **env):
    result = subprocess.run(
        ["bash", str(ROOT / "slurm" / "submit.sh"), "--dry-run"],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{ROOT / '.venv' / 'bin'}{os.pathsep}{os.environ['PATH']}", **env},
    )
    return result, re.findall(r"MEDS_MODEL_ARGS_FILE=([^,\\ ]+)", result.stdout)


def test_the_anchor_resolves_identically_from_the_model_directory():
    """Row 1: the ordinary invocation is unchanged by anchoring."""
    result, paths = _dry_run_args_paths(ROOT)
    assert result.returncode == 0, result.stderr
    assert paths and all(path.startswith(str(ROOT / "runs")) for path in paths)


def test_an_explicit_relative_run_root_is_refused_from_an_unignored_subdirectory():
    """Row 4: anchoring must not accidentally launder a deliberate override into the right place.

    `RUN_ROOT=runs` from src/ resolves outside the model's ignored runs/, so it is refused. The
    same value from the repository root resolves to the model's own `runs/` and is accepted -- the guard
    compares resolved paths, not strings, so identical text can be right or wrong depending on cwd.
    """
    result, _ = _dry_run_args_paths(ROOT / "src", RUN_ROOT="runs")
    assert result.returncode == 4, "src/runs is in-tree but outside the ignored root runs/"

    accepted, paths = _dry_run_args_paths(ROOT, RUN_ROOT="runs")
    assert accepted.returncode == 0, accepted.stderr
    # Resolved against the invocation directory, not compared as a string: submit.sh passes through
    # whatever form RUN_ROOT had, so the same location can appear relative or absolute.
    assert paths and all(
        Path(ROOT / path).resolve().is_relative_to((ROOT / "runs").resolve()) for path in paths
    )


def test_the_anchored_run_root_is_what_the_publisher_resolves():
    """Row 5: fixing #9 must not walk into #8 by breaking the publisher's lookup.

    `scripts/github-sync.sh` hardcodes `result_dir="runs/$run_id"` relative to its working directory. An
    anchor that makes `RUN_ROOT` absolute but lands anywhere other than that produces a leak-free run
    that cannot be published -- trading a silent defect for a loud one. Rows 1 to 4 all pass for such an
    anchor, so this is the assertion that discriminates.
    """
    result, paths = _dry_run_args_paths(ROOT)
    assert result.returncode == 0, result.stderr
    assert paths, result.stdout[:400]
    run_id = re.search(r"RUN_ID=([^,\\ ]+)", result.stdout).group(1)

    # <root>/<run_id>/command-args/<command>.args -> the run directory submit.sh actually writes.
    written_run_dir = Path(ROOT / paths[0]).resolve().parent.parent
    # What `result_dir="runs/$run_id"` resolves to when github-sync.sh is run from the model directory,
    # which is how README.md documents invoking it.
    publisher_run_dir = (ROOT / "runs" / run_id).resolve()
    assert written_run_dir == publisher_run_dir, (
        f"submit.sh writes the run to {written_run_dir}, but github-sync.sh would publish from "
        f"{publisher_run_dir}; a completed run could not be published"
    )


def test_the_mode_guard_detects_a_link_that_arrived_without_anyone_linking(tmp_path):
    """The case a source-text tripwire cannot see: `cp -R` carries a symlink through.

    GNU `cp -R` preserves symlinks; only `-L` dereferences them. Measured: copying a `manifests/`
    directory whose `predict.yaml` is a link into scratch yields a link in the destination, and staging
    that gives mode `120000` with the host path as blob content, exit 0, no warning. Alongside it a real
    manifest stages `100644`.

    So a run whose manifest happens to be a link into scratch -- an ordinary cluster arrangement, not a
    workaround -- leaks without anyone attempting the symlink route, and nothing in `github-sync.sh`'s
    text is wrong. This asserts the mode check actually discriminates that case, rather than only passing
    on a clean fixture where it could not fail.
    """
    repository = tmp_path / "checkout"
    run = repository / "runs" / RUN_ID
    (run / "manifests").mkdir(parents=True)
    outside = tmp_path / "scratch"
    outside.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    (run / "spec.yaml").write_text("dataset: {}\n")
    (run / "result.yaml").write_text("status: complete\n")

    commands = _profile_commands()
    (outside / "linked.yaml").write_text("dataset: /groups/example/MIMIC_MEDS\n")
    (run / "manifests" / f"{commands[0]}.yaml").symlink_to(outside / "linked.yaml")
    for command in commands[1:]:
        (run / "manifests" / f"{command}.yaml").write_text("contract_version: '2'\n")

    subprocess.run(["git", "add", *_published_paths()], cwd=repository, check=True, capture_output=True)
    modes = _staged_modes(repository)
    linked = [path for path, mode in modes.items() if mode == "120000"]
    assert linked == [f"runs/{RUN_ID}/manifests/{commands[0]}.yaml"], (
        f"the mode guard did not detect the carried-through symlink; staged modes were {modes}"
    )
    # And the real manifests alongside it are unaffected, so the check is specific rather than blanket.
    assert all(
        modes[f"runs/{RUN_ID}/manifests/{command}.yaml"] == "100644" for command in commands[1:]
    )


def _publisher_can_publish_a_relocated_run() -> bool:
    """Whether this port's publisher can actually publish a run stored outside the checkout.

    A **conjunction**, deliberately. Resolving `RUN_ROOT` is not sufficient: `git add` refuses a path
    outside the repository (exit 128, "is outside repository"), so a resolve-only publisher fails at
    exactly the same point as the hardcode it replaced. It must also copy the evidence into the
    checkout's `runs/<id>/`, dereferencing as it goes -- `cp -R` preserves symlinks and would stage a
    host path as blob content.

    Keying the relaxation on "mentions RUN_ROOT" would demand relaxing the guard for a publisher that
    still cannot publish, which is the trapdoor this test exists to close, in executable form.
    """
    text = (ROOT / "scripts" / "github-sync.sh").read_text()
    resolves_run_root = "RUN_ROOT" in text
    dereferences_on_copy = re.search(r"\bcp\s+-[A-Za-z]*L\b", text) is not None
    return resolves_run_root and dereferences_on_copy


def test_a_run_root_inside_the_work_tree_but_elsewhere_is_always_refused(tmp_path):
    """The half of the guard that must survive any convention change.

    Both conventions in the fleet refuse this case, and it is the dangerous one: run output lands inside
    the work tree at a path the ignore rules do not cover, so the `command-args` files carrying
    `external_meds_dir` sit unignored. Asserted separately from the off-tree case because relaxing the
    guard to "accept off-tree" must not be implemented as "accept anything" -- that would be permissive
    about precisely the case neither convention permits, and silent where the current behaviour is loud.
    """
    inside_but_elsewhere = ROOT / "scratch-runs"
    result, _ = _dry_run_args_paths(ROOT, RUN_ROOT=str(inside_but_elsewhere))
    assert result.returncode == 4, (
        f"RUN_ROOT={inside_but_elsewhere} is inside the work tree and outside the ignored model runs/; "
        f"it must be refused under either convention with the would-leak status 4, got {result.returncode}"
    )
    assert "No convention in this fleet permits it" in result.stderr
    assert "external_meds_dir" in result.stderr, "the leaking case must name what would leak"


def test_off_tree_acceptance_tracks_this_ports_publisher(tmp_path):
    """Couple the guard to *this* port's publisher, not to a better one existing elsewhere.

    Today `scripts/github-sync.sh` hardcodes `result_dir="runs/$run_id"`, so an off-tree `RUN_ROOT` must
    be refused: it would otherwise produce a completed run the publisher cannot find. Another port has
    built a publisher that resolves `RUN_ROOT` and dereferences on copy; when that lands *here*, off-tree
    becomes the intended configuration and this assertion flips automatically rather than waiting for
    someone to remember.

    Adopting the relaxation from another port's file while this port still hardcodes would recreate the
    failure the guard prevents -- and doing it by removing the guard entirely would be worse than either
    convention, since the in-tree-elsewhere case above would stop being refused.
    """
    result, _ = _dry_run_args_paths(ROOT, RUN_ROOT=str(tmp_path / "off-tree-runs"))
    if _publisher_can_publish_a_relocated_run():
        assert result.returncode == 0, (
            "scripts/github-sync.sh both resolves RUN_ROOT and dereferences on copy, so an off-tree run "
            "root is now publishable and is the intended configuration; the guard should no longer "
            "refuse it -- while still refusing the in-tree-elsewhere case, which is asserted separately "
            "and must survive this relaxation"
        )
    else:
        # Status 5, not merely non-zero: the two refusals are different policies and a survey reading
        # only the status must distinguish "would leak" from "leak-safe but unpublishable".
        assert result.returncode == 5, (
            f"expected the unpublishable status 5, got {result.returncode}.\n"
            f"  THIS port's {ROOT / 'scripts' / 'github-sync.sh'} cannot yet publish a relocated run.\n"
            "  Relaxing requires ALL of: (a) it resolves RUN_ROOT, (b) it dereferences on copy -- `git "
            "add` refuses an out-of-repo path at exit 128, so resolving alone fails where the hardcode "
            "did, and `cp -R` would stage a symlink with the host path as its content -- and (c) the "
            "relaxation keeps refusing an in-tree-elsewhere RUN_ROOT (exit 4), the case that leaks and "
            "that no convention permits.\n"
            "  Another port having such a publisher is not this condition; the anchor is this file."
        )
