"""Model-owned profile adapters plus shared MEDS-DEV MIMIC demo provisioning."""

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
    """Execute the real `supervised` model profile and return an `E2EResult`.

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


def resolved_meds_dev_revision(meds_dev):
    """Verify **the provisioned clone at `meds_dev`** is at the pinned revision, and return its SHA.

    The anchor is stated first deliberately. This checks the clone that `provision_meds_dev` returns --
    the checkout that is actually installed into and executed from -- and *not* `MEDS_DEV_DIR`, which is
    only the source that clone may have been copied from. Checking the source would verify provenance
    rather than what ran, and would say nothing at all on the URL path. `MEDS_DEV_DIR` appears below as the
    remedy for a failure, never as the thing under test; a reader who takes the first mention of it as the
    anchor will misread this function, which has already happened once across the fleet.

    `provision_meds_dev` pins the revision only on the branch that clones from the URL::

        run([git, "clone", source or MEDS_DEV_URL, checkout], ...)
        if source is None:
            ref = os.environ.get("MEDS_DEV_REF", MEDS_DEV_PREDICATES_REF)
            run([git, "-C", checkout, "checkout", "--detach", ref], ...)

    Supplying `MEDS_DEV_DIR` takes the other branch, so the checkout is cloned at whatever the source's
    HEAD happens to be and never pinned. The assertion that follows it,
    `assert_meds_dev_predicates_capability`, checks that the predicates passthrough is *present*; a newer
    `main` still carrying it passes. That is a capability check, not a revision check, despite the name.

    A gate run against an unpinned MEDS-DEV is not the gate it reports being, so this turns the difference
    into an error and returns the SHA for the record. `MEDS_DEV_REF` overrides the expectation, matching
    what `provision_meds_dev` itself honours, so a deliberate revision change stays possible and explicit.

    `tests/integration.py` is byte-identical across the fleet and is not edited to achieve this.
    """
    import subprocess

    from .integration import MEDS_DEV_PREDICATES_REF

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(meds_dev), *args], capture_output=True, text=True
        )

    head = git("rev-parse", "HEAD")
    if head.returncode:
        # An opaque failure inside a guard against silent passes is itself a poor guard, so this says
        # what could not be established rather than only that a command failed.
        raise AssertionError(
            f"Could not establish which MEDS-DEV revision was provisioned: {meds_dev} is not a readable "
            f"git checkout, so the pin is unverified and this gate's result cannot be attributed to a "
            f"revision. git said: {head.stderr.strip() or '(no output)'}"
        )
    head = head.stdout.strip()

    expected = os.environ.get("MEDS_DEV_REF", MEDS_DEV_PREDICATES_REF)
    resolved = git("rev-parse", f"{expected}^{{commit}}")
    target = resolved.stdout.strip() if resolved.returncode == 0 else expected
    if head != target:
        raise AssertionError(
            f"MEDS-DEV is at {head}, not the pinned {target}. `provision_meds_dev` only pins when it "
            "clones from the URL, so a checkout supplied through MEDS_DEV_DIR arrives at its own HEAD; "
            "the capability assertion does not catch that because a newer revision still carries the "
            "predicates passthrough. Set MEDS_DEV_REF to change the pin deliberately."
        )
    return head


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
    print(f"MEDS-DEV revision: {resolved_meds_dev_revision(meds_dev)}")
    meds_dir, labels_dir = prepare_fixture(tmp_path / "fixture")
    output_dir = tmp_path / "meds-dev-run"

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

    The task invocation puts MEDS-DEV's virtualenv ``bin`` on ``PATH``; the dataset invocation must not,
    and does not. Verified at the pinned ref rather than assumed: ``datasets/__main__.py`` wraps its build
    in ``temp_env`` and passes ``env=env``, and ``temp_env`` prepends the dataset venv's ``bin`` to
    ``PATH`` whenever the dataset ships a ``requirements.txt`` -- MIMIC-IV does -- so that call site
    resolves its own scripts. ``tasks/__main__.py`` passes no ``env`` at all, so its ``run_in_env`` falls
    back to ``os.environ`` and its ``aces-cli`` / ``expand_shards`` subprocesses are not found, because
    running ``.venv/bin/meds-dev-task`` by absolute path never activates that virtualenv. Without this the
    dataset builds successfully and the task step then fails with exit 127, several minutes in.

    Patching the dataset call too would be harmless -- MEDS-DEV prepends its own venv over anything passed
    -- but it would encode a defect that does not exist, so it is deliberately not done here.
    """
    from .integration import RUN_TIMEOUT, run, venv_bin

    print(f"MEDS-DEV revision: {resolved_meds_dev_revision(meds_dev)}")
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
        # MEDS-DEV's task runner launches venv-installed helpers by bare name.
        # Prepend the pinned checkout so an ambient MEDS-DEV install cannot win.
        # `run()` merges this over os.environ, so nothing else about the environment is disturbed.
        env={"PATH": f"{venv_bin(meds_dev)}{os.pathsep}{os.environ['PATH']}"},
        timeout=RUN_TIMEOUT,
    )
    predicates_file = meds_dev / "src" / "MEDS_DEV" / "datasets" / "MIMIC-IV" / "predicates.yaml"
    return dataset_dir, labels_dir, predicates_file
