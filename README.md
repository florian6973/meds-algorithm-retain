# RETAIN

The model package lives at the root of this private repository. Clone
[`florian6973/meds-model-retain`](https://github.com/florian6973/meds-model-retain) and run installation, test, and SLURM commands
from the checkout root; `pyproject.toml`, `model.yaml`, `src/`, and `tests/` are here.
Use this repository for all further model development. Earlier commits retain the historical
`models/01_19_retain` layout; recorded historical paths and revisions remain provenance.

RETAIN implemented against the minimal MEDS model contract.

This repository implements the **`supervised`** model profile under version 2 of the minimal MEDS model
contract. The contract standardizes MEDS input, command meanings and arguments, artifact envelopes,
final predictions, measurements, MEDS-DEV packaging, and operational interfaces. The model owns every
computational choice and every intermediate payload.

## Model profile commands

```text
preprocess_data -> supervised_train -> predict
```

```bash
meds-model commands
meds-model <command> key=value ...
```

Intermediate artifacts may be tensors, files, databases, external-system references, or any other
implementation-owned representation. Generic tests inspect only their `manifest.yaml` envelopes. They
must not assume batches, Lightning, PyTorch, Parquet intermediate data, or iteration order.

## Inputs and predicates

- `external_meds_dir` is the standard MEDS dataset boundary.
- `external_labels_dir` supplies task prediction times and training labels where the model profile needs them.
- `external_predicates_file` is **not used**. This port is rendered `uses_predicates=false` and is
  predicate-free by construction: RETAIN's published input is a sequence of medical codes, which the MEDS
  `code` column already is, so there is nothing for a dataset predicate to resolve. Every command rejects
  the argument rather than accepting an input it would ignore, `model.yaml` declares no
  `{predicates_path}` slot, and this repository ships no `predicates.yaml`.

A command that consumes predicates records their absolute path and SHA-256 digest in its manifest. No
command here consumes one, so no such digest is recorded — recording provenance for an input that did not
influence the run would be worse than recording nothing.

## Implementing the model

Read [AGENTS.md](AGENTS.md), [SPEC.md](SPEC.md), and
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) completely before changing implementation
files. The implementation source recorded when this repository was generated is:

```text
cards/01_supervised_models/19_retain.md, https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html, and https://github.com/mp2893/retain
```

The model-owned surface is `src/retain/commands.py`, command configuration, `model.yaml`, and
`tests/e2e.py`. Implement every registered command, define exact intermediate compatibility checks,
finalize MEDS-DEV command strings, and complete the report. Keep `IS_STUB = True` until the actual model profile
runs end to end.

## Verification

```bash
uv run pytest -rs
uv run pytest -m meds_dev -rs
uv run pytest -m real_data -rs
```

- The default suite runs the local end-to-end adapter and contract tests.
- `meds_dev` registers `model.yaml`, installs this repository from `requirements.txt` in an isolated
  environment, and executes MEDS-DEV's real command slots.
- `real_data` runs the full selected model profile on the MEDS-DEV MIMIC-IV demo and the pinned
  `mortality/in_icu/first_24h` task. Set `MEDS_DEMO_DIR` to reuse an existing demo and `MEDS_DEV_DIR` to
  reuse a checkout. MEDS-DEV's MIMIC predicates are located and checked for presence by the test, but this
  model does not read them.

Until MEDS-DEV PR #325 is merged, automatic provisioning pins its tested predicates-passthrough commit.
Set `MEDS_DEV_REF` to use another revision containing that capability. A checkout supplied through
`MEDS_DEV_DIR` is checked and fails clearly if it cannot populate `{predicates_path}`.

Both integration tiers explicitly `model_stub`-skip until implementation is complete. A completed port
must have zero `model_stub` skips and must account for every other skip using `-rs` output.

These commands are gates, not an unordered checklist. Follow the canonical staged workflow in
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md): local, MEDS-DEV, pre-demo report and human
review, MIMIC-IV demo, demo report and human review, authorized full MIMIC-IV, then final report and GitHub
synchronization. Full-data cluster jobs run only from a clean, pushed commit and use the reviewed shared
`work/` task bundle produced by `meds-task-selection`.

Final predictions must conform to MEDS evaluation, contain unique `(subject_id, prediction_time)` keys,
and exactly cover the requested evaluation index. Tests compare the actual key sets; self-reported manifest
counts are not accepted as evidence. Row order is never an association mechanism.

## Measurements

Every command manifest records monotonic wall time and sampled live process-tree memory. On Linux, peak
PSS from `/proc/*/smaps_rollup` is the primary comparison metric; aggregate RSS is retained as a
diagnostic. The measurement method and sampling interval are explicit. GPU measurements and SLURM
`MaxRSS`, when available, are separate metrics. Missing measurements are `null`; observed zero is `0`.
Time and memory are evidence, not portable test thresholds.

## Cluster execution and GitHub communication

```bash
$EDITOR slurm/config.sh
./slurm/submit.sh --dry-run
./slurm/submit.sh
scripts/github-sync.sh <run-id>
```

SLURM follows the selected model profile and chains stages by successful dependencies. Structured evidence belongs
under `runs/<run-id>/`; raw logs, datasets, checkpoints, and credentials do not. GitHub synchronization
pushes a `runs/retain/<run-id>` branch and optionally opens a draft PR. A failed push leaves the
result committed and recoverable locally.
