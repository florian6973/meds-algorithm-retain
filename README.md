# RETAIN

RETAIN ([Choi et al., NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html))
for datasets in the [Medical Event Data Standard](https://github.com/Medical-Event-Data-Standard/meds): a
reverse-time attention model that predicts a binary outcome from a subject's sequence of visits and
reports which codes drove each prediction.

- **What it runs:** `preprocess_data` (code vocabulary fitted on the training split), `train_supervised`
  (the whole model, for one task) and `predict_supervised` (one probability per requested key).
- **Model:** a PyTorch reimplementation of the official Theano code
  ([`mp2893/retain`](https://github.com/mp2893/retain) @ `9fd39c46`), with the official defaults, checked
  against a NumPy transcription of the official equations.
- **MEDS adaptation:** a visit is the set of codes in one time bucket (`visit_bucket_minutes`, default one
  day), and each history is cut at its prediction time before visits are formed. No dataset predicates
  are needed.
- **Status:** [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) lists what was verified, including the
  temporal-leakage checks, and what was not.

```bash
pip install retain-meds
retain-meds preprocess_data external_meds_dir=$MEDS_DIR output_data_dir=$RUN/data
retain-meds train_supervised input_data_dir=$RUN/data external_labels_dir=$LABELS_DIR \
    output_supervised_algorithm_dir=$RUN/algorithm
retain-meds predict_supervised input_data_dir=$RUN/data input_supervised_algorithm_dir=$RUN/algorithm \
    external_index_dir=$LABELS_DIR splits=[held_out] output_predictions_dir=$RUN/predictions
```

Settings are the fields of `RETAINConfig` (`src/retain_meds/config.py`); pass a YAML or JSON file of
overrides to every step with `config_file=<path>`.

## Python API and named command

The implementation is `src/retain_meds/retain_meds.py`, with configuration in `config.py`.
Both concrete classes are public package exports:

```python
from retain_meds import RETAIN, RETAINConfig

algorithm = RETAIN(RETAINConfig())
```

Run typed requests through `meds_algorithm.StepRunner.run(...)` so validation, measurements and
publication apply. Calling an implementation method directly bypasses that execution contract.
After installation, the algorithm's own `retain-meds` command runs the CLI:

```bash
retain-meds commands --profile supervised
```

RETAIN (Choi et al., NeurIPS 2016) reverse-time attention model for MEDS, a supervised classifier behind the MEDS algorithm contract.

This repository declares the **`supervised`** profiles under version 4 of the minimal MEDS algorithm
contract. The contract standardizes MEDS input, command meanings and arguments, artifact envelopes,
final predictions, measurements, MEDS-DEV packaging, and operational interfaces. The algorithm owns every
computational choice and every intermediate payload.

## Algorithm profile commands

- `supervised`: `preprocess_data → train_supervised → predict_supervised`

```bash
retain-meds profiles
retain-meds commands --profile supervised
retain-meds <step> --profile <profile> key=value ...
```

Each profile has `algorithms/<profile>/model.yaml`. Multiple profiles share one Python package and one
registry; select one explicitly for each run. `STUB_PROFILES` tracks implementation status separately.
Read [Software contract](SPEC.md) for exact class responsibilities and step inputs.

Intermediate artifacts may be tensors, files, databases, external-system references, or any other
implementation-owned representation. Generic tests inspect only their `manifest.yaml` envelopes. They
must not assume batches, Lightning, PyTorch, Parquet intermediate data, or iteration order.

## Profile runner (prototype)

Install `uv sync --extra orchestration --extra test`, then fill the input paths and new output directory
in `configs/runs/run.yaml`. Algorithm settings belong under `algorithm`; core validates them using
this repository's concrete Config class.

```bash
uv run retain-meds run --dry-run
uv run retain-meds run execution=local
# Use a NEW output_dir and configure your cluster's resources before submission:
uv run retain-meds run execution=slurm output_dir=/runs/new-cluster-run
```

A dry run prints all resolved arguments, paths, configuration and resources without writing files or
submitting jobs. Local execution uses fresh subprocesses; Submitit submits separate dependent SLURM
jobs. See the [shared runner guide](https://meds-algorithm.readthedocs.io/en/latest/running/#profile-runner-prototype)
for stage subsets, artifact reuse, resource configuration and prototype limitations. The existing
per-step CLI and shell launcher remain available for comparison. This prototype does not implement
Hydra multirun, multi-node training or checkpoint/resume.

## Publishing this algorithm

Follow [the release guide](RELEASING.md) for package metadata, one-time PyPI Trusted Publisher setup,
installed-wheel checks, and version-tag publication. All profiles ship in the `retain-meds`
distribution. Tests run in CI; publishing is an explicit workflow dispatch after implementation.

## Inputs and predicates

- `external_meds_dir` is the standard MEDS dataset boundary.
- `external_index_dir` supplies only `(subject_id, prediction_time)` to inference and scoring.
- `external_labels_dir` supplies permitted training targets or explicit in-context examples.
- `external_evaluation_labels_dir`, when supplied, is used only to attach truth after scores are fixed.
- `external_predicates_file` supplies dataset-specific predicates when useful. For MIMIC-IV, reuse the
  predicates from MEDS-DEV. This repository decides whether and how to interpret them.

A command that consumes predicates records their absolute path and SHA-256 digest in its manifest.
Predicates do not imply a standard feature representation.

## Implementing the algorithm

Read [AGENTS.md](AGENTS.md), [SPEC.md](SPEC.md), and
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) completely before changing implementation
files. The implementation source recorded when this repository was generated is:

```text
https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html and https://github.com/mp2893/retain@9fd39c46e44b3fdae3a667a47fbdd5ffbd87bff5
```

The algorithm-owned surface is `src/retain_meds/retain_meds.py` (the typed algorithm), `src/retain_meds/config.py`
(its immutable configuration), `algorithms/<profile>/model.yaml`, and
`tests/e2e.py`. Implement the selected profile methods, define exact intermediate compatibility checks in
`validate_inputs`,
finalize MEDS-DEV command strings, and complete the report. Remove a profile from `STUB_PROFILES` only after it runs end to end. Complete every declared profile.

For each generated test and its purpose, read [the test guide](docs/TESTING.md).

## Configuration and execution ownership

Declare typed fields/defaults in `RETAINConfig`, with `validate()` for constraints. Pass
`config_file=/path/config.yaml` to a step for a JSON/YAML values mapping; unknown fields fail. Core saves
the complete resolved config and original typed request as `config.json` and `request.json` in each
published artifact. These JSON envelopes can be reloaded with the corresponding explicit classes.

Algorithm methods receive typed requests, write payloads into `request.output.directory` (staging), and
return `PayloadMetadata`. `create_repository().run(...)` constructs a fresh algorithm and delegates to
core's runner. It owns measurements, prediction validation, optional post-score truth attachment,
metadata and publication. Never publish from inside algorithm methods or write the reserved metadata files.
Reopen fitted artifacts on each step. Test config/request round trips and execution in a fresh process.

## Verification

```bash
uv run pytest -rs
uv run pytest -m meds_dev -rs
uv run pytest -m real_data -rs
```

- The default suite runs the local end-to-end adapter and contract tests.
- `meds_dev` registers `algorithms/<profile>/model.yaml`, installs this repository from `requirements.txt` in an isolated
  environment, and executes MEDS-DEV's real command slots.
- `real_data` runs the full selected algorithm profile on the MEDS-DEV MIMIC-IV demo, the pinned
  `mortality/in_icu/first_24h` task, and MEDS-DEV's
  MIMIC predicates. Set `MEDS_DEMO_DIR` to reuse an existing demo and `MEDS_DEV_DIR` to reuse a checkout.

Automatic provisioning pins MEDS-DEV PR #330's tested commit, which has the `{predicates_path}`
passthrough (PR #325). Set `MEDS_DEV_REF` to use another
revision containing those capabilities. A checkout supplied through `MEDS_DEV_DIR` is checked and fails
clearly if it cannot populate them.

Both integration tiers explicitly `algorithm_stub`-skip until implementation is complete. A completed port
must have zero `algorithm_stub` skips and must account for every other skip using `-rs` output.

These commands are gates, not an unordered checklist. Follow the canonical staged workflow in
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md): local, MEDS-DEV, pre-demo report and human
review, MIMIC-IV demo, demo report and human review, authorized full-data run, then final report and GitHub
synchronization. Full-data cluster jobs run only from a clean, pushed commit and use the reviewed shared
task bundle for the selected dataset (for example, MIMIC-IV or NWICU in MEDS format), built upstream from
clinical ACES tasks or randomly sampled code-by-horizon tasks. The provided `real_data` test remains the MIMIC-IV demo regardless of that choice.

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
PROFILE=supervised ./slurm/submit.sh --dry-run
PROFILE=supervised ./slurm/submit.sh
scripts/github-sync.sh <run-id>
```

SLURM follows the selected algorithm profile and chains stages by successful dependencies. Structured evidence belongs
under `runs/<run-id>/`; raw logs, datasets, checkpoints, and credentials do not. GitHub synchronization
pushes a `runs/retain_meds/<run-id>` branch and optionally opens a draft PR. A failed push leaves the
result committed and recoverable locally.
