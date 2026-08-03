# TECO

Transformer-based Encounter-level Clinical Outcome (TECO) model for continuously updated ICU mortality prediction over MEDS data.

A MEDS model generated from [MEDS_model_template](https://github.com/mmcdermott/MEDS_model_template)
(`supervised` profile). It exposes the standard `meds-model` command interface and is contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

## The model

TECO comes from Kim *et al.*, ["A deep learning model for clinical outcome prediction using longitudinal
inpatient electronic health records"](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091),
*JAMIA Open* 8(2) ooaf026 (2025). It predicts ICU mortality from an encounter as it unfolds, with one
encounter-level sequence model that is re-scored as new measurements arrive rather than a separate model
per prediction time.

**No code was published with the paper**, so this is a reimplementation from its Methods section.
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) is the ledger: every element of the paper recorded
as ported, adapted or omitted, every assumption the paper leaves open, and what was actually run. Read it
before comparing anything here to the published numbers.

How it works, and what the MEDS port changes:

| | TECO | here |
|---|---|---|
| input | 11 named clinical variables, averaged into 15-minute intervals and carried forward, plus baseline variables on every interval | the same construction over the **code vocabulary** of whatever MEDS dataset is supplied (`IntervalGrid`), plus MEDS static measurements |
| encoder | 6 layers, 8 heads, `d_model` 512, GELU, dropout 0 | identical — and `teco_parameter_count` reproduces the paper's reported 18 928 130 parameters exactly |
| positions | relative positional encoding | fixed sinusoidal encoding of each interval's offset back from the prediction time (parameter-free, as the parameter count requires) |
| output | linear layer + softmax over 2 classes | identical |
| training | SGD (momentum 0.9), lr 0.01 halved every 50 epochs, batch 32, ≤500 epochs, gradient clipping 1.0, early stopping | identical, in `configs/` |
| horizons | 9 sub-models, one per prediction horizon (0, 12, … 96 h) | one trained model per task — the horizon lives in `external_labels_dir`, which is what MEDS-DEV passes |

The interval grid is the load-bearing part. TECO's `forward` reads a dense, fully-imputed
`[batch, intervals, variables]` matrix; a MEDS batch is a sparse, irregular measurement stream. Feeding
the stream to the encoder directly would be a different model, so `IntervalGrid` rebuilds the paper's grid
from `code` / `numeric_value` / `time_delta_days` — and `tests/test_interval_grid.py` checks it against a
naive dense implementation written straight from the paper's sentences.

## Install

```bash
uv sync --all-extras
```

## The command graph

Five commands form a DAG, not a fixed pipeline. Run `meds-model commands` to see which this model supports.

| Command | Reads | Writes |
|---|---|---|
| `preprocess_data` | an external MEDS dataset, sharded by split | `<data_dir>/patients/` |
| `pretrain` | patient data | a pretrained model directory |
| `infer` | patient data + a pretrained model + `external_labels_dir` | `<data_dir>/inference/<name>/` |
| `supervised_train` | patient data + `external_labels_dir` (+ at most one prior artifact) | a supervised model directory |
| `predict` | `external_labels_dir` + exactly one source | `predictions.parquet` |

`data_dir` is a shared workspace: `patients/` is written once and never modified, and `infer` appends a
sibling subdirectory. Every artifact directory carries a `manifest.yaml` recording
what produced it and from which inputs, and is published atomically — so a directory that exists is a
directory that finished.

### The input dataset must be sharded by split

`external_meds_dir` must look like `data/train/…`, `data/tuning/…`, `data/held_out/…`. That is what
`meds-dev-dataset` and the standard MEDS ETL produce, so a dataset from MEDS-DEV needs no preparation.
Anything else is refused up front, with the resharding pipeline to run printed in the error.

The reason is that meds-torch-data reads split membership from the shard path and nothing else, so this
layout is the only one it can tensorize into an artifact that still knows its splits. It is also why
`patients/` carries no `subject_splits.parquet`: split membership travels as the shard layout, and a
second copy could only ever disagree with it.

That matters if you use `pipeline=`. A MEDS-transforms pipeline that filters subjects — `filter_subjects`,
or a `filter_measurements` aggressive enough to empty a timeline — tokenizes them away, and their labels
are then dropped to match. `preprocess_data` warns and records `dropped_by_pipeline` in the manifest when
it happens. **Your evaluation then covers fewer subjects than the task defines, and nothing downstream can
tell**: `meds-dev-evaluation` only ever sees the predictions file. If you are comparing numbers against
another model, that difference belongs in your write-up.

## Usage

```bash
meds-model preprocess_data  external_meds_dir=$MEDS_ROOT output_data_dir=runs/data
meds-model supervised_train input_data_dir=runs/data external_labels_dir=$LABELS_DIR output_supervised_model_dir=runs/models/supervised
meds-model predict          input_data_dir=runs/data external_labels_dir=$LABELS_DIR input_supervised_model_dir=runs/models/supervised \
                            output_predictions_dir=runs/predictions
```

Any extra argument is a Hydra override, e.g.
`meds-model supervised_train trainer.max_epochs=50 model.d_model=128`.

### Logging

`pretrain` and `supervised_train` take a `logger` config group; `configs/logger/<name>.yaml` is what makes
`logger=<name>` valid, so **use these configs rather than constructing a logger in `model.py`** — a logger
built there is invisible to `logger=`, and a name with no file is a Hydra composition error.

Only `csv` is rendered here, because `use_wandb` and `use_mlflow` were both answered no. To add one, copy
a config into `configs/logger/` and add its package to `[project.optional-dependencies]` — or re-render
with `copier update` after flipping the answer in `.copier-answers.yml`.

A logger group member is a mapping of *name → logger config*: `build_trainer` instantiates every
`_target_` child of `cfg.logger`, so one file can enable several loggers at once, and a file with a bare
top-level `_target_` enables none of them without failing.

### Two rules worth knowing

**Sources are alternatives, not layers.** `supervised_train` and `predict` each accept several optional
input artifacts. Supplying more than one is an error rather than a precedence decision, and supplying one
the model does not handle is an error rather than a silently ignored argument.

**Predictions cover the whole index.** `predict` writes one row per task timepoint in the selected splits,
or fails. A short predictions file is indistinguishable downstream from a complete one, so it is never
written. Restrict the request with `splits=[held_out]` instead.

`predict` emits `predictions.parquet` in the
[meds-evaluation](https://github.com/kamilest/meds-evaluation) `PredictionSchema`
(`subject_id, prediction_time, predicted_boolean_probability`). **This repository ends at predicted
probabilities — it never reads ground-truth labels, and evaluation is a separate, shared tool.**

## Editing

**Start here: `src/teco/model.py`.** It holds `IntervalGrid` (the port of TECO's preprocessing) and
`Model` (the encoder, head and objective), and its module docstring explains how the architecture was
recovered from the paper's parameter count. The template generated the command DAG around it, not the
model itself.

- **`src/teco/commands.py`** — which commands this model supports.
- **`src/teco/configs/`** — `model/` the paper's architecture, `optimizer/sgd.yaml` its optimizer,
  `profile/default.yaml` its training schedule and the `static_inclusion_mode` the baseline variables
  need, `paths/` locations.
- **`src/meds_model_base/`** — the vendored contract (managed by `copier update`; don't edit). It holds
  the command graph, manifests, schemas and a small MEDS-batch adapter layer — no models.

### On a SLURM cluster

The paper's budget — 500 epochs of a 19M-parameter encoder over 384 interval positions — is a cluster job,
so the chain also ships as sbatch scripts:

```bash
$EDITOR slurm/config.sh       # partitions, account, MEDS_ROOT, LABELS_DIR, RUN_DIR, SETUP
./slurm/submit.sh --dry-run   # print the sbatch commands
./slurm/submit.sh             # queue the chain, one job per command, chained with --dependency
```

`preprocess_data` is queued without a GPU and the rest with one. TECO's nine prediction horizons are nine
tasks, not nine model configs, so the fan is a job array over label directories:

```bash
printf '%s\n' /labels/mortality_{0,12,24,36,48,60,72,84,96}h > tasks.txt
./slurm/submit.sh --tasks-file tasks.txt
```

`preprocess_data` still runs once — `patients/` is task-free — and the per-task stages chain element-wise.
See [`slurm/README.md`](slurm/README.md). These came from
[MEDS_model_template#14](https://github.com/florian6973/MEDS_model_template/pull/14); once it merges,
`copier update` reconciles them and `.copier-answers.yml` moves off `29f58b2`.

## Testing

```bash
uv run pytest -m "not slow" -rs   # grid equivalence, CLI smoke, and the end-to-end conformance test
uv run pytest -m slow -rs         # designed-signal learnability, with a negative control
uv run pytest -m meds_dev -rs     # opt-in: clones MEDS-DEV and runs this model through it
```

Nothing skips for want of a model any more — `model.py` is implemented, so both tiers run. The counts and
what they produced are in [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) §Verification.

The conformance runs train a deliberately small model (`CONFORMANCE_MODEL` in `tests/conftest.py`), because
`run_chain` pins the accelerator to CPU and the published geometry is 18.9M parameters over 384 interval
positions. That is a statement about the fixture's compute budget, not about the model: the shipped
defaults are the paper's, and `pytest src/teco/model.py` builds and checks the full-size one.

## Contributing to MEDS-DEV

`model.yaml` in this repository is the MEDS-DEV specification for this model's DAG. MEDS-DEV does not read
it from here — it discovers models with `importlib.resources.files("MEDS_DEV.models").rglob("*/model.yaml")`,
so a model is registered only by living inside the *installed* `MEDS_DEV` package. Fork MEDS-DEV, clone it,
install it editable (`pip install -e '.[dev,tests]'` — the PyPI distribution cannot be contributed to), then:

```bash
meds-model-add-to-meds-dev --meds-dev ~/MEDS-DEV --dry-run   # print the plan
meds-model-add-to-meds-dev --meds-dev ~/MEDS-DEV             # write it
```

This copies `model.yaml` verbatim, writes a `README.md` stub (MEDS-DEV requires one), and **rewrites**
`requirements.txt` — the one part that is not a copy. MEDS-DEV builds an isolated venv and runs
`uv pip install -r <file>` from its own working directory, so a local checkout must be referenced by
absolute path; the default writes `-e {this repo}`. Once the model is installable by URL,
edit this repository's `requirements.txt` (replace `YOUR_ORG`, or pin a release) and pass `--published` to
use it as-is.

The helper refuses to write if it finds `MEDS_DEV` importable from somewhere other than the checkout you
named — that is the non-editable install case, where the files would land somewhere nothing imports.

## Updating the template

```bash
copier update
```
