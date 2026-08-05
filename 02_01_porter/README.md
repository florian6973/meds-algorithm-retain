# PORTER

PORTER: language-grounded event representations for portable structured EHR foundation models — a frozen text-encoder input pathway feeding an autoregressive temporal Transformer, probed with L2 logistic regression.

A MEDS model generated from [MEDS_model_template](https://github.com/mmcdermott/MEDS_model_template)
(`probe` profile). It exposes the standard `meds-model` command interface and is contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

## What this model is

A reimplementation of **PORTER** ([arXiv:2606.24102](https://arxiv.org/abs/2606.24102), Guo, Yan, Vettese
& Sung, 2026) over MEDS data. PORTER has no public repository and no Python package, so nothing here
wraps an upstream implementation — every element is built from the paper, and
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) is the ledger of what was ported, adapted or
omitted. **Read it before comparing any number here to the paper's.**

The idea is to stop encoding clinical events as indices into a fixed vocabulary. Instead:

1. every vocabulary entry's **natural-language description** is encoded once by a *frozen* text encoder
   (BioLORD-2023) and cached as a lookup table — so a code the model never saw still has a representation,
   which is what lets one model transfer across institutions and schemas;
2. numeric values enter through a separate **FiLM** pathway that scales and shifts that text
   representation, rather than through the vocabulary;
3. a **causal temporal Transformer** (SwiGLU, RMSNorm, RoPE over patient age in days, alternating global
   and 128-wide local attention) is pretrained by next-event prediction over *composite* tokens — concept
   × value decile — then frozen. Only the label is quantized; the input stays continuous, and that
   distinction is the difference between PORTER and the NumText ablation the paper reports as worse;
4. an **L2-regularised logistic regression** is fitted on the frozen representation at each prediction
   time — which is exactly the `probe` DAG below.

Consequences worth knowing before you run it:

- **`pretrain` downloads the text encoder** (~0.4 GB from HuggingFace, cached under `~/.cache/huggingface`)
  and encodes every code description once. `infer` and `predict` do **not**: the encoded table is a buffer
  inside the pretrained checkpoint. Use `model.text_encoder=random` for the paper's own non-semantic
  control arm (§2.6.1), which needs no network.
- **The default is the published model**: 28 × 768, 115,649,280 backbone parameters, 16,384-event
  context. It needs a GPU (≈13 GiB at full context in bf16 — see `IMPLEMENTATION_REPORT.md` §6.6).
  `model=small` is a 0.46 M toy of the same shape for laptops, and is what the test suite runs.
- **`preprocess_data` runs a pipeline by default.** `pipelines/quantiles.yaml` fits the per-code deciles
  that the composite pretraining label is defined against. `pipeline=null` skips it, and the label then
  degrades to the bare code — a deviation from the paper, not a preference.
- **Demographics ride in the sequence.** `static_inclusion_mode: PREPEND` puts MEDS static measurements
  (sex, race, …) at the front of each timeline as one event, which is how Table S3 treats them. It raises
  on a cohort with no statics at all — set `PORTER_STATIC_INCLUSION=OMIT` for such a dataset.
- **The event descriptions matter as much as the architecture.** A MEDS dataset whose
  `metadata/codes.parquet` has no `description` column falls back to the code string, and the whole
  premise of the model — semantic grounding — is then absent. Check that column before drawing conclusions.

## Install

```bash
uv sync --all-extras
```

`flex_attention` (the sliding-window kernel, see below) compiles through Triton, which builds a small C
extension at runtime — so the interpreter needs its development headers. A uv-managed Python has them; a
bare system `python3` often does not, and the model then falls back to a dense attention mask. If
`uv run pytest tests/test_attention.py -rs` reports a skip about "Triton cannot build its runtime shim",
recreate the environment with `uv venv --clear --python 3.12` and re-sync.

## Running on a cluster

```bash
$EDITOR slurm/env.sh                # MEDS_ROOT, LABELS_DIR, OUTPUT_DIR, account, partition
bash slurm/01_preflight.sh          # check the dataset BEFORE spending a GPU-day on it
bash slurm/00_prefetch_encoder.sh   # once, on the LOGIN node — compute nodes are usually offline
bash slurm/submit.sh                # preprocess -> pretrain -> infer+probe+predict, chained on afterok
```

If `sbatch` says **"Invalid account or account/partition combination specified"**, the account either
does not exist or is not permitted on `GPU_PARTITION`. Diagnose without queueing anything:

```bash
sacctmgr show assoc user=$USER format=account,partition,qos   # what you are actually allowed
DRY_RUN=1 bash slurm/submit.sh                                # sbatch --test-only, validates and exits
```

Set `SLURM_ACCOUNT` to an account that listing shows, or leave it **empty** — `submit.sh` then omits
`--account` entirely and lets SLURM use your default association, which is right on most clusters.

`01_preflight.sh` trains nothing and takes seconds. It exists for one question above all: **does
`metadata/codes.parquet` have a `description` column?** PORTER's entire input pathway is that text. An
extract without it still preprocesses, still trains, still produces an AUROC — and is not PORTER. The
script prints the strings the frozen encoder will actually receive, so you can see what the model reads
before you commit to it, and it also sizes `batch_size` against the 16,384-event budget from the real
timeline lengths.

`slurm/env.sh` is the only file to edit; every job sources it. **Edit the file rather than setting
variables in your shell** — or if you do set them, `export` them:

```bash
MEDS_ROOT=/other/dataset
bash slurm/submit.sh                      # IGNORED: a child process sees only exported variables

export MEDS_ROOT=/other/dataset           # works
MEDS_ROOT=/other/dataset bash slurm/...   # also works (inline prefix exports for that command)
```

`submit.sh` prints the resolved dataset, labels, output, account and partitions before it queues
anything, and warns about paths that do not exist, so a value that failed to take is visible in the first
four lines rather than in tomorrow's log. The three things that most often go wrong
on a cluster are handled there and worth knowing about:

- **The text encoder download.** `pretrain` fetches BioLORD-2023 (~0.4 GB) from HuggingFace. Nothing
  downstream does — the encoded table travels inside the checkpoint — but that first job will hang or
  fail on a node with no outbound network. `00_prefetch_encoder.sh` populates `$HF_HOME` from the login
  node and the jobs then run with `HF_HUB_OFFLINE=1`.
- **Compile caches.** `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` default into `$OUTPUT_DIR`, not
  `$HOME`. On clusters where home is small or not writable from compute nodes, leaving them at the
  default is what silently drops you onto the slow attention path.
- **Precision.** The paper trains bf16 mixed (§2.4.4); this repository defaults to fp32 because the test
  tiers run on CPU. `slurm/env.sh` sets `trainer.precision=bf16-mixed` for you.

### Attention, and what "flash attention" means here

There is nothing to install. The backbone alternates global and 128-wide local attention (Table S4), and
each is served differently:

| layer | how | memory |
|---|---|---|
| global (even depths) | `scaled_dot_product_attention(..., is_causal=True)` — **no mask tensor**, so PyTorch is free to dispatch its built-in FlashAttention kernel | linear in context |
| local (odd depths) | `flex_attention` with a block-sparse causal-window mask, `torch.compile`d | linear in context |
| local, fallback | a cached `L × L` band mask through SDPA | quadratic (268 MB of mask at 16,384) |

Dropping the padding mask from the global layers is what makes the flash kernel eligible at all, and it
is sound only because meds-torch-data right-pads: with causal attention a real query can never reach a
pad key. `porter.model.require_right_padding` checks that on every forward rather than assuming it, and
`tests/test_attention.py` proves padding is inert.

`model.attention_impl` is `auto` (block-sparse on CUDA at long context, dense otherwise), `dense` (the
reference implementation), or `flex` (force the kernel and raise instead of falling back). **Run one
short job with `model.attention_impl=flex` before committing a GPU-day**: `auto` degrades quietly to the
quadratic path if Triton cannot build on your nodes, and `flex` is how you find that out in a minute
rather than from a memory profile.

Measured on the published geometry, batch of one patient, bf16 (`IMPLEMENTATION_REPORT.md` §6.6):
20.2 GiB peak at 16,384 context on the dense path, 13.0 GiB on the block-sparse one.

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
meds-model pretrain         input_data_dir=runs/data output_pretrained_model_dir=runs/models/pretrained
meds-model infer            input_data_dir=runs/data external_labels_dir=$LABELS_DIR input_pretrained_model_dir=runs/models/pretrained
meds-model supervised_train input_data_dir=runs/data external_labels_dir=$LABELS_DIR input_inference_subdir=inference/embeddings \
                            output_supervised_model_dir=runs/models/supervised
meds-model predict          input_data_dir=runs/data external_labels_dir=$LABELS_DIR input_supervised_model_dir=runs/models/supervised \
                            output_predictions_dir=runs/predictions
```

Any extra argument is a Hydra override, e.g.

```bash
meds-model pretrain trainer.precision=bf16-mixed         # the published scale is already the default
meds-model pretrain model=small                          # 0.46 M, for a laptop smoke test
meds-model pretrain model.text_encoder=random            # §2.6.1's non-semantic control arm
meds-model pretrain model.attention_impl=dense           # skip the compiled kernel
meds-model supervised_train --multirun model.l2=1,0.1,0.01,0.001,0.0001   # the paper's C grid
```

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

### Reproducibility

`pretrain` and `supervised_train` seed with `seed_everything(seed, workers=True)` (`seed: 0`; `null` skips
seeding entirely), `num_workers` defaults to `0`, and the Trainer ships `deterministic: warn` and
`benchmark: false`. What that buys is worth stating exactly, because it is less than "reproducible":

> **same seed + same config + same environment → same metric.**

Changing `num_workers` or `batch_size` alters RNG consumption order and legitimately changes results, and
reproducibility across different hardware or library versions is not reachable at all.

```bash
meds-model supervised_train trainer.deterministic=true   # raise instead of warn on a non-deterministic op
meds-model supervised_train trainer.precision=bf16-mixed # trade precision for speed
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set for you — without it `deterministic=true` raises on CUDA at the
first matmul rather than at startup. Float32 matmul precision is left at torch's default (`highest`); this
repository does not quietly run fp32 matmuls in bfloat16 to buy speed. If you need that, `precision` above
is the better lever, and a model port that requires TF32 specifically should set it in `model.py` and
record it.

### One contract defect this repository works around

`meds_model_base.commands._runtime.run_predict_step` pairs the index (`dataset.schema_df` order) with
model outputs (dataloader iteration order), assuming the loaders it uses do not shuffle. meds-torch-data's
`Datamodule.train_dataloader()` **does** shuffle, so the default `infer` writes every **train-split** row
of `artifacts.parquet` against another subject's key — silently, and differently on each run.

`porter.commands.OrderStableInferCommand` replaces it with a `shuffle=False` loader for every split, and
`tests/test_infer_alignment.py` fails if the misalignment returns. Measured on the synthetic signal
fixture, 5-fold CV AUROC on the materialized train-split embeddings goes from **0.49 (chance)** to
**1.00**. Filed upstream as
[MEDS_model_template#21](https://github.com/florian6973/MEDS_model_template/issues/21); delete the
subclass once the contract is fixed. Details in
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) §5.1.

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

- **`src/porter/model.py`** — the model. `Model` is PORTER's encoder (`build_text_table` →
  `event_representations` → `backbone_states` → `compute_loss` / `encode`); `Probe` is the downstream
  logistic regression. Every element cites the paper section it comes from.
- **`src/porter/configs/model/`** — `default.yaml` (the published 115.6 M geometry), `small.yaml` (a
  test fixture), `probe.yaml` (the head). `optimizer/porter.yaml` carries §2.4.4's AdamW settings.
- **`slurm/`** — the cluster jobs; `env.sh` is the only file to edit.
- **`src/porter/commands.py`** — which commands this model supports.
- **`src/meds_model_base/`** — the vendored contract (managed by `copier update`; don't edit). It holds
  the command graph, manifests, schemas and a small MEDS-batch adapter layer — no models.

## Testing

```bash
uv run pytest -m "not slow" -rs   # doctests + CLI smoke + the end-to-end conformance test
uv run pytest -m slow -rs         # designed-signal learnability, with a negative control
uv run pytest -m meds_dev -rs     # opt-in: clones MEDS-DEV and runs this model through it
```

The doctests in `model.py` are where the paper's numeric constants are pinned — the `[0, 1.5]` and
`[-2, 3]` clips, the divisor of 7, the FiLM gating, the 115.6 M parameter arithmetic. They run without a
network. The measured results of every tier are recorded in
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) §6.

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
