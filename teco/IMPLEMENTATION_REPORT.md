# Implementation report — TECO → `teco`

Written per [`docs/PORTING-A-MODEL.md`](docs/PORTING-A-MODEL.md). It records what was ported from the
source, what was adapted, what was left out, and what was actually run.

**This port has an unusual starting condition: there is no source code.** TECO was published without a
repository and without a package, so there is nothing to wrap, nothing to diff against, and no upstream
loss function to assert equality with. Every element below was reconstructed from the paper's Methods
section — which means the failure mode this document exists to catch is not only "an element was silently
dropped" but also "an element was silently *invented*". Sections 3 and 4 separate the two: §3 is what the
paper states, §4 is what it does not state and this implementation had to choose.

---

## 1. Source

| | |
|---|---|
| paper | Kim *et al.*, "A deep learning model for clinical outcome prediction using longitudinal inpatient electronic health records", *JAMIA Open* **8**(2) ooaf026 (2025) — [DOI](https://doi.org/10.1093/jamiaopen/ooaf026), [PMC11984207](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984207/) |
| package | **none** — no PyPI package, no conda package |
| repository | **none** — the article carries no code-availability statement; the model card's targeted search (`models/cards/01_supervised_models/01_teco.md`, checked 2026-07-30) found no public repository |
| depended on as | nothing to depend on. Reimplemented from the Methods section; **no code vendored from anywhere** |
| reference implementation stated | PyTorch 1.8.1, single NVIDIA V100 |
| template | [MEDS_model_template](https://github.com/florian6973/MEDS_model_template) @ `29f58b2`, `supervised` profile (`preprocess_data → supervised_train → predict`) |
| data | development: 2 579 THR COVID-19 ICU encounters (private). External validation: MIMIC-IV ARDS (n=2 799) and sepsis (n=6 622) |

The `supervised` profile is the right shape: the paper trains the mortality predictor directly from
labelled ICU encounters, with no reusable pretraining stage, which is what the model card classifies it as.

---

## 2. How the architecture was recovered

The paper states 6 encoder layers, 8 attention heads, an embedding dimension of 512, GELU, dropout 0, a
"linear classification layer with the Softmax activation function", and **18 928 130 trainable
parameters**. It does not state the feed-forward width, whether the encoder stack carries a final
LayerNorm, whether the embedding has a bias, or how many parameters the positional encoding holds.

The reported total settles all four, because the natural decomposition closes on it exactly:

```
per encoder layer (d=512, ff=2048):
  self-attention   4d² + 4d              = 1 050 624
  feed-forward     2·d·ff + ff + d       = 2 099 712
  two LayerNorms   4d                    =     2 048
                                          -----------
                                            3 152 384   × 6 layers = 18 914 304
  bias-free Linear(25, 512) embedding     =    12 800
  Linear(512, 2) head                     =     1 026
                                          -----------
                                            18 928 130   ← exactly the published number
```

Three consequences, each of which this implementation follows and each of which is a *derivation*, not a
quotation:

1. **`dim_feedforward = 2048`** — PyTorch's own default, which is what a 1.8.1-era
   `nn.TransformerEncoderLayer(512, 8)` would have given.
2. **No final LayerNorm** on the stack (`nn.TransformerEncoder(..., norm=None)`); adding one would put the
   count 1 024 over.
3. **The positional encoding holds no parameters at all.** The residual after the encoder and the head is
   12 800 = 512 × 25 exactly — an integer multiple of the embedding width, with nothing over. Any learned
   positional parameters would have to be carved out of that, and a learned table over 384 intervals alone
   would be 196 608. So the paper's "relative positional encoding" is a *fixed* function of position, which
   is why this implementation uses a sinusoidal encoding rather than a learned one.

That residual also recovers the input width the paper never tabulates: 25 features, embedded without a
bias — consistent with 11 time-dependent variables plus one-hot baseline variables. This is the
decomposition that closes cleanly, not a proof that no other does; it is recorded as a derivation in §4.
That the embedding is **linear and bias-free** is what makes `IntervalGrid`'s sparse formulation exact
rather than approximate (see §3, "interval alignment").

`teco_parameter_count()` in `src/teco/model.py` is this arithmetic, and its doctest asserts both that it
equals 18 928 130 and that a `Model(vocab_size=25)` built from the shipped config has exactly that many
parameters.

---

## 3. Element ledger

Every element of the paper's Methods section. `evidence` is a test, a doctest, or a command that was run
and whose output is in §6.

### Preprocessing

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| 11 named time-dependent variables (temperature, respiration rate, SpO₂, mSOFA total, mSOFA respiratory sub-score, SF ratio, BMI) | Methods, "Study design and cohort" | **adapted** — the feature axis is the dataset's **code vocabulary** instead | `IntervalGrid`, `src/teco/model.py` | `tests/test_interval_grid.py::test_matches_the_dense_construction` |
| 15-minute interval alignment, mean per variable per interval | Methods | **ported** | `IntervalGrid.interval_index` + the scatter-mean in `IntervalGrid.forward`; `model.interval_minutes=15.0` | same test, parametrised over three geometries including the paper's 15 min / 96 h |
| carry-forward of a missing interval value (LOCF) | Methods | **ported** | `IntervalGrid.forward` (cummax-gather) | `test_carry_forward_reaches_the_prediction_time` |
| 96-hour input range (given for the 24-hour sub-model) | Methods | **ported** | `model.lookback_hours=96.0` → 384 intervals | `test_measurements_before_the_window_are_dropped`; §6 run shows `grid intervals: 384` |
| baseline variables (age, sex, ethnicity, race) held constant across the time range | Methods | **ported** | `IntervalGrid.static_embedding` + `static_inclusion_mode: INCLUDE` in `configs/profile/default.yaml` | `test_statics_are_added_to_every_interval` |
| bias-free linear embedding of the concatenated features to 512 dimensions | derived from the parameter count (§2) | **ported** | `IntervalGrid.embed` = `nn.Linear(vocab_size, 512, bias=False)` | `teco_parameter_count` doctest |
| input normalisation | *not stated in the paper* | **deviation** — the template's preprocessing z-scores numeric values per code | `preprocess_data` (`fit_normalization` → `normalization` MTD stages) | §6: `runs/verify/data/patients/{fit_normalization,normalization}/` exist; `codes.parquet` carries `values/sum`, `values/sum_sqd` |
| training-range restriction: "only the data ranges that led to the eventual outcome at the ICU endpoint" | Methods | **omitted** | — | Interface: the set of scored timepoints *is* `external_labels_dir`, produced upstream by `meds-dev-task`. A model that re-cut the cohort's time ranges would be answering a different task than the one it is scored on, and `predict`'s coverage check would fail rather than hide it. |

### Architecture

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| 6 Transformer-encoder layers | Methods | **ported** | `Model.encoder`, `configs/model/default.yaml` `num_layers: 6` | parameter-count doctest |
| 8 attention heads | Methods | **ported** | `nhead: 8` | same |
| 512-dimension embedding | Methods | **ported** | `d_model: 512` | same |
| GELU activation | Methods | **ported** | `activation="gelu"` | code; not separately measurable |
| dropout 0 | Methods, "hyperparameter set yielding the highest AUC" | **ported** | `dropout: 0.0` | `Model(vocab_size=25).hparams.dropout` doctest |
| feed-forward width 2048; no final LayerNorm | derived from the parameter count (§2) | **ported** | `dim_feedforward: 2048`, `norm=None` | parameter-count doctest |
| relative positional encoding | Methods | **adapted** — fixed sinusoidal encoding of each interval's offset *back from the prediction time* | `sinusoidal_encoding` | `sinusoidal_encoding` doctests (offset 0 at the prediction time; the encoding of the recent end is invariant to how much history precedes it); parameter-free, as §2 requires |
| linear classification layer + softmax, 2 classes | Methods | **ported** | `Model.head` = `Linear(512, 2)`, `predict_step` returns `softmax(...)[:, 1]` | parameter count contributes exactly 1 026; `test_end_to_end` validates the output against `meds_evaluation.PredictionSchema` |
| 18 928 130 trainable parameters | Methods | **reproduced** | — | `teco_parameter_count()` doctest; §6 run reports 18 920 962 at `vocab_size=11`, which is the same formula at that input width |
| sequence → single prediction (pooling) | *not stated* | **assumed** — the last interval, i.e. the prediction time | `Model.encode`, `pooling: last` | `test_model_reads_the_last_interval`; see §4 |

### Objective and optimisation

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| training objective / loss function | *not named in the paper* — only "Softmax activation" | **assumed** cross-entropy over the 2 logits | `Model.compute_loss` | `test_learns_designed_signal` (AUROC 1.0000); see §4 |
| SGD with momentum 0.9 | Methods | **ported** | `configs/optimizer/sgd.yaml` | §6 resolved config of the verification run |
| learning rate 0.01 | Methods | **ported** | `configs/optimizer/sgd.yaml` | same |
| LR halved every 50 epochs | Methods | **ported** | `StepLR(step_size=50, gamma=0.5)` in `configs/model/default.yaml`, stepped **per epoch** by `Model.configure_optimizers` | code + config; the epoch/step correction is an upstream finding, §7.1 |
| weight decay | *not mentioned* | **ported as 0** (SGD's default) | `configs/optimizer/sgd.yaml` carries no `weight_decay` | resolved config. Note this is a deliberate *departure from the template's* generic `adamw.yaml`, which would have applied `weight_decay: 0.01` — a regulariser the published model never had |
| weight-decay parameter grouping | *not mentioned* | **omitted — nothing to port** | — | with weight decay at 0 there is nothing to group |
| warmup | *not mentioned* | **omitted — nothing to port** | — | — |
| gradient clipping 1.0 | Methods | **ported** | `configs/trainer/default.yaml` already sets `gradient_clip_val: 1.0` | resolved config |
| batch size 32 | Methods | **ported** | `supervised_train.yaml` already defaults to 32 | resolved config |
| maximum 500 epochs | Methods | **ported** | `trainer.max_epochs: 500` in `configs/profile/default.yaml` | resolved config |
| early stopping: val loss changing by less than 1e-4 over 100 epochs | Methods | **ported** | `callbacks.early_stopping` in `configs/profile/default.yaml` | resolved config; not triggered in the short verification runs |
| class-imbalance handling | *not addressed in the paper* (36% mortality in development) | **omitted — nothing to port** | — | — |
| seeding | *not stated in the paper* | **added** | `seed: 0` in `supervised_train.yaml` (template default), consumed by `seed_everything(..., workers=True)` | §6, reproducibility |
| determinism / cuDNN benchmarking | *not stated* | **added** | `deterministic: warn`, `benchmark: false` in `configs/profile/default.yaml` | §6, reproducibility — measured, and **not sufficient on CUDA** |
| float32 matmul precision | *not stated*; the V100 has no TF32 path | **left at torch's `highest`** | `precision: 32-true` in `configs/profile/default.yaml` | §6 |
| metrics logged during training | *not stated* | **added** loss + accuracy | `Model.compute_loss` returns `{"acc": ...}` | CSV logger output under `runs/logs/` |

### Evaluation and inference

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| AUC as the evaluation metric | Results | **adapted** — computed downstream by `meds-evaluation` from `predictions.parquet`, not inside this repository | `predict` output | `test_end_to_end` asserts `PredictionSchema` conformance; §6 reports AUROC computed from the written file |
| predictions updated hourly as the encounter evolves | Methods / Discussion | **ported** | `predict` scores every `(subject_id, prediction_time)` in `external_labels_dir`, and fails rather than writing a short file | `test_end_to_end` asserts `n_written == n_expected`; §6 shows 40/40 |
| 9 sub-models, one per horizon (0, 12, … 96 h) | Methods | **adapted** — one trained model per task | the horizon lives in `external_labels_dir`; `model.yaml` registers one `supervised.train` / `supervised.predict` pair, which MEDS-DEV runs once per task | `model.yaml`; design-interface §`external_labels_dir` ("a model receives *where* to predict, never *what*") |
| 20 random 80/20 development splits | Methods | **omitted** | — | Interface: split membership is a property of the MEDS dataset (`train` / `tuning` / `held_out` shards) and is fixed by whoever produced it. Re-splitting inside the model would make its `held_out` predictions incomparable to every other MEDS-DEV model's. |
| SHAP expected-gradients feature attribution (20 background/batch pairs) | Methods, "Feature importance" | **omitted** | — | Interface: the command DAG has five commands and none of them is an attribution step; attribution is not on the path from data to `predictions.parquet`. Recorded rather than silent, and it would be additive work rather than a change to anything above. |
| private THR COVID-19 development cohort; MIMIC-IV ARDS/sepsis validation cohorts | Methods | **omitted** | — | The development data is not public, and dataset selection is MEDS-DEV's job, not the model's. **Nothing in §6 was measured on either cohort.** |
| PyTorch 1.8.1 on an NVIDIA V100 | Methods | **adapted** | torch 2.13.0 + CUDA 13.0 on an NVIDIA GB10 | §6 |
| console entry points | — | **n/a** | — | No code was published, so there is no `pyproject.toml` and no entry point to account for. This row exists because the porting checklist asks for it. |

---

## 4. Assumptions the paper leaves open

These are choices, not ports. Each is exposed as configuration so it can be changed without editing code,
and each would change results.

1. **The loss is cross-entropy.** The paper names the output activation (softmax) but never the objective.
   Cross-entropy is the near-universal partner of a softmax head, and the class-count-of-2 head is what the
   parameter budget shows, but this is inference.
2. **The head reads the last interval** (`pooling: last`). The paper says the encoder "outputs the
   probability of death using a linear classification layer" without saying which position, and the
   parameter count rules out any pooling that carries weights. `last` is the prediction time itself, which
   is the question a continuously re-scored encounter-level model is answering; `pooling: mean` is the
   other parameter-free reading and is one CLI override away.
3. **The positional encoding is sinusoidal, anchored at the prediction time.** §2 establishes that it is
   parameter-free; the specific functional form is not stated. Anchoring at the prediction time is what
   makes it "relative" in a way that matters for this model: re-scoring the same history an hour later
   does not shift every position underneath it.
4. **The embedding is a single bias-free linear map.** Read off the parameter count (12 800 = 512 × 25),
   not from the text, which says only "embedded these features into a 512-dimension vector".
5. **A code with no numeric value contributes 1.0** in its column. TECO's 11 variables are all numeric so
   the paper never faces the question; MEDS carries value-less codes (diagnoses, admissions), and 1.0 makes
   such a code's contribution exactly its embedding vector — the usual occurrence-indicator convention.
   A consequence worth stating: after z-scoring, a *carried-default* zero and an *observed-mean* z-score of
   0 are the same number in the grid. TECO's own default-filled grid has the same ambiguity, which is what
   its relative positional encoding is motivated by, but the collision is more common here.

---

## 5. Deviations that affect results

Read this before comparing anything here to TECO's published AUCs. (Nothing in §6 is comparable to them
anyway — see the last row of §3 — but these are the reasons it could not be even with the right data.)

**1. The feature axis is the code vocabulary, not TECO's 11 clinical variables — the significant one.**
TECO's inputs are a hand-built physiological panel: body temperature, respiration rate, SpO₂, the mSOFA
total and its respiratory sub-score, the SpO₂/FiO₂ ratio, and BMI. mSOFA and the SF ratio are *derived*
quantities computed by cohort-specific logic from other measurements; the model card's own framing applies
here — resolving them needs dataset-specific predicates that a model handed an arbitrary MEDS dataset does
not have, exactly as the template refuses to resolve ACES task definitions for the same reason.

So `IntervalGrid` runs TECO's construction over whatever codes the dataset has. On a MEDS dataset with a
comparable panel this is a superset of TECO's input; on one without SpO₂/FiO₂ it cannot reconstruct the SF
ratio at all, and the published feature-importance finding (mSOFA dominant, then BMI, SF ratio and
temperature) has no counterpart here. **Restoring it is a MEDS-transforms `pipeline:` away** —
`preprocess_data pipeline=<yaml>` accepts one, and a dataset-specific pipeline that derives mSOFA and the
SF ratio as codes would put TECO's exact panel into the vocabulary without touching the model. That is the
work this port leaves undone, and it is dataset-specific by nature rather than skipped for convenience.

**2. Numeric values reach the grid z-scored.** The template's `preprocess_data` runs meds-torch-data's
`fit_normalization` / `normalization` stages, so a value in the grid is `(x − mean) / std` per code. The
paper states no normalisation of its own. This changes the scale the linear embedding sees and interacts
with assumption 4.5 above (a carried default of 0 now sits at the code's mean rather than below its range).

**3. The grid's right edge is the last measurement, not `prediction_time`.** A `MEDSTorchBatch` carries
inter-measurement gaps but no absolute timestamp, so the offset between a subject's final observed event
and the label's prediction time is not recoverable inside the model. Under `TO_END` sampling the last
measurement is the most recent event at or before the prediction time, so the two differ by less than the
gap to the next observation — under one interval for a densely monitored ICU encounter, more for a sparse
one. TECO's grid, built outside the model from timestamps, has no such slack.

**4. `max_seq_len` bounds how much of the 96-hour window is actually visible.** The grid spans 96 hours,
but the batch it is built from holds at most `max_seq_len` measurements (default 256, ending at the
prediction time). A densely monitored encounter — 11 variables at 15-minute resolution is ~4 200
measurements over 96 hours — is truncated to its most recent 256, and the earlier intervals then hold only
carried-forward values from whatever survived. TECO's dense grid has no equivalent limit. Raise
`max_seq_len` to cover the window at the dataset's measurement density: the encoder's cost is set by the
*interval* count, which does not change, so what grows is data loading and grid construction rather than
the model's compute.

Which of the two limits actually binds is dataset-dependent, and on the MIMIC-IV demo cohort it is the
*lookback*, not `max_seq_len`: 44% of the measurements the loader brings in are older than 96 hours and are
dropped, and the loaded windows span a median of 111 hours. Measured in §6.5.

**5. One model per task, not 9 horizon sub-models.** TECO trains a separate model for each of 9 prediction
horizons. Here the horizon is a property of `external_labels_dir`, so reproducing the fan is running
`supervised_train` nine times against nine label sets — the same total training, arranged by the caller.
No single artifact produced here is "the TECO model"; each is one of its sub-models.

**6. The training-range restriction is not applied** (§3). TECO trains only on "the data ranges that led to
the eventual outcome at the ICU endpoint"; here the training index is exactly the label file it is handed.

**7. Same-seed runs are not bit-reproducible on CUDA.** Measured, quantified and explained in §6. This is a
deviation from what the shipped config *appears* to promise, and it is why every number in §6 is reported
with the device it was produced on.

---

## 6. Verification

Everything below was executed on this machine, not inferred. Environment: Python 3.12.3 (aarch64),
torch 2.13.0 + CUDA 13.0, lightning 2.5.6, meds 0.4.1, meds-torch-data 0.9.0, MEDS-transforms 0.6.7,
meds-evaluation 0.0.6, polars 1.43.2, NVIDIA GB10.

### 6.1 Test tiers, before and after the port

| tier | before (generated stub) | after |
|---|---|---|
| `pytest -m "not slow" -rs` | **55 passed, 1 skipped, 3 deselected** — the skip is `skip_if_stub` on `test_end_to_end` | **71 passed, 0 skipped, 3 deselected** |
| `pytest -m slow -rs` | **0 passed, 3 skipped** — all three `skip_if_stub`, including `test_meds_dev_e2e`, which never reached its own opt-in gate | **2 passed, 1 skipped** (the `meds_dev` opt-in) |
| `ruff check .` / `ruff format --check .` | not measured (copier's post-copy task runs both) | `All checks passed!` / `38 files already formatted` |

Zero `skip_if_stub` skips remain. The 16 additional passing tests are `tests/test_interval_grid.py` (11,
new here), the `src/teco/model.py` doctests (4, new here), and `test_end_to_end`, which was the tier that
had been skipping — this is the first time it has run in this repository.

The one remaining skip is `test_meds_dev_e2e`, which is opt-in because it clones MEDS-DEV and builds two
virtualenvs. **It was not run, and this repository therefore has no MEDS-DEV evidence at all.** The reason
it was not attempted is a defect already recorded against the local MEDS-DEV checkout
(`~/Git/meds-dev-venv-bug.md`): `venv_dir` defaults inside `output_dir`, which `run_in_env` then deletes,
so `do_overwrite=true` runs fail with `command not found` before reaching the model. That is a prediction
about the outcome, not an observation — nothing here was measured through MEDS-DEV. §9.5.

### 6.2 Learnability (`pytest -m slow`), run three times

Synthetic designed-signal cohort (200 train / 40 tuning / 40 held-out subjects), conformance-size model on
CPU, 25 epochs, batch 16.

| run | designed signal, held-out AUROC | negative control (shuffled labels), held-out AUROC |
|---|---|---|
| 1 | **1.0000** | **0.3885** |
| 2 | **1.0000** | **0.3885** |
| 3 | **1.0000** | **0.3885** |

The negative control is the load-bearing half: at 0.3885 on shuffled labels the model has learned nothing
transferable, so the 1.0000 is a statement about the designed signal rather than about the test being
vacuous. Both numbers are identical across all three runs — consistent with §6.4's CPU result.

Note that `assert_learns_signal`'s threshold is 0.9 and the control's ceiling is 0.70, so neither number is
near its bar.

### 6.3 The shipped (paper-geometry) model, end to end

The tests train a deliberately small model because `run_chain` pins the accelerator to CPU (see
`CONFORMANCE_MODEL` in `tests/conftest.py`). This run uses **the shipped defaults** — 6 layers, `d_model`
512, 384 intervals — on the same synthetic cohort, 30 epochs, batch 32, seed 0, on the GPU.

```
supervised_train (a): 45.50 s   monitored val/loss = 0.0005075751105323434, epochs=30, steps=210
supervised_train (b): 45.84 s   monitored val/loss = 0.0005082486313767731, epochs=30, steps=210
checkpoint selection: monitored_best on val/loss  (the `final_weights` fallback did not fire)
reload:  hparams {'vocab_size': 11, 'd_model': 512, 'nhead': 8, 'num_layers': 6,
                  'dim_feedforward': 2048, 'dropout': 0.0, 'interval_minutes': 15.0,
                  'lookback_hours': 96.0, 'pooling': 'last'}
         trainable parameters: 18 920 962      grid intervals: 384
predict: predictions (40, 4), held-out AUROC = 1.0000, coverage {'held_out': {'n_expected': 40, 'n_written': 40}}
```

`18 920 962` is `teco_parameter_count(n_features=11)` — the same decomposition as §2 at this dataset's
vocabulary size (10 codes plus the pad index). The full 18 928 130 appears at `n_features=25`, which is the
doctest.

### 6.4 Reproducibility: same seed, same config, same device

The porting procedure requires a metric produced twice before it is recorded. It was, and **the two paper-
geometry runs above disagree** — 0.00050758 vs 0.00050825, differing in the fourth significant figure.
That is a finding, so it was chased down rather than averaged away. All runs below: seed 0, 10 epochs,
same config, same machine, monitored `val/loss` from the artifact manifest.

| device | settings | run A | run B | identical? |
|---|---|---|---|---|
| CPU (conformance geometry) | shipped defaults | 0.004379396326839924 | 0.004379396326839924 | **yes, bit-identical** |
| CUDA (paper geometry) | `deterministic` absent (Lightning default `False`) | 0.005261504091322422 | 0.005242484156042337 | no |
| CUDA (paper geometry) | `deterministic: warn` (the shipped default) | 0.0052414946258068085 | 0.005282382480800152 | no |
| CUDA (paper geometry) | `trainer.deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 0.005236954428255558 | 0.005236954428255558 | **yes, bit-identical** |
| CUDA (paper geometry) | `trainer.deterministic=true`, **no** `CUBLAS_WORKSPACE_CONFIG` | 0.005236954428255558 | 0.005236954428255558 | **yes** — and equal to the row above |

**Conclusion.** Three things were established, and only the first is a property of the shipped defaults.

1. **On CPU, the shipped configuration reproduces exactly** — bit-identical to the last digit. That is the
   claim the porting procedure asks for, and it holds: same seed + same config + same environment → same
   metric.
2. **On CUDA it does not**, with either `deterministic` unset or at the shipped `warn`. `warn` asks for
   deterministic kernels and, by design, warns rather than failing where none exists — so it improves
   nothing measurable here. The disagreement is small — 0.13% between the 30-epoch pair in §6.3 and
   0.36–0.78% between the 10-epoch pairs above — but it is not zero, and a GPU number from this repository
   should not be quoted to more than two significant figures unless point 3 is applied.
3. **`trainer.deterministic=true` closes it**, giving four bit-identical runs across two settings — and on
   this stack (torch 2.13.0 / CUDA 13.0) it did **not** need `CUBLAS_WORKSPACE_CONFIG`: the runs without it
   neither raised nor differed. No op in this model lacks a deterministic kernel here, so the flag cost
   nothing but is not shipped on, because `true` *raises* where a kernel is missing and that is a
   machine-dependent property. Anyone quoting a GPU number from this repository should set it; the config
   comment in `configs/profile/default.yaml` says so.

The paper-geometry numbers in §6.3 predate this investigation and were produced at the shipped `warn`
setting, which is why they are reported as a disagreeing pair rather than as one value.

### 6.5 Scaling: measured on a tensorized MIMIC-IV demo cohort

Deviation 5.1 replaces TECO's 25-feature input with the dataset's code vocabulary, and deviation 5.4 notes
that `max_seq_len` bounds the window. Both were measured against a real tensorized MIMIC-IV **demo**
workspace (`~/Git/trash/medsdev-run/train/data/patients`, 1 680 train subjects), at the shipped geometry —
384 intervals, `d_model` 512, `max_seq_len` 256, batch 32.

```
vocab_size                     7 090          (TECO's own input width: 25)
model parameters              22 545 410      (TECO's own model:  18 928 130)
distinct codes per batch (U)   1 025 – 1 402  (mean 1 192 of 7 090)
grid tensor                    48 – 66 MiB    (dense over the full vocabulary: 332 MiB — 6x more)
grid output                    (32, 384, 512)
```

Three things this settles:

- **The vocabulary costs parameters, not memory.** The grid is `[batch, intervals, U]` where `U` is the
  distinct codes *present in the batch*, so it is bounded by `batch_size × max_seq_len` (8 192 here)
  **regardless of vocabulary size**. Only the embedding matrix grows with the vocabulary, at
  `vocab_size × d_model` — 3.6M parameters here, and ~20M at a 40 000-code vocabulary. Full MIMIC-IV is
  larger than this demo (extrapolated tens of thousands of codes; **not measured**), and the conclusion is
  unchanged: a bigger model, not a bigger activation.
- **The 96-hour lookback, not `max_seq_len`, is what mostly limits history on this dataset.** Of 81 920
  loaded measurements, **56% fall inside the 96-hour window** and the rest are older and dropped. The
  loaded windows span a median of 111 hours, with a long tail (p90 7 945 h, max 36 461 h) — MIMIC
  timelines carry history well beyond the ICU stay TECO was built for. Whether `max_seq_len` *additionally*
  truncates a densely monitored ICU stay needs a per-subject check that was not done.
- **`interval_minutes` and `lookback_hours`, not the vocabulary, are the memory knobs.** They set the
  interval count directly, and the grid is linear in it.

### 6.6 Contract guards

The porting procedure asks which of the vendored contract's guards fired against a real model for the first
time. Of the three it names:

- **`predict`'s coverage check** — ran and passed on every prediction run: 40/40 on the synthetic cohort,
  and `n_written == n_expected` asserted by `test_end_to_end` on the `meds_testing_helpers` fixture. It did
  not fire as an error.
- **`_persist_checkpoint`'s `final_weights` fallback** — did not fire; every run had a validation split and
  a `ModelCheckpoint`, and every manifest records `selection: monitored_best`.
- **`load_pretrained_weights`'s zero-match `RuntimeError`** — not reachable in the `supervised` profile,
  which has no warm-start source. Not exercised.

---

## 7. Findings for the template

Three things this port ran into that are template-side rather than model-side, recorded per
`docs/PORTING-A-MODEL.md` step 6 and filed upstream as
[#15](https://github.com/florian6973/MEDS_model_template/issues/15),
[#16](https://github.com/florian6973/MEDS_model_template/issues/16) and
[#17](https://github.com/florian6973/MEDS_model_template/issues/17).

**7.1 `BaseLightningModule.configure_optimizers` hardcodes `interval: "step"`.** TECO halves its learning
rate "every 50 epochs"; the contract's scheduler wiring would have made that every 50 *optimizer steps* —
here, a 7× faster decay at 7 steps/epoch, and arbitrarily faster on a real dataset. The scheduler config
has no way to say which, because the interval is not read from config at all. `Model.configure_optimizers`
overrides it, but a model author who does not read the base class will silently get a schedule orders of
magnitude off the one they configured. Suggested fix: read `interval` (and `frequency`, `monitor`) from the
scheduler config node.

**7.2 `static_inclusion_mode: INCLUDE` cannot be exercised by the template's own property fixture.**
`meds_model_base.testing.synthetic.build_signal_dataset` writes a cohort with **no static measurements**,
and meds-torch-data's collate then fails with `ValueError: Cannot infer dtype from empty values; provide an
explicit schema=` when the static tensors are empty for a whole batch. Any model that reads static data —
TECO reads it for the baseline variables, which are a third of its input — hits this on the flagship
learnability test. Worked around here by `tests/test_property.py::add_baseline_variables`, which injects
one value-less and one numeric static per subject, drawn independently of the labels. Suggested fixes:
give `build_signal_dataset` static measurements (or a flag for them), and/or handle the empty-static case in
meds-torch-data.

**7.3 `run_chain` has no extension point for extra overrides.** It builds its own argument list, so a model
whose published geometry is too large for a CPU-only conformance run has no supported way to size it down
for the fixture. `tests/conftest.py` wraps `harness.run_cli` to append overrides, which works but reaches
into the contract's internals. Suggested fix: an `extra: list[str] | Mapping[CommandName, list[str]]`
parameter on `run_chain`.

Separately, the version generated from (`29f58b2`) has no `deterministic` / `benchmark` / `precision` keys
in `configs/trainer/default.yaml`, so `trainer.deterministic=true` is a Hydra struct error rather than an
override. They are added here in `configs/profile/default.yaml`; the template's own
`reproducible-by-default` work appears to add them upstream, in which case this repository's copies become
redundant on the next `copier update`.

---

## 8. Step 7 checklist

- [x] **Every entry point in the source's `pyproject.toml` appears in the ledger** — there is no source
      distribution; recorded as such in §1 and as a ledger row.
- [x] **The batch-field test was performed and its conclusion recorded** — §3 and the `model.py` module
      docstring. TECO's `forward` reads a dense, fully-imputed `[batch, intervals, variables]` matrix and
      no timestamps or missingness mask, which is coherent only because its preprocessing moved that
      information into the grid. Porting the encoder without the grid would have discarded all of it, so
      `IntervalGrid` is a port of the preprocessing, not a convenience wrapper.
- [x] **Every optimiser detail has a ledger row** — grouping, LR, schedule, warmup, clipping, weight decay,
      batch size, epoch budget, early stopping.
- [x] **Seeding, determinism and matmul precision have ledger rows, and the settings used are in
      Verification** — §3 and §6.4, including the measured failure to reproduce on CUDA.
- [x] **Training ran twice at one seed and the reported metric agreed** — on CPU, bit-identical. On CUDA it
      did **not** agree; §6.4 reports the disagreement rather than the first number.
- [x] **`pytest -m "not slow" -rs` shows zero `skip_if_stub` skips, and the tier that was skipping passed** —
      §6.1, with before/after counts.
- [x] **`test_smoke_pipeline` and `test_property` both ran** — §6.1, §6.2. The negative control is at 0.3885
      against a 0.70 ceiling, so it is not passing trivially.
- [x] **No ledger row is justified by the test fixture** — the one place where the fixture and the model
      collided (`static_inclusion_mode: INCLUDE`) was resolved by fixing the fixture, §7.2.
- [x] **"Deviations that affect results" is present** — §5, seven entries.
- [x] **Every row has evidence or is marked unverified** — the unverified ones are named: `test_meds_dev_e2e`
      (§6.1), the GELU activation and several config-only rows whose evidence is the resolved config rather
      than a behavioural test, and the `load_pretrained_weights` guard (§6.6).

## 9. Work required to complete the port

In rough order of how much it would change results:

1. **A MEDS-transforms `pipeline:` that derives TECO's clinical panel** (mSOFA and its respiratory
   sub-score, the SF ratio, BMI) as codes, for a named dataset. This is deviation 5.1, the only one that
   changes what the model can see.
2. **Raise `max_seq_len` to cover 96 hours** at the target dataset's measurement density, and measure what
   it costs (deviation 5.4).
3. **Run the horizon fan** — nine label sets, nine models — and report the AUC-vs-horizon curve the paper's
   figures are built from (deviation 5.5). `./slurm/submit.sh --tasks-file` submits exactly this shape as a
   job array; the scripts are dry-run-validated against this model but **have not been run on a real
   scheduler** (none was in reach), so the `afterok` / `aftercorr` wiring is unverified against sbatch.
4. **Settle `pooling` and the loss empirically** (§4.1, §4.2) on a real cohort, since the paper does not.
5. **Run `pytest -m meds_dev`** once the MEDS-DEV venv defect is resolved (§6.1).
