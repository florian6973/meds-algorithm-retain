# Implementation report — PORTER → `porter`

Written per [`docs/PORTING-A-MODEL.md`](docs/PORTING-A-MODEL.md). It records what was ported from the
source, what was adapted, and what was left out. It is the ledger, not a summary of the work.

**Read section 4 before comparing any number produced here to the paper's published results.** The
backbone is scaled down, the pretraining targets are not the paper's composite tokens, and the numeric
pathway's reference-range branch is unreachable on MEDS data. None of those is cosmetic.

## 1. Source

| | |
|---|---|
| model | **PORTER** ("Portable EHR Representations") |
| paper | [arXiv:2606.24102v1](https://arxiv.org/abs/2606.24102) — *PORTER: Language-Grounded Event Representations for Portable Structured EHR Foundation Models*, Guo, Yan, Vettese & Sung (The Hospital for Sick Children), 23 Jun 2026, CC BY 4.0 |
| package | **none** |
| repository | **none.** The paper's own §6 (Code Availability) announces `https://github.com/sungresearch/ehr-fm` as forthcoming; it was not resolvable when this port was done, and the model card records the same finding ("Not found — no public model-specific repository", checked 2026-07-30) |
| depended on as | **nothing — reimplemented from the paper.** No upstream code was read, wrapped or vendored |
| target profile | `probe` (`preprocess_data → pretrain → infer → supervised_train → predict`), which is PORTER's own downstream protocol: freeze the backbone, extract representations, fit a logistic-regression probe (§2.7.1) |

**Consequence for this report.** [`docs/PORTING-A-MODEL.md`](docs/PORTING-A-MODEL.md) Step 1 asks for every
console script in the source's `pyproject.toml`. There is no `pyproject.toml`: the ledger below is keyed to
*paper sections* instead, and every row's "source reference" is a section, table or figure. Where the paper
does not state something, the row says so rather than presenting a choice as a port. Nineteen such gaps are
marked **unstated**; they are the honest boundary of this reimplementation.

## 2. The batch-field test (Step 2)

The check that catches a silently-dropped preprocessing stage: list the fields the source model's forward
pass reads, and ask whether reading *fewer* fields than the batch carries is evidence that preprocessing
moved information into the ones it does read.

PORTER's forward pass reads, per event (§2.4, §2.4.2, §2.4.3):

| source input | MEDS batch field | reached here? |
|---|---|---|
| the event's natural-language description | `batch.code` used as a **key into `metadata/codes.parquet`**, never as a learned embedding index | yes |
| numeric value | `batch.numeric_value` | yes |
| numeric value presence | `batch.numeric_value_mask` (`& padding_mask`) | yes |
| reference range (low, high) | **no MEDS field carries one** | no — see ledger |
| patient age in days | `cumsum(batch.time_delta_days)` | yes |

**Conclusion: PORTER reads more of the batch than a fixed-vocabulary model does, not less.** It consumes
every dynamic field `MEDSTorchBatch` carries, so there is no field whose information a missing
preprocessing stage would have to have relocated — the failure mode this step exists to catch does not
apply on the *input* side. Three real gaps remain, and each has a ledger row rather than a silence:

1. **The reference range.** MEDS standardises no column for it, so §2.4.2's primary numeric scalar is
   unreachable and every value takes the log-magnitude fallback. The branch is implemented and tested; it
   activates for any dataset whose code metadata carries one of `porter.model.REFERENCE_RANGE_COLUMNS`.
2. **Demographics.** The paper renders sex, birth and death as ordinary timeline events (Table S3). Both
   datamodule configs set `static_inclusion_mode: OMIT`, so static rows never enter the batch.
3. **The pretraining *targets*.** This was the one place the Step 2 failure mode really applied — the
   paper's composite token carries the numeric magnitude into the objective and the bare code does not.
   It is now implemented (§3.4), with the binning confined to the label so the input pathway stays
   language-grounded and continuous. The remaining caveat is that the paper's composite token also
   discretises *non-numeric* attributes, which MEDS has already folded into `code`.

One thing the test surfaced that the paper could not have: **meds-torch-data z-scores `numeric_value` per
code during tensorization**, so the value reaching `forward` is a standard deviation, not a clinical
magnitude, and `log(1 + |v|) / 7` applied to it would be meaningless. The model inverts that transform
(`build_value_statistics`) using the same per-code statistics MEDS-transforms used. See the ledger row.

## 3. Element ledger

Status is **ported** (implemented as the paper describes), **adapted** (implemented differently, with the
reason), **omitted** (not implemented, with the reason), **deviated** (a stated value replaced by a
different one), **unstated** (the paper does not specify it, so a choice was made and is recorded), or
**new** (has no counterpart in the paper — reserved for things this port added, never for a paper element that simply was not run: that is **omitted**).

### 3.1 Input pathway — the frozen text encoder (§2.4.1, §2.6.1, Table S3, Table S6)

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| Frozen text encoder, BioLORD-2023, 768-d | §2.4.1; Table 3 | **ported** | `model.py::_encode_descriptions`; `configs/model/default.yaml:text_encoder,d_text` | `test_end_to_end` and both `test_property` tiers run with the real BioLORD checkpoint; §6 records the runs |
| Encoder never fine-tuned | Abstract, §§2.4, 2.4.1, 2.7.3 | **ported** | `text_table` is a `register_buffer`, not an `nn.Parameter`; it is absent from `parameters()` and therefore from the optimizer | doctest asserts `requires_grad` is `False`; `test_frozen_text_table.py::test_table_is_frozen_and_not_a_parameter` asserts it is a buffer and not a parameter |
| Descriptions encoded once and cached | §2.4.1; Fig 1 caption; Table S6 (15.12 M = 19,686 × 768) | **ported** | `build_text_table`, called once from `Model.__init__` at `pretrain` only | `tests/test_frozen_text_table.py::test_checkpoint_carries_the_table_and_reload_needs_no_encoder` reloads the checkpoint with `sentence_transformers` blocked from import and asserts the table survives |
| Two-layer GELU projection MLP into the backbone width | §2.4.1; width derived from Table S6 (1.18 M = 2 × (768² + 768)) | **ported** | `Model.text_proj` = `Linear(d_text, H) → GELU → Linear(H, H)` | doctest on `event_representations` compares against `text_proj(text_table[code])` |
| PORTER-Random control arm (unit-norm normal vectors keyed by the description string) | §2.6.1 | **ported** | `model.py::_random_embeddings`; `model.text_encoder=random` | doctest asserts unit norm and cross-call determinism |
| BGE-M3 (1024-d) / Qwen3-Embedding-8B (4096-d) arms | §2.6.1; Table 3 | **omitted** | reachable as `model.text_encoder=<hf-id> model.d_text=<dim>` | `test_frozen_text_table.py::test_encoder_width_mismatch_is_a_named_error` — a declared width that disagrees with the encoder raises rather than reshaping. Neither arm is benchmarked here |
| Text-encoder pooling (mean / CLS / last-token) | **unstated** for PORTER (§2.4.1 says only "encoded"); last-token pooling is stated only for the unrelated Qwen3 serialization comparator (§2.5) | **unstated** | whatever the `sentence-transformers` checkpoint declares — for BioLORD-2023 that is mean pooling | — |
| L2-normalising the cached embeddings | **unstated** (only the Random arm is stated to be normalised) | **unstated** | not applied to encoder output; applied to the Random arm, as stated | — |
| Nine templated description categories (`"Measurement: {name}"`, `"Drug: {name}. Route: {route}"`, …) | §2.4.1; Table S3 | **adapted** | `model.py::render_event_description` + `CATEGORY_ALIASES`. The paper fills slots from OMOP columns (`measurement_concept_name`, `drug_route_concept_name`); MEDS has already flattened those tables into `code` + `description`, so the first `//`-segment picks the category and `description` fills the name slot | 7 doctest cases on `render_event_description`, including the paper's own `Measurement:` example and both fixed demographic strings |
| `"Patient Birth Date"` / `"Patient Death Date"` as fixed strings | Table S3 footnote | **ported** | `FIXED_DESCRIPTIONS`, keyed on MEDS' `MEDS_BIRTH` / `MEDS_DEATH` | doctest |
| Null template segments are dropped | Table S3 footnote * | **adapted** | a null description, or one that merely repeats the code, falls back to the code's own segments | doctest (`"SIGNAL//POS"` → `"Signal: POS"`) |
| Drug route / non-numeric measurement result as extra slots | §2.4.1; Table S3 | **omitted** | MEDS carries no separate route or result-concept column; whatever the ETL folded into `code` or `description` is what the encoder sees | — |
| Units absent from the primary templates | Table S3; §2.6.2 | **ported** | no unit slot is rendered | inspection of `render_event_description` |
| PORTER-NumText arm (values, units, ranges written into the description text) | §2.6.2; Table 4 | **omitted** | needs value **and unit and reference range** per event to render; MEDS supplies none of the three at description-build time (descriptions are per-*code*, not per-event) | — |
| PORTER-NoNum arm (no numeric pathway at all) | §2.6.2; Table 4 | **omitted** | the ablation is not run here and has no named config; it is reachable by holding the FiLM heads at zero, which is what the zero-init doctest already exercises at step 0 | — |
| Demographics as ordinary timeline events | Table S3; Fig 1B | **ported** | MEDS splits these in two and both halves now reach the model. *Timestamped* ones always did — `MEDS_BIRTH` is dynamic and renders as `'Patient Birth Date'`, the paper's exact string. *Static* ones (sex, race) do now that both `configs/datamodule/*.yaml` default to `static_inclusion_mode: PREPEND`, which is precisely Table S3's treatment: meds-torch-data puts them at the front of the sequence as one event with time delta 0. No model change was needed — they arrive as ordinary events | `tests/test_static_demographics.py` drives the real model over a PREPEND batch on a fixture that *has* statics, and asserts PREPEND and OMIT actually differ; `tests/test_default_is_the_paper.py` asserts PREPEND is the shipped default in all four commands |

### 3.2 Input pathway — the numeric pathway and FiLM (§2.4.2, Fig 1)

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| Reference-range scalar: position in range, 0 = low, 1 = high, clipped `[-2, 3]` | §2.4.2 | **adapted** | `Model.numeric_features`; ranges read by `build_reference_ranges`. Implemented and reachable, but **never selected on any dataset used here** — MEDS standardises no reference-range column | doctest installs a range and checks the midpoint maps to 0.5 and the lower clip to −2.0 |
| Log-magnitude fallback `log(1 + \|v\|) / 7`, clipped `[0, 1.5]` | §2.4.2 | **ported** | `Model.numeric_features`; `LOG_FALLBACK_DIVISOR`, `LOG_FALLBACK_CLIP` | doctest pins `log(1 + 1000)/7 ≈ 0.987` — the paper's own worked value |
| Four-dimensional feature vector: scalar + 3 indicators; exactly one scaling indicator active | §2.4.2; Fig 1 caption | **ported** | `Model.numeric_features` returns `[B, L, 4]` | doctest checks all four components for a valued event, an unvalued event and a pad position |
| Zeros and gate-off for events with no numeric content | §2.4.2 | **ported** | `gate = numeric_value_mask & padding_mask(batch)` | doctest: the unvalued event is exactly `text_proj(text_table[code])` even with non-zero FiLM heads |
| Numeric MLP: two 128-wide FC layers with GELU | §2.4.2 | **ported** | `Model.numeric_mlp`; `configs/model/default.yaml:numeric_hidden` | inspection |
| Two linear heads producing `gamma`, `beta` at backbone width, **zero-initialised** | §2.4.2 | **ported** | `Model.gamma_head`, `Model.beta_head`; weights *and* biases zeroed | doctest: the fusion is exactly the text representation at initialization |
| FiLM fusion `h = (1 + gamma) * h_text + beta` | §2.4.2 | **ported** | `Model.event_representations` | doctest |
| FiLM parameter count ≈ 0.46 M | Table S6 (1.64 − 1.18 M) | **deviated** | the prose's 128-wide MLP gives ≈0.22 M at hidden 768; 0.46 M implies 256-wide layers. The **prose** was followed and the discrepancy is recorded rather than silently resolved | arithmetic in this row |
| "without relying on concept-specific distributional summaries, such as means, standard deviations, or quantiles" | §2.4.2 | **adapted** — *and this is the subtle one* | meds-torch-data's tensorization z-scores `numeric_value` per code before the model ever sees it. `build_value_statistics` reads the same `values/sum`, `values/sum_sqd`, `values/n_occurrences` back out of `codes.parquet` and **inverts** the transform, so the scalar sees a raw magnitude as §2.4.2 defines it. Per-code statistics are therefore used to *undo* upstream preprocessing, never to *define* the representation — but the port does depend on them being present. `model.denormalize_numeric_values=false` opts out | doctest on `build_value_statistics` checks the round trip is exact |
| `numeric_value_mask` is undefined at padding | meds-torch-data, not the paper | **new** (a defect this port had to guard) | presence is `numeric_value_mask & padding_mask(batch)`. meds-torch-data builds the mask as `~isnan(numeric_value)` and pads with `0.0`, so the raw mask is **`True` at every pad position** | doctest asserts the pad position yields all zeros |

### 3.3 Temporal backbone (§2.4.3, Table S4, Table S6)

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| Decoder-only Transformer, 28 layers × 768 hidden × 12 heads = 115.6 M | §2.4.3; Table S6 | **ported** | `PAPER_BACKBONE`, and `DEFAULT_BACKBONE is PAPER_BACKBONE` — `configs/model/default.yaml` **is** the published geometry. `model=small` (0.46 M) is a test fixture and is what the suite runs, through `PORTER_*` env vars set in `tests/conftest.py` | `backbone_parameter_count(PAPER_BACKBONE)` doctest → `115_649_280`; measured forward+backward at the published scale in §6.6 |
| Intermediate size 768 (equal to hidden) | Table S4 | **ported** | preserved at every scale, in both configs | doctest arithmetic |
| SwiGLU activation, bias-free | Table S4 | **ported** | `model.py::SwiGLU` — three bias-free projections | this is what makes 28 × (4·768² + 3·768²) = 115,605,504 reproduce the paper's figure |
| Head dimension 64 = 768 / 12 | derived (§2.4.3) | **ported** | `hidden // heads` | `PorterBlock` raises on a non-divisible pair |
| Alternating global / local attention, **starting global**, window 128 | §2.4.3; Table S4 footnote | **ported** | `Model.backbone_states` passes `window=None` on even depths and `window=128` on odd ones; `causal_attention` dispatches | `tests/test_attention.py` checks every path against a naive masked softmax, and checks the two windowed paths agree |
| RoPE applied to **patient age in days** | §2.4.3; Fig 1 caption | **ported** | `RotaryTimeEmbedding` over `cumsum(nan_to_num(batch.time_delta_days))`. RoPE is relative, so the arbitrary offset from a window that starts mid-timeline cancels in every query·key product | inspection; see also deviation 6 |
| RoPE base / theta; any scaling of age before rotation | **unstated** | **unstated** | `rope_theta: 10000.0`, `rope_time_scale: 1.0`, both exposed as config keys rather than hidden as constants | — |
| Normalisation type and placement | **never named** anywhere in the paper | **unstated** | pre-norm `nn.RMSNorm`. Chosen because it is what the 115.6 M arithmetic supports and what SwiGLU + RoPE + alternating attention implies | `backbone_parameter_count` docstring records the 21,504-parameter (0.019%) gap between this and the paper's stated 115.63 M, which corresponds to one norm gain per layer rather than two |
| Attention bias / QK-norm / logit softcapping / GQA | **unstated** | **unstated** | none of them; consistent with the parameter arithmetic | — |
| Dropout | **absent from Table S4 entirely** | **ported** | 0.0 everywhere — no dropout module exists in this implementation | inspection |
| One input position per clinical event | §2.4.3 | **ported** | meds-torch-data's default `SM` batch mode is already one row per measurement | — |
| Context window 16,384 events | §2.4.3; Table S4 | **ported** | `max_seq_len: 16384` in all four command configs. meds-torch-data pads to the longest sequence *in the batch*, not to this bound, so it costs nothing on short timelines | §6.6 runs a forward+backward at exactly 16,384 events |
| Event-budget sequence packing (16,384 events/batch, ≥1 patient) and patient-boundary masking | §2.4.3; Table S4 | **omitted** | meds-torch-data pads each subject to `max_seq_len` and never packs two subjects into one row, so there is no cross-patient attention for boundary masking to prevent. The consequence is a different (padding-based) compute profile, not a different model | — |
| Truncation direction for over-long timelines | **unstated** | **unstated** | `TO_END` at `infer`/`predict` (the most recent window, which is what "at prediction time" requires) and `RANDOM` at `pretrain`, both from the template's datamodule configs | — |
| Ordering / tie-breaking of simultaneous events | **unstated** | **unstated** | MEDS shard order, unchanged | — |
| PyTorch 2.7 | §2.4.3 | **deviated** | torch 2.13.0+cu130 (this repository's lockfile) | §6 |

### 3.4 Pretraining objective and heads (§2.4.4, Fig S2)

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| Autoregressive next-event prediction as classification | §2.4.4 | **ported** | `Model.compute_loss` | doctest: finite scalar loss, `next_code_accuracy` / `perplexity` metrics |
| Cross-entropy loss | §2.4.4; Fig S2 caption | **ported** | `F.cross_entropy(..., ignore_index=PAD_INDEX)` | doctest shows padding is excluded from the objective |
| Composite output token = concept ID × discretised attributes, numeric values in 10 quantile bins (≈98,300 classes) | §2.4.4; head size derived from Table S6 (75.60 M) | **ported** | two pieces. `pipelines/quantiles.yaml` (the shipped default `pipeline:`) fits per-code deciles with `aggregate_code_metadata`, **`train_only: true`**, and writes them into the *artifact's* `metadata/codes.parquet` — `external_meds_dir` is never touched (verified: byte-identical before and after). `Model.composite_targets` bins against them at loss time; `build_composite_index` lays out the head, honouring "where applicable" so a valueless code costs one class rather than ten | `tests/test_composite_targets.py` (7 tests) + a `build_composite_index` doctest. Three carry the weight: the same code with three very different values gets three ordered labels; `value_bins=10` and `value_bins=1` produce an identical text table (the input side is untouched); and the fitted boundaries equal a hand recomputation from the train shards while differing from an all-splits one |
| Prediction head with a per-class bias | Table S6 (75.60 M head vs 75.50 M fixed-vocab table ⇒ 98,307 × 769) | **ported** | `nn.Linear(hidden, vocab_size)`, bias included | doctest counts the head's parameters |
| The head is discarded before downstream representation extraction | §2.4.4; Fig 1 caption | **ported** | `Model.encode` never calls `self.head`; `infer` calls only `encode` | doctest: `encode` output equals `backbone_states` at the last real position, which does not involve the head |
| A `nan` loss when a batch has nothing to predict | not in the paper — a defect guard | **new** | `_check_ar_target` raises before `cross_entropy` silently returns `nan` | doctest with an expected traceback |
| No `has_labels` branch | implied by §2.7 (probe, not fine-tune) | **ported** | `compute_loss` raises on a labelled batch, because in this DAG `supervised_train` is `ProbeTrainCommand` and never instantiates `Model`. The template's stub docstring says otherwise; it is wrong for the `probe` profile | doctest with an expected traceback |

### 3.5 Optimiser, schedule, precision, determinism (§2.4.4, Table S4)

Every one of these has a row, because the template supplies a generic default for all of them and generic
defaults are not the source's.

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| AdamW | §2.4.4 | **ported** | `configs/optimizer/porter.yaml` (new; `optimizer/adamw.yaml` is left in place, selectable with `optimizer=adamw`) | resolved config in every run artifact |
| Peak learning rate 5 × 10⁻⁴ | §2.4.4; Table S4 | **ported** | `optimizer/porter.yaml:lr` — replaces the template's generic `1e-3` | doctest on `configure_optimizers` checks the fallback carries the same value |
| Weight decay 0.05 | §2.4.4; Table S4 | **ported** | `optimizer/porter.yaml:weight_decay` — replaces the template's generic `0.01` | as above |
| Adam β₁ = 0.9, β₂ = 0.95 | §2.4.4; Table S4 | **ported** | `optimizer/porter.yaml:betas` — torch's default is (0.9, 0.999) | as above |
| Weight-decay parameter grouping (norm gains / biases excluded?) | **unstated** | **unstated** | one global decay group, which is what the paper describes; recorded because it is the exact omission the porting guide was written about | — |
| Gradient clipping 1.0 | §2.4.4; Table S4 | **ported** | `configs/trainer/default.yaml:gradient_clip_val` — already 1.0 as generated, now commented as the paper's value rather than a coincidence | — |
| `cosine_with_min_lr` schedule with warmup | §2.4.4; Table S4 | **ported** | `Model.configure_optimizers` (a `LambdaLR` at `interval: step`). Built in Python rather than as a `scheduler` config because it needs `trainer.estimated_stepping_batches`, which no config can express; `model.scheduler=…` still overrides it | — |
| Warmup 150 steps | Table S4 | **adapted** | `warmup_steps: 150`, clamped to `min(150, total_steps // 10)`. The paper's 150 assumes a 894 M-event run; unclamped it would consume most of a short one | — |
| Minimum LR of the cosine | **never given** | **unstated** | `min_lr_ratio: 0.1` of peak, exposed as a config key | — |
| 5 epochs | §2.4.4; Table S4 | **ported** | `trainer/default.yaml:max_epochs: 5` — replaces the template's generic 10 (test harnesses override to 2 / 25) | — |
| Gradient accumulation 2 | Table S4 | **ported** | `trainer/default.yaml:accumulate_grad_batches: 2` | — |
| Early stopping: none | §2.4.4; Table S4 | **deviated** | `callbacks/default.yaml` keeps `ModelCheckpoint(monitor=val/loss, save_top_k=1)`. Nothing *stops* early, but the published checkpoint is the best-validation one rather than the final one, which is a selection the paper does not make | manifests record `checkpoint.selection: monitored_best` |
| bfloat16 mixed precision | §2.4.4; Table S4 | **deviated** | `precision: 32-true`. The template ships fp32 deliberately and the tiers run on CPU; reach for it with `trainer.precision=bf16-mixed` | see deviation 6 |
| 16,384 events per batch | §2.4.3; Table S4 | **adapted** | `batch_size: 1` × `max_seq_len: 16384` in `pretrain`, `infer` and `predict` = exactly the paper's budget with its stated minimum of one patient per batch. The paper *packs* patients to fill the budget; meds-torch-data pads instead, so a dataset of short timelines wastes most of it and should raise `batch_size`. `supervised_train` is excluded and set to 256: in the probe DAG that command runs no backbone at all, so its `batch_size` is the probe's minibatch over dense `[N, 768]` features and an event budget is meaningless there | — |
| Single H100 / L40S; ≈1.01 × 10¹⁸ FLOPs; 894 M events | §2.4.4; §3.1; Table S6 | **omitted** | not reproducible at the scale this repository is exercised at | — |
| Seeds and restarts | **not reported** — each configuration was pretrained once | **new** | `seed: 0` in `pretrain.yaml` / `supervised_train.yaml`; the contract calls `seed_everything(seed, workers=True)` | §6 records two runs at seed 0 agreeing bit-for-bit |
| Determinism flags, matmul precision | **not mentioned** | **new** | `trainer/default.yaml`: `deterministic: warn`, `benchmark: false`. `torch.set_float32_matmul_precision` is **not** called — the paper reports bf16, not TF32, so there is nothing to reproduce here | §6 |

### 3.6 Downstream probe (§2.7.1, Table S4)

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| L2-regularised logistic regression on frozen representations | §2.7.1 | **ported** | `Probe`: a single `nn.Linear(input_dim, 1)` under `binary_cross_entropy_with_logits` | doctest: `[B]` logits, not probabilities |
| Penalty on the coefficients, not the intercept | scikit-learn semantics implied by §2.7.1 | **ported** | `Probe.configure_optimizers` puts `linear.weight` and `linear.bias` in separate decay groups | doctest inspects the two param groups |
| `StandardScaler` fitted on the **training subset** of each task's split | §2.7.1; Table S4 | **ported** | `Probe.on_train_start` fills the `feat_mean` / `feat_std` buffers from the training dataloader only; they ride in the checkpoint, so `predict` scales identically | doctest lists the two buffers in `state_dict` |
| scikit-learn LBFGS, `max_iter = 10000` | §2.7.1; Table S4 | **deviated** | mini-batch AdamW inside Lightning's `fit` loop, because that is what `ProbeTrainCommand` drives. The objective minimised is the same convex one; the solver and its convergence guarantees are not | see deviation 5 |
| `C ∈ {1, 1e-1, 1e-2, 1e-3, 1e-4}` selected on the validation set | §2.7.1; Table S4 | **omitted** | this DAG has no inner hyperparameter search. `l2` is a config key and the grid is a documented `--multirun` (`weight_decay ≈ 1 / (C · n_train)`), but nothing selects automatically | `configs/model/probe.yaml` |
| The metric `C` was selected on | **never named** | **unstated** | — | — |
| Class weighting, intercept fitting, tolerance | **unstated** | **unstated** | unweighted, intercept fitted, torch defaults | — |
| L2-normalising representations before probing | **unstated** for the probe | **omitted** | not applied | — |
| Representation taken "at prediction time" | §2.7.1 | **ported** | `Model.encode` takes the last non-padding position; the task datamodule samples `TO_END`, so that position *is* the prediction time | doctest checks the vector equals `backbone_states` at each subject's own last real event, for subjects of different lengths |
| Which layer the representation comes from | **unstated** | **unstated** | the final layer, after `norm_f` | — |
| Pooling over events | **unstated** | **unstated** | none (`pooling: last`); `pooling: mean` is available and is a masked mean | doctest covers the `last` path |
| AUROC on the held-out temporal split | §2.7.1 | **ported** | `predict` → `predictions.parquet` → `meds-evaluation` (a separate shared tool, by contract) | `test_end_to_end` validates against `PredictionSchema` |

### 3.7 Evaluation and analysis with no home in this DAG

All **omitted**; each is a property of the paper's *study*, not of the model, and this repository's
interface has no slot for any of them. Listed rather than silently skipped.

| element | source reference | why there is no home |
|---|---|---|
| 74 SickKids downstream tasks; MIMIC's 36; prediction time 23:59; cohort exclusions | §2.3; Table S2 | tasks arrive as `external_labels_dir`; a model receives *where* to predict, never *what* |
| SickKids / MIMIC cohorts, 894 M events, REB approval | §§2.1–2.2 | private data |
| Hash-based ≈90/10 patient split | §2.2 | splits come from the MEDS dataset's own shard layout |
| Cross-vocabulary transfer via re-rendered (SEDAR) descriptions | §2.7.3 | the *mechanism* is supported — regenerate `text_table` from a second `codes.parquet` and reload the same weights — but no second vocabulary exists here to test it with. This is the paper's headline result and it is **not reproduced** |
| Cross-site transfer to MIMIC | §2.7.4 | as above |
| Fixed-Vocab FM comparator | §2.5 | a different model |
| Qwen3 text-serialization comparator | §2.5; Fig S4; Table S5 | a different model |
| 329× amortized-FLOPs analysis | §2.9; Fig 2C | an accounting exercise over the two above |
| Synonym-invariance and numeric-ordinal-sensitivity analyses | §2.8; §3.5 | need the private vocabulary and a second rendering |
| Sample-efficiency sweep (k = 2 … 32,768, 10 seeds, balanced) | §2.7.2 | no shot-size loop in this DAG |
| Wilcoxon signed-rank with Holm correction across 74 tasks | §2.10 | scoring is `meds-evaluation`'s job, downstream of this repository |

### 3.8 This port's own additions (no counterpart in the paper)

Recorded so that a reader can tell them apart from ported elements.

| element | why it exists | where it landed | evidence |
|---|---|---|---|
| `OrderStableInferCommand` | the contract's `infer` writes every train-split embedding against another subject's key — see §5.1 | `src/porter/commands.py` | `tests/test_infer_alignment.py` (5 tests; 2 go red against a negative control that restores the contract's loader) |
| Inversion of meds-torch-data's per-code z-scoring | PORTER's numeric scalar is defined on raw values; MEDS-transforms normalises before the model sees anything | `model.py::build_value_statistics`, `model.denormalize_numeric_values` | doctest checks the round trip is exact |
| `render_event_description` | the paper's OMOP slot templates have no direct MEDS counterpart | `model.py`; `CATEGORY_ALIASES`, `FIXED_DESCRIPTIONS` | 7 doctest cases |
| `_check_ar_target` | `cross_entropy` returns `nan` rather than raising when every target is padding | `model.py` | doctest with an expected traceback |
| `configs/model/small.yaml` + `PORTER_*` env vars | the published geometry is the default, so the *tests* need a way down rather than the model needing a way up | `src/porter/configs/model/small.yaml`, `tests/conftest.py` | the whole suite runs on it; `backbone_parameter_count` doctests pin both geometries |
| Pad-safe numeric presence | `numeric_value_mask` is `True` at padding in meds-torch-data | `Model.numeric_features` | doctest asserts a pad position yields all zeros |
| Mask-free global attention + block-sparse windows | the paper names no kernel; a dense `L x L` mask is 268 MB at the published context and blocks FlashAttention | `causal_attention`, `require_right_padding`, `model.attention_impl` | `tests/test_attention.py` (11 tests: every path against a naive reference, flex vs dense on CUDA, and padding proven inert); memory measured in §6.6 |
| SLURM scripts for a cluster run | the paper reports single-GPU runs but ships no orchestration | `slurm/` | `submit.sh` dry-run produces the correct three-job `afterok` chain |

## 4. Deviations that affect results

Read this before comparing anything here to PORTER's published numbers. Ordered by how much they matter.

**1. Nothing in this section is about the composite targets any more.** They are implemented, the
boundaries are fitted on the training split alone (`train_only: true`, asserted against a hand
recomputation in `tests/test_composite_targets.py`), and the input pathway is provably untouched. Kept as
item 1 only to state what the design does *not* do, because it is the trap: the obvious route to
composite tokens, MEDS-transforms' `bin_numeric_values`, rewrites `HR` into `HR//value_[80,90)`. That
moves the magnitude into the *input* description and drops `numeric_value` — which is PORTER-NumText
(§2.6.2), the ablation the paper reports as strictly worse (0.880 vs 0.884 in-domain, numeric ordinal
sensitivity 0.67 vs 1.00). `tests/test_composite_targets.py::test_inputs_are_untouched_by_quantization`
is what keeps this implementation on the right side of that line.

The remaining honest caveat is narrower than it was: the paper's composite token also discretises
*non-numeric* event attributes, which MEDS has already folded into `code` before this repository sees
anything, so that half is structural rather than a choice.

**2. Every measured number in this report is a small-model number.** The *shipped default* is now the
published geometry — 28 × 768, 115,649,280 backbone parameters, 16,384-event context (§3.3) — and §6.6
runs a forward+backward at exactly that scale. But the conformance and learnability tiers cannot train it
on CPU, so they run `model=small` (0.46 M, a factor of 250 smaller) through the `PORTER_*` environment
variables in `tests/conftest.py`. **No AUROC in §6 was produced by PORTER.** The port is verified; the
model is not evaluated.

**3. The reference-range branch of the numeric pathway is never taken.** §2.4.2's primary scalar is the
value's position within its reference range; the log-magnitude fallback is explicitly the *secondary*
path. MEDS standardises no reference-range column, so on every dataset used here 100% of values take the
fallback. The branch is implemented and doctested, and activates automatically for a dataset whose code
metadata carries one of `porter.model.REFERENCE_RANGE_COLUMNS` — but nothing here has exercised it on real
data. Note also that the fallback only sees a clinically meaningful magnitude because the model inverts
meds-torch-data's per-code z-scoring (§3.2); with `denormalize_numeric_values=false` it would be applied to
a standard deviation.

**4. Batches are padded, not packed.** The context window itself is now the paper's (`max_seq_len:
16384`), and `batch_size: 1` reproduces Table S4's "16,384 clinical events per batch, minimum one
patient". What is *not* reproduced is the packing: the paper fills the event budget by concatenating
short patients behind patient-boundary masking, while meds-torch-data pads every sample to the batch
maximum. On a cohort of short timelines the paper's batch therefore holds many more patients than this
one does at the same budget, which changes gradient noise per step and wall-clock per epoch — not what
the model can see. Raising `batch_size` trades the budget back for throughput.

**5. The probe is fitted by mini-batch AdamW, not LBFGS, and `C` is not selected.** The objective is the
same convex one, so with enough steps the optima coincide, but the step budget is not chosen to guarantee
convergence and nothing checks it. On the synthetic fixture the question does not bite — the probe reaches
AUROC 1.000, and a scikit-learn LBFGS fit on the same embeddings also reaches 1.000 (§6.4), so the two
agree where it is measurable. On a real task with a harder boundary they need not. Separately, the paper
picks the regularisation strength per task from a five-point grid on validation; this ships a single fixed
`l2 = 1e-4` and a documented `--multirun`, with no automatic selection.

**6. fp32, not bf16.** No effect on what the model *is*, but it is not the paper's numerical regime, and it
makes any wall-clock or throughput comparison meaningless. Related: RoPE is exercised only trivially here —
the synthetic fixture's `time_delta_days` is a constant 1 hour, so the rotation spans a fraction of a
radian and `rope_theta` / `rope_time_scale` are effectively untested.

**7. The FiLM module follows the prose (128-wide), against Table S6's arithmetic (≈0.46 M ⇒ 256-wide).**
The paper is internally inconsistent here; the prose was followed.

**8. The test tiers run without static demographics, though a real run does not.** The shipped default is
`static_inclusion_mode: PREPEND`, so sex, race and the rest reach the model on any real MEDS dataset.
But the template's synthetic learnability fixture has *no* static measurements, and meds-torch-data
raises on such a cohort under PREPEND (upstream MEDS_model_template#16), so `tests/conftest.py` forces
OMIT for the session. The learnability numbers in §6.2 are therefore from a model that saw no statics —
which for that fixture is vacuously true anyway, since it has none. `tests/test_static_demographics.py`
covers the shipped path separately, on the real fixture.

## 5. Defects found in the vendored contract

[`docs/PORTING-A-MODEL.md`](docs/PORTING-A-MODEL.md) Step 6 notes that this port is the first time several
of the contract's guards run against a real model, and asks for anything that fires to be reported. One
did not fire — it should have.

### 5.1 `infer` mis-assigns every train-split embedding (silent, and it changes results)

**What.** `meds_model_base.commands._runtime.run_predict_step` pairs the index — `dataset.schema_df`, in
index order — with the outputs of `trainer.predict(module, dataloaders=<loader>)`, which come back in
*iteration* order. `SPLIT_ATTRS` is commented "Only the non-shuffling loaders are listed: prediction order
must match `schema_df` order." That is true for `tuning` and `held_out`. It is **false for `train`**:

```python
# meds_torchdata/extensions/lightning_datamodule.py
def train_dataloader(self):
    return self.__dataloader(self.train_dataset, shuffle=True)
```

and meds-torch-data's own class docstring says so outright — *"The train dataloader shuffles so doesn't
return stable outputs, but the others do not shuffle."* Lightning notices (`Your predict_dataloader's
sampler has shuffling enabled…`) but only warns.

So every train-split row of `<data_dir>/inference/<name>/artifacts.parquet` is written against one
subject's key carrying **another subject's** embedding, permuted differently on every run.

**Why it is silent.** The artifact has the right keys, the right row count, the right vector width, a valid
`kind: embeddings` manifest, and passes the probe's coverage check. Nothing downstream can see it. The
conformance suite misses it because `test_end_to_end` scores `splits=[held_out]`, which is one of the two
splits that happen to be correct.

**Measured.** Synthetic signal fixture (200 train / 40 tuning / 40 held-out subjects; the label is presence
of `SIGNAL//POS`). 5-fold cross-validated AUROC of a standardised logistic regression fitted on the
materialized embeddings of each split, from one fixed pretrained checkpoint:

| split | `DefaultInferCommand` | `porter.commands.OrderStableInferCommand` |
|---|---|---|
| `train` | **0.491** (chance) | **1.000** |
| `tuning` | 1.000 | 1.000 |
| `held_out` | 1.000 | 1.000 |

Both columns come from the same checkpoint and the same data; only the loader differs. The left column is
not a stable number — a separate run measured 0.588 on the train split, because the permutation is
redrawn every time. That instability *is* the defect.

And end-to-end: a probe fitted on the scrambled train split scored **AUROC 0.461** on held-out, while one
fitted on the (correctly aligned) tuning split scored **1.000** — from the same embeddings and the same
frozen backbone.

**How it was found.** Not by a failing test. `pytest -m slow` passed both times; the two runs reported
different signal AUROCs at the same seed (0.9427, then 1.0000), and Step 6's rule — *"Two runs that
disagree are a finding about the port — not a reason to record the first number and move on"* — forced the
bisection: `preprocess_data` reproduces byte-for-byte, `pretrain` reproduces bit-for-bit (identical
`val/loss` to 14 digits and identical weight digests at both 6 and 25 epochs), `supervised_train` and
`predict` reproduce bit-for-bit from a fixed input — and `infer` did not, on 200 of 280 rows. 280 − 200 =
80 = exactly the tuning and held-out splits.

Worth stating plainly: **the learnability tier passed for the wrong reason.** The probe trained on
scrambled features and `ModelCheckpoint(monitor=val/loss)` selected whichever epoch happened to score best
against the *clean* tuning split, which is why it still reached 0.94–1.00 and why it landed somewhere
different each run.

**Fix, and where it lives.** `porter.commands.OrderStableInferCommand` builds the loader from the dataset
with `shuffle=False` for every split. It is in the user-owned `commands.py`, not in
`src/meds_model_base/`, because `copier update` overwrites the latter — `CLAUDE.md` is explicit that a
contract problem is "a finding to raise upstream and to record, not a local patch."
`tests/test_infer_alignment.py` pins the invariant directly (batch row *i* holds the timeline of the
subject `schema_df` lists at row *i*, for all three splits) and was checked against a negative control:
reverting `ordered_split_loader` to the contract's loader turns 2 of its 5 tests red.

**Scope beyond this repository.** Every `probe` model is affected identically. `SupervisedPredictCommand`
shares `run_predict_step`, so any model predicting with `splits=[train]` (or the default "every split") is
too. Suggested upstream fix: build the loader from the dataset in `_runtime.split_dataset_and_loader`
rather than reading `train_dataloader()` off the datamodule, and add an alignment assertion to the
conformance suite (`test_end_to_end` predicting `splits=[train]` would have caught it).

**Filed** as [MEDS_model_template#21](https://github.com/florian6973/MEDS_model_template/issues/21).

### 5.2 `infer` and `predict` ignore `cfg.trainer` entirely

Filed as [MEDS_model_template#22](https://github.com/florian6973/MEDS_model_template/issues/22).

`_runtime.run_predict_step` builds its own `Trainer(accelerator="auto", devices=1, logger=False,
enable_checkpointing=False, enable_progress_bar=False)`. Both `infer.yaml` and `predict.yaml` list
`- trainer: default` in their defaults, so `trainer.deterministic`, `trainer.benchmark`,
`trainer.precision` and `trainer.accelerator` all *look* live on those commands and none of them are.

Two consequences, both observed here. The determinism settings the contract added for reproducibility do
not reach inference. And the conformance harness passes `trainer.accelerator=cpu`, so training ran on CPU
while `infer` and `predict` ran on this machine's GPU — the numbers in §6 are from a two-device run, and
nothing said so. There is also no way to ask for `bf16-mixed` at inference, which a full-scale model may
need to fit.

Not worked around here: unlike §5.1 it changes no result, and a subclass would have to duplicate the
whole command to fix it.

### 5.3 The probe profile's rendered `model.py` documents a hook it never calls

Filed as [MEDS_model_template#23](https://github.com/florian6973/MEDS_model_template/issues/23).

`model.py.jinja` renders, for `profile in ['finetune', 'probe']`, "``compute_loss`` — used by ``pretrain``
and ``supervised_train``" and "This DAG trains twice. ``batch.has_labels`` distinguishes them". True for
`finetune`; false for `probe`, where `supervised_train` is `ProbeTrainCommand` and builds `Probe` from
`configs/model/probe.yaml`, never touching `Model`.

Following it produces a `has_labels` branch in the encoder that can never execute, and suggests the
encoder is fine-tuned on the task — the opposite of a frozen-representation probe. This port raises on a
labelled batch instead, with a doctest.

### 5.4 Defects this port introduced, and how they were caught

Recorded because the porting procedure asks for evidence rather than confidence, and because the way
they were found is the point. After the paper-scale defaults landed, four independent reviewers were run
over the change — 50 candidate findings, each handed to a separate agent instructed to *refute* it; 13
survived. The three that mattered were all in code written for this change, and none of them would have
failed a test:

1. **`torch.compile(flex_attention, dynamic=False)` silently degraded to the eager path.**
   meds-torch-data pads to the longest sequence *in each batch*, so the sequence length changes almost
   every step and every distinct length was a fresh compile. Dynamo gives up after `recompile_limit`
   (8 on this install) and runs `flex_attention` eagerly — which materialises the full score matrix:
   **39 GiB for one layer at 16,384**. The `except Exception` fallback then swallowed the resulting OOM
   into the dense path, so a cluster run would have been correct, quadratic, and quiet. Measured before
   the fix: peak per layer jumped from 0.098 GiB at step 8 to 1.398 GiB at step 9. `dynamic=True`
   compiles once; after the fix, memory across eleven distinct lengths grows smoothly (0.095 → 0.177 GiB)
   with no cliff.
2. **The block-sparse mask was rebuilt every layer, every forward** — 229 ms and 2.6 GiB transient at
   16,384, times fourteen local layers. Now memoised, like the dense one.
3. **The dense-mask cache held up to 32 entries** of `[1, 1, L, L]` bools — 268 MiB each at the published
   context, so 7.8 GiB of GPU memory described in a comment as "all small". Both caches are now
   single-entry: the length changes constantly, so a large cache is all cost.

Also fixed from the same review: `attention_impl` was a *saved hyperparameter*, so a checkpoint
pretrained on a Triton-capable node carried `flex` onto nodes that cannot compile it (it is now an
execution detail, not model state); `_FLEX_UNAVAILABLE_REASON` was never set on runtime failure, so
`auto` retried a broken kernel on every layer of every batch; `numeric_features` and `apply_rope`
returned float32 into half-precision modules, which broke `trainer.precision=bf16-true`; a
`local_attention_window` of 0 silently zeroed every local layer instead of raising; and
`slurm/env.sh`'s `${MEDS_ROOT:?}` aborted the *first* documented step, which does not need it.

Two of the review's findings were about the tests rather than the code, and both were fair:
`test_flex_matches_dense` could compare dense against dense when the kernel was unavailable (`flex` now
raises rather than falling back), and the published geometry was unobservable to the suite because
`conftest.py` sets the size variables process-wide — `tests/test_default_is_the_paper.py` now clears
them and asserts the real default.

### 5.5 Guards that did fire, or were checked

| guard | fired? | note |
|---|---|---|
| `predict`'s `CoverageError` | no | coverage was complete on every run; `n_expected == n_written` in every predictions manifest |
| `load_pretrained_weights`' zero-match `RuntimeError` | n/a | the probe DAG never warm-starts — `ProbeTrainCommand` does not load the pretrained checkpoint |
| `_persist_checkpoint`'s `final_weights` fallback | no | `ModelCheckpoint` monitors `val/loss`, which `Probe.validation_step` and `BaseLightningModule.validation_step` both log, so every artifact records `selection: monitored_best` |
| `arbitrate_sources` rejecting two sources | yes | `test_conflicting_sources_are_rejected` passes |
| `read_manifest(require_kind=embeddings)` | yes | exercised on every `supervised_train` and `predict` |
| `_require_split_sharded` | yes | `test_split_sharding` passes |

## 6. Verification

Everything below was executed on this machine, not inferred. Environment: Linux aarch64, NVIDIA GB10,
Python 3.12.13, torch 2.13.0+cu130, lightning 2.5.6, polars 1.43.2, meds 0.4.1,
sentence-transformers 5.6.1, transformers 5.14.1. All training in the test tiers runs on **CPU**
(`run_chain` passes `trainer.accelerator=cpu`), fp32, `deterministic: warn`, `benchmark: false`,
`num_workers=0`, `seed=0`, and at the `model=small` geometry. §6.6 is the exception and the only section
that touches the published scale.

### 6.1 Tiers

| tier | command | result |
|---|---|---|
| lint | `ruff check .` / `ruff format --check .` | clean, 39 files |
| doctests | `pytest src/porter/model.py` | **17 passed** — the paper's constants are pinned here (the `[0, 1.5]` and `[-2, 3]` clips, the divisor 7, FiLM gating and zero-init, the 115.6 M arithmetic, the `TO_END` last-position pooling, the checkpoint round trip) |
| fast conformance | `pytest -m "not slow" -rs` | **117 passed, 1 skipped** |
| learnability | `pytest -m slow -rs` | **2 passed**, 1 skipped (`meds_dev`, opt-in) |
| MEDS-DEV end-to-end | `MEDS_DEV_DIR=… pytest -m meds_dev -rs` | **1 passed** — see 6.5 |

**Skip accounting, which the porting procedure asks for explicitly.**

| | passed | skipped | which skips |
|---|---|---|---|
| before the port (generated stub) | 57 | 2 | `test_end_to_end` (`skip_if_stub`) + `test_unsupported_command_fails_clearly` |
| after | 117 | 1 | `test_unsupported_command_fails_clearly` only |

**Zero `skip_if_stub` skips remain.** The one that stays is correct and cannot be fixed: its `parametrize`
list is empty because the `probe` DAG registers all five commands, so there is no unsupported command to
assert an error for. The +60 tests are 17 doctests, `test_end_to_end` (which was the skipping tier), and
42 added by this port: `test_infer_alignment.py` (5), `test_frozen_text_table.py` (4),
`test_attention.py` (11), `test_default_is_the_paper.py` (12), `test_static_demographics.py` (3) and
`test_composite_targets.py` (7).

### 6.2 Learnability, with the negative control

`tests/test_property.py`, run twice end to end at the same seeds (25 epochs, batch 16, CPU, seed 0):

| run | designed signal (threshold `> 0.9`) | negative control (threshold `≤ 0.70`) |
|---|---|---|
| 1 | **1.0000** | **0.4937** |
| 2 | **1.0000** | **0.4937** |

n = 40 held-out timepoints in each. **The control is not passing trivially.** The identical pipeline on the
signal dataset separates perfectly with probabilities spanning the full `[0.0000, 1.0000]` range, so the
0.4937 is a model that *can* discriminate declining to, not a constant predictor scoring 0.5 by default.
The one-sidedness of the threshold is still worth noting: a perfectly anti-correlated model would also pass
it.

### 6.3 Reproducibility

The claim being made is the achievable one — *same seed + same config + same environment → same metric* —
and it is made stage by stage, because the first attempt at it failed and that failure is §5.1.

| stage | evidence |
|---|---|
| `preprocess_data` | two independent builds produced byte-identical data shards and an identical `codes.parquet` (same `code/vocab_index` assignment) |
| `pretrain`, 6 epochs, seed 0 | `val/loss = 1.90627920627594` in both runs; float64 digest of the full `state_dict` identical (`e10a00fe…`) |
| `pretrain`, 25 epochs, seed 0 | `val/loss = 1.906916856765747` in both runs; digest identical (`c5dc8b2d…`) |
| `infer` → `supervised_train` → `predict` from fixed inputs | probe `val/loss = 0.5317332148551941` in both runs; `predictions.parquet` identical, max abs difference `0.0` |
| the whole chain (learnability tier) | both runs in 6.2 identical to four decimal places on both arms |

**Before the fix in §5.1, the last line read 0.9427 and 1.0000.** That disagreement is what forced the
bisection; it is recorded here rather than replaced, because it is the only reason the defect was found.

### 6.4 The `infer` alignment defect

Measured, from one fixed pretrained checkpoint on the synthetic signal fixture — 5-fold CV AUROC of a
standardised logistic regression on the materialized embeddings:

| split | contract's loader | this repository's loader |
|---|---|---|
| `train` | 0.491 | 1.000 |
| `tuning` | 1.000 | 1.000 |
| `held_out` | 1.000 | 1.000 |

End to end: probe fitted on the scrambled train split → held-out AUROC **0.461**; probe fitted on the
correctly aligned tuning split → **1.000**. Full account in §5.1.

### 6.5 The MEDS-DEV end-to-end tier

This is the tier the template records as never having run past the model. It ran. MEDS-DEV was cloned,
installed editable into its own virtualenv, handed the model through `meds-model-add-to-meds-dev`, and
driven with `meds-dev-model mode=full dataset_type=full`; MEDS-DEV then built a *second*, isolated
environment from this repository's `requirements.txt` and executed the whole chain in it. Artifacts
produced, each with a complete manifest:

```
run/synthetic/unsupervised/train/data/patients/            <- preprocess_data
run/synthetic/unsupervised/train/pretrained/               <- pretrain
run/synthetic/unsupervised/train/data/inference/embeddings <- infer
run/synthetic/smoke_task/train/model/                      <- supervised_train (the probe)
run/synthetic/smoke_task/predict/                          <- predict
```

Checked, rather than assumed, in that isolated environment:

- the pretrained artifact records `module_class: porter.model.Model`, `seed: 0`, 5 epochs,
  `checkpoint.selection: monitored_best` on `val/loss = 1.507`;
- its resolved config shows the **real** encoder was used — `text_encoder: FremyCompany/BioLORD-2023`,
  `d_text: 768` — not the offline control arm;
- the inference artifact declares `kind: embeddings`, 21 rows, `width: 128`;
- the probe artifact records `module_class: porter.model.Probe`, `input_dim: 128`,
  `feature_column: embedding`, `from: inference_artifacts`, 13/13 train labels matched;
- `predictions.parquet` validates against `meds_evaluation.PredictionSchema` and the manifest records
  `held_out: n_expected 4, n_written 4`.

Two caveats, stated because the tier's speed (≈26 s) invites the wrong conclusion: the fixture is six
subjects, and the run reuses warm `uv` and HuggingFace caches. It proves the *plumbing* end to end — MEDS-DEV
can install, register, drive and score this model — not anything about model quality.

### 6.6 The published scale actually runs

Deviation 2 used to say the paper's geometry had never been executed here. It has now. One
forward+backward of `Model(backbone=PAPER_BACKBONE)` — 115,649,280 backbone parameters — under
`autocast(bfloat16)` on the GB10, batch of one patient, measured with `torch.cuda.max_memory_allocated`:

| `attention_impl` | context | peak GPU memory | fwd+bwd | loss |
|---|---|---|---|---|
| `dense` | 4,096 | 4.36 GiB | 0.88 s | 8.4846 |
| `dense` | **16,384** (the paper's) | **20.23 GiB** | 3.65 s | 8.4850 |
| `flex` | 4,096 | 3.90 GiB | 4.11 s | 8.4850 |
| `flex` | **16,384** | **12.98 GiB** | 2.86 s | 8.4848 |

Three things worth reading off it. The published model **fits on a single 40 GB GPU at full context**, so
the paper's "a single NVIDIA H100 or L40S" is reproducible. The block-sparse path saves 7.3 GiB (36%) at
16,384 and the margin grows with context, which is why `auto` selects it there. And the losses agree to
three decimals across implementations at full scale — the same equivalence `tests/test_attention.py`
asserts exactly at small scale, holding where it is expensive to check.

The `flex` numbers include a one-off `torch.compile`; they are not a throughput comparison.

**Caveat on the kernel.** `flex_attention` compiles through Triton, which builds a small C shim at
runtime and therefore needs Python development headers and a C compiler on the node. The system Python on
this machine ships no `Python.h`, and the path failed until the environment was rebuilt on a uv-managed
interpreter — `causal_attention` degrades to the dense mask in that case rather than failing, which is
why the numbers above have a `dense` row at all. See the README's cluster notes.

### 6.7 Determinism settings used for all of the above

`seed: 0` with `seed_everything(seed, workers=True)`; `num_workers: 0`; `trainer.deterministic: warn`;
`trainer.benchmark: false`; `precision: 32-true`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set by the contract);
`torch.set_float32_matmul_precision` **not** called. Training in the test tiers is forced to CPU by
`run_chain`. Note that `infer` and `predict` build their own `Trainer` with `accelerator="auto"` and
therefore ran on the GPU (NVIDIA GB10) while training ran on CPU — the contract exposes no way to change
that, and it is a second (benign, but real) gap between what the trainer config says and what those two
commands do.

## 7. Work required to complete the port

In priority order:

1. **Fit the quantile boundaries on the training split only** (deviation 1) — a train-only pre-pass
   joined back into the shared metadata, which the template's single `pipeline:` hook cannot express.
2. **A pretraining run at the published scale on real data.** The geometry, the context window and the
   kernels are in place and measured (§6.6), and `slurm/` submits the chain — but no run against a real
   MEDS dataset with real code descriptions has been done, so nothing here says what PORTER *scores*.
3. **The cross-vocabulary experiment** (§2.7.3), which is the paper's central claim and is *structurally*
   supported here but untested: regenerate `text_table` from a second `codes.parquet` describing the same
   cohort with different concept names, reload the same weights, and compare AUROC. Doing this needs a
   command-level way to swap the buffer in a published checkpoint — today it means re-running `pretrain`.
4. **A reference-range-carrying dataset** (deviation 3), to exercise §2.4.2's primary numeric branch on
   real data rather than only in a doctest.
5. **Probe convergence and the `C` grid** (deviation 5): either drive `--multirun` selection from a
   validation metric, or replace the Lightning loop with a full-batch LBFGS fit.
6. **Demographics** (deviation 8): decide whether `static_inclusion_mode: PREPEND` is workable for the rest
   of this DAG, and if so render Table S3's demographic templates.
7. **Upstream**: three issues are filed against the template —
   [#21](https://github.com/florian6973/MEDS_model_template/issues/21) (the `infer` misalignment, §5.1),
   [#22](https://github.com/florian6973/MEDS_model_template/issues/22) (`infer`/`predict` ignore
   `cfg.trainer`, §5.2) and
   [#23](https://github.com/florian6973/MEDS_model_template/issues/23) (the probe profile's rendered
   `model.py` docstring). Delete `OrderStableInferCommand` once #21 lands.

## 8. Step 7 checklist

The porting procedure's "before claiming completion" list, answered rather than ticked.

| item | status |
|---|---|
| Every entry point in the source's `pyproject.toml` appears in the ledger | **n/a, and stated why** — there is no package and no repository (§1). The ledger is keyed to paper sections instead, and each row cites one. |
| The batch-field test was performed and its conclusion recorded | **yes** — §2. PORTER reads *more* of the batch than a fixed-vocabulary model, so the input-side failure mode does not apply; the three real gaps (reference range, demographics, composite targets) each have a row. |
| Every optimiser detail (grouping, LR, schedule, warmup, clipping) has a ledger row | **yes** — §3.5, sixteen rows, including the two the paper leaves unstated (decay grouping, the cosine floor). |
| Seeding, determinism and matmul precision have ledger rows, and the settings are in Verification | **yes** — §3.5 and §6.7. `torch.set_float32_matmul_precision` is deliberately not called, and that is a row, not a silence. |
| Training ran twice at one seed and the reported metric agreed | **yes, after a fix** — §6.3. It did *not* agree on the first attempt (0.9427 vs 1.0000), and chasing that disagreement is what surfaced §5.1. Both arms are now identical across runs. |
| `pytest -m "not slow" -rs` shows zero `skip_if_stub` skips; record the counts before and after | **yes** — 57 passed / 2 skipped before, 82 passed / 1 skipped after. The remaining skip is an empty `parametrize` list, not a stub gate (§6.1). |
| `test_smoke_pipeline` and `test_property` both ran; a trivially-passing negative control is a finding | **yes** — §6.2. The control sits at 0.4937 while the identical pipeline reaches 1.0000 with probabilities spanning `[0, 1]` on the signal dataset, so it is not passing by emitting a constant. |
| No ledger row is justified by the test fixture | **yes.** The one row that used to be — a scaled-down default backbone, chosen so the tiers ran on CPU — has been removed: `configs/model/default.yaml` is now the published 28 × 768 geometry at 16,384-event context, and the *tests* scale down instead, through the `PORTER_*` environment variables in `tests/conftest.py`. That is the direction the porting procedure asks for, and it names env vars as the legitimate way to do it. Of the remaining justifications: the composite-token omission is the size of the preprocessing work, the reference-range branch is MEDS having no such column, the LBFGS deviation is the template's Lightning training loop, and demographics is `static_inclusion_mode` in the datamodule contract. |
| "Deviations that affect results" is present, even if it says *none* | **yes** — §4, eight of them. |
| Every row has evidence or is marked unverified | **yes.** Rows marked **unstated** carry no evidence by construction — there is nothing to verify them against — and are labelled as such rather than presented as ported. |

**One thing this report cannot claim.** No number here is comparable to the paper's. Every measurement is
on a synthetic six- or 280-subject fixture whose "descriptions" are the code strings themselves, with a
backbone 250× smaller than the published one and a pretraining objective over bare codes. What has been
demonstrated is that the architecture is implemented as described, that its pieces are individually pinned
to the paper's own constants, and that the whole DAG runs and learns a designed signal end to end —
including through MEDS-DEV. What has *not* been demonstrated is any claim PORTER actually makes:
in-domain parity with a fixed-vocabulary model, cross-vocabulary transfer, or cross-site transfer. §7 lists
what closing that would take.
