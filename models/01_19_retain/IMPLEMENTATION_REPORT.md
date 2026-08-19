# Implementation report: RETAIN

## Source

- Paper/repository: `cards/01_supervised_models/19_retain.md`,
  https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html, and
  https://github.com/mp2893/retain
- Exact revision/version:
  - Paper: Choi, Bahadori, Kulas, Schuetz, Stewart, Sun, *RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism*, NIPS 2016, pp. 3504-3512. Read in full from arXiv
    `1608.05745v4` (26 February 2017, 6 pages).
  - Official code: `github.com/mp2893/retain` @ `9fd39c46e44b3fdae3a667a47fbdd5ffbd87bff5`
    (2017-11-03, tip of `master`; the repository has had no commit since). Files read in full:
    `retain.py`, `test_retain.py`, `process_mimic.py`, `README.md`.
  - Third-party, consulted only: PyHealth `pyhealth/models/retain.py`, `sunlabuiuc/PyHealth` @ `master`,
    fetched 2026-08-15. Not followed; see *Scientific deviations* item 4.
- Availability of the appendices: the arXiv record for this paper is the 6-page conference version. Its
  Appendices A (timestamps), B (hyper-parameter tuning) and C (ESM) are referenced in the body but are not
  in that PDF. Every hyper-parameter default used here therefore comes from the official code, which is
  the authoritative artefact for exactly the details those appendices would have carried.

## Model profile choice

- Selected model profile: `supervised`
- Justification: RETAIN has no task-independent stage to place anywhere else. `train_RETAIN` in
  `retain.py` initialises `W_emb`, both attention GRUs and the risk head together, and optimises all of
  them against the labelled endpoint in one loop; there is no pretraining objective, no frozen encoder and
  no reusable inference product. The official README states that the released code performs sequence
  classification, which is exactly the labelled-endpoint setting MEDS-DEV supplies. The decomposition used
  here is `preprocess_data` (task-independent tokenisation of the code stream, no labels read),
  `supervised_train` (the whole model, fitted to one task), `predict` (scoring a task index). The graph was
  not changed to avoid implementing a stage; there is no stage to avoid.

## Source-element ledger

| Source element | Status | Implementation evidence | Justification/deviation |
|---|---|---|---|
| Step 1, `v_j = W_emb x_j`, linear, no bias, no activation (`emb = T.dot(x, W_emb)`) | ported | `model.py` `Retain.forward`, `F.embedding_bag(..., mode="sum")` | `x` is binary, so summing the named rows of `W_emb` is the same product without materialising a `|C|`-wide one-hot |
| Step 2, `RNN_alpha` over reversed visits, `e_j = w_alpha g_j + b_alpha`, `alpha = softmax(e)` | ported | `model.py`, `gru_alpha` + `w_alpha` + masked `softmax(dim=1)` | softmax normalises over real visits only; padded positions are masked to `-inf` |
| Step 3, `RNN_beta` over reversed visits, `beta_j = tanh(W_beta h_j + b_beta)` | ported | `model.py`, `gru_beta` + `W_beta` + `tanh` | |
| Step 4, `c_i = sum_j alpha_j beta_j * v_j` | ported | `model.py`, `(alpha.unsqueeze(-1) * beta * visits).sum(dim=1)` | |
| Step 5, `y = sigmoid(w_output c + b_output)` | ported | `model.py`, `w_output` | binary case, as in `build_model`; the paper's softmax reduces to this for `s = 1` |
| GRU cell of `gru_layer` (`r`, `z`, `n` with the reset gate on the whole recurrent term) | ported | `nn.GRU`, verified against a NumPy transcription in `tests/test_model_fidelity.py` | `nn.GRU` is gate-for-gate identical; `bias_hh` pinned to zero, see deviation 5 |
| Reverse time ordering, then hidden states re-reversed (`gru_layer(...)[::-1]`) | ported | `model.reverse_padded`, applied to the input and again to both hidden sequences | per-row reversal, so padding never enters a hidden state |
| `* 0.5` scaling of both GRU outputs | ported | `model.py`, `alpha_states * 0.5`, `beta_states * 0.5` | in `retain.py` *and* `test_retain.py`; not in the paper. Kept, and called out as deviation 3 |
| Dropout on the embedding, feeding both the GRUs and the context sum (`keep_prob_emb`) | ported | `model.embedding_dropout` applied before the split | `build_model` rebinds `emb`, so the dropped tensor is used in both places |
| Dropout on the context vector (`keep_prob_context`) | ported | `model.context_dropout` | |
| Loss: `-(y log(y_hat + eps) + (1-y) log(1 - y_hat + eps))`, mean over the batch | ported | `training._cross_entropy` | written out rather than `binary_cross_entropy_with_logits`, because `log_eps` sits inside both logarithms upstream |
| L2 on `w_output`, `w_alpha`, `W_beta`, `W_emb`, weights only | ported | `model.regularization` | added to the cost, not folded into optimizer weight decay, matching `build_model` |
| Adadelta solver (rho 0.95, eps 1e-6, unit learning rate) | ported | `training._optimizer` | the Theano block is exactly `torch.optim.Adadelta(lr=1.0, rho=0.95, eps=1e-6)` |
| Adam solver (`lr=0.0002`, `b1=0.1`, `b2=0.001`) | ported | `training._optimizer` | upstream parameterises `b` as `1 - beta`, so this is the usual `(0.9, 0.999)`; its `fix1`/`fix2` are standard bias correction |
| Weight init `uniform(-0.1, 0.1)`, biases zero (`get_random_weight`, `init_params`) | ported | `model.reset_parameters` | applied to the GRU weights too, replacing PyTorch's default init |
| Length-sorted minibatches (`len_argsort`) | ported | `training._length_sorted` | |
| Reshuffled batch *order* each epoch (`random.sample(range(n_batches), n_batches)`) | ported | `training.train`, seeded `random.Random` | |
| Model selection on best validation AUC per epoch | ported | `training.train`, `TrainingHistory.best_epoch` | records `selected_by` so a run with no usable validation signal cannot silently claim a selection |
| Batch size 100, 10 epochs, embed/alpha/beta 128, L2 1e-3, keep-probs 0.5, `log_eps` 1e-8 | ported | `config.RetainConfig` defaults | every default is the official `retain.py` default |
| Optional visit durations (`--time_file`, `--use_log_time`), concatenated to the GRU inputs only | ported | `config.use_time`, `windows.visit_gap_days`, `model.forward` | default `false`, which is the upstream default (no `--time_file`) |
| Eq. (5) per-code contributions (`test_retain.py`) | ported | `interpret.write_contributions`, verified in `tests/test_commands.py::test_contributions_reproduce_the_score_they_explain` | off by default (`interpretation_top_k=0`); NDJSON, not parquet, see *Intermediate artifacts* |
| Prediction at every timestep (`attentionStep` scanned over all prefixes) | adapted | `model.forward` computes the final timestep only | the training graph reads `y_hat[lengths-1]`, so for sequence classification only `c_T` is used; `test_retain.py` computes exactly that. Same value, `O(T)` instead of `O(T^2)` |
| Theano 0.8 / Python 2.7 runtime | adapted | PyTorch `>= 2.2` | the original stack is not installable today; the cell and optimiser map exactly, and the fidelity test pins the equivalence |
| `sklearn.roc_auc_score` | adapted | `training.roc_auc`, rank-based with tie correction | avoids a scikit-learn runtime dependency for one function; tested against the pairwise definition |
| `process_mimic.py` visit construction (admissions -> ICD-9 code sets) | adapted | `preprocess.py` + `windows.py` | MEDS has no admission concept without dataset predicates; see *Inputs* |
| 20/10/70 or `.train`/`.valid`/`.test` file split (`load_data`, `load_data_simple`) | adapted | `commands._training_rows` | the MEDS dataset's own `train`/`tuning`/`held_out` splits are authoritative; the seeded fallback only fires when no `tuning` split exists |
| 3-digit ICD-9 truncation / medical groupers (README, paper §4.1) | omitted | -- | both are dataset-specific vocabulary knowledge. Generic frequency filters (`min_code_count`, `max_vocab_size`) provide the same dimensionality control without encoding ICD semantics in a predicate-free port |
| Demographics and outcome markers excluded from visits (paper §4.1 "diagnosis, medication and procedure codes"; `process_mimic.py` builds sequences from `DIAGNOSES_ICD` alone and reads `PATIENTS.csv` only for the label) | ported | `preprocess.MEDS_STANDARD_MARKERS`, `config.exclude_meds_standard_markers` | Drops `MEDS_BIRTH` and `MEDS_DEATH`. These are MEDS-standard codes, not dataset predicates, so naming them is not dataset knowledge |
| Non-binary input values (paper Eq. 5 remark) | omitted | -- | the official code is binary-only (`xvec[subseq] = 1.`); MEDS `numeric_value` is ignored |
| External pre-trained code embeddings (`--embed_file`, `--embed_finetune`) | omitted | -- | requires an external Med2Vec-style artefact that MEDS-DEV does not supply. `W_emb` is always randomly initialised and fine-tuned, which is the documented default |
| Re-training from a saved model (`--model_file`) | omitted | -- | no contract command corresponds to resuming a fit; a rerun is a fresh `supervised_train` |

## Inputs and predicates

- **MEDS assumptions.** Only `subject_id`, `time` and `code` from `data/**/*.parquet`, plus
  `metadata/subject_splits.parquet` when present. Nothing reinterprets MEDS semantics: static events are
  the rows with `time = null`, splits are the dataset's own, and `code` is used as an opaque string.
- **Required predicate names: none.** This port is rendered `uses_predicates=false` and is predicate-free
  by construction, not by omission. RETAIN's published input is a sequence of medical codes and the MEDS
  `code` column already is one, so there is nothing for a dataset predicate to resolve.
  `external_predicates_file` is **rejected** by every command rather than accepted and ignored, so no run
  can record a predicates digest for an input that did not influence it. `model.yaml` contains no
  `{predicates_path}` slot and the repository ships no `predicates.yaml`. No charted vital is referenced,
  so the missing `chartevents` in the current full extract does not constrain this port.
- **Predicate interpretation and derived features: not applicable.** The MEDS adaptation is instead:

  | RETAIN concept | MEDS realisation |
  |---|---|
  | medical code | the `code` string, verbatim |
  | vocabulary `C` | learned in `preprocess_data` from the **training split only**, ordered by descending frequency then code string so indices are reproducible. `min_code_count` and `max_vocab_size` filter it. Closed: a code absent from it is dropped and counted, never folded into a shared out-of-vocabulary bucket |
  | visit | events bucketed onto an absolute grid of `visit_bucket_minutes` (default 1440, one calendar day, the closest generic analogue of the admission-level visits of `process_mimic.py`). `0` keeps every distinct timestamp as its own visit |
  | `x_j` | binary multi-hot over the vocabulary. A code repeated inside a bucket contributes once |
  | history for a key | events with `time <= prediction_time`, optionally limited by `lookback_days` and to the most recent `max_visits` visits. Both default to unrestricted |
  | static events | excluded. A `time = null` event has no position in a reverse-time visit sequence, and the published input is visit codes |
  | `MEDS_BIRTH` / `MEDS_DEATH` | excluded by default (`exclude_meds_standard_markers`). The paper's visits are diagnosis, medication and procedure codes, and the official `process_mimic.py` reads `PATIENTS.csv` only to derive the label. Keeping `MEDS_BIRTH` made it the first visit of 100% of demo windows, decades before the rest; keeping `MEDS_DEATH` would put the canonical outcome marker in the input space of any task whose window reaches it |
  | `numeric_value` | ignored, matching the binary official implementation |
  | splits | MEDS `train` fits the parameters, `tuning` selects the epoch, `held_out` is scored |

- **Why the history is inclusive of the prediction time.** MEDS-DEV's `mortality/in_icu/first_24h`
  declares `index_timestamp: end` with `end_inclusive: True`, so `prediction_time` *is* the last instant
  of the input window the task grants. Excluding it would discard data the task definition hands to the
  model. It carries no outcome information: the task's `gap` window runs from the trigger to `+48h` and
  `target` begins after `gap.end`, so the label cannot be determined until at least 24 hours after the
  prediction time. RETAIN filters raw events, so `<=` admits exactly the boundary instant and nothing
  else. `include_events_at_prediction_time=false` is available for a task whose label-defining event could
  coincide with its own prediction time. This boundary is derived from the task definition; other ports in
  this repository resolve the same question differently because their aggregation granularity differs, and
  no cross-port claim is made here.
- **Empty histories are scored, not excluded.** A key whose subject has no qualifying event yields a
  sequence of length zero, which is padded to one all-zero visit: `v = 0`, so `c = 0` and the score is
  `sigmoid(b_output)`, the model's learned intercept. This keeps the published index exactly equal to the
  requested index with **no exclusion policy**, at the cost of an uninformed prediction. The count and
  fraction are reported per split at every gate and must be read before any performance figure.

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| `preprocess_data` | `retain-patients/v1`: `sequences/**.parquet` of `(subject_id, time, code_index)` sorted and deduplicated, `vocabulary.parquet`, `subject_splits.parquet`, `config.json` | -- | `external_meds_dir` path; config digest |
| `supervised_train` | `retain-supervised-model/v1`: `model.pt` (state dict + vocabulary size), `model.json` (config, training history, split roles, the patients-artifact path) | requires `artifact.type == patients` and payload format `retain-patients/v1`, and that `min_code_count`/`max_vocab_size` match the published vocabulary, all before any expensive work | `input_data_dir` path + its config digest, `external_labels_dir` path |
| `predict` | `meds-evaluation-predictions/v1`: `predictions.parquet`, optionally `interpretation.ndjson` | requires `supervised_model` + `retain-supervised-model/v1`, re-checks the patients artifact, and refuses any configuration argument that would change the model it loaded | both input artifact paths and formats, `external_labels_dir`, requested `splits` |

Events are stored per `(subject_id, time)` rather than per visit, and bucketed into visits only *after*
the prediction-time cut, so no visit can straddle a prediction time and carry later events into its own
history. This is tested directly
(`tests/test_data_path.py::test_a_bucket_is_cut_at_the_prediction_time_rather_than_straddling_it`).

Eq. (5) contributions are written as NDJSON rather than parquet on purpose: the contract's own
`assert_prediction_keys` concatenates every `*.parquet` under the predictions directory, so a second keyed
parquet there would present as duplicate prediction keys and fail the coverage check.

Publication is atomic through `measured_artifact`: work happens inside the context manager, the manifest is
written last, and a failure removes the staging directory without publishing
(`tests/test_commands.py::test_a_failed_command_publishes_nothing`).

## Scientific deviations

1. **Only the final timestep is computed.** The official training graph scans `attentionStep` over every
   prefix and builds `c_t` for all `t`, then reads `y_hat` at `lengths - 1`. For sequence classification
   the other timesteps never reach the loss, so computing `c_T` alone is the *same value* while turning an
   `O(T^2)` scan into `O(T)`. The official `test_retain.py` computes exactly this single-timestep form.
   Verified by `tests/test_model_fidelity.py`, which compares against a NumPy transcription of
   `build_model` to `rtol=1e-9`.
2. **PyTorch instead of Theano 0.8 on Python 2.7.** The original stack is not installable on a current
   machine. `nn.GRU` is gate-for-gate identical to `gru_layer`, and both solvers map exactly onto their
   Theano blocks.
3. **The `* 0.5` scaling is kept.** It appears in both `retain.py` and `test_retain.py` but nowhere in the
   paper. On `alpha` it acts as a softmax temperature of 2. It is upstream behaviour applied consistently
   in two independent files, so it is treated as deliberate rather than as a typo; PyHealth omits it.
4. **PyHealth is not followed.** Its `RETAINLayer` computes `alpha` and `beta` from the reversed sequence
   and then multiplies them against the **forward-order** `x` (`c = attn_alpha * attn_beta * x`) without
   reversing the hidden states back, where the official code does reverse them (`gru_layer(...)[::-1]`).
   That misaligns each attention weight with a different visit and changes the model. The paper and the
   official repository are authoritative, so this port reverses back.
5. **`bias_hh` is pinned to zero and excluded from optimisation.** Theano's `gru_layer` has a single bias
   added to `W x`; `nn.GRU` has a second recurrent bias whose `n` component sits *inside* the reset-gate
   product as `r * b_hn`, a term the official cell does not contain. Freezing it at zero makes the two
   cells algebraically identical rather than merely similar. Asserted in
   `tests/test_model_fidelity.py::test_the_recurrent_bias_that_theano_does_not_have_is_pinned_to_zero`.
6. **AUC is computed here** with a rank-based implementation rather than by taking on scikit-learn as a
   runtime dependency, tested against the pairwise definition including ties.
7. **Visit bucketing, vocabulary capping, `max_visits`, the split-role mapping and the empty-history
   policy have no upstream counterpart.** They are the MEDS adaptation, listed as *adapted* in the ledger,
   and they are the part of this port that a reviewer should scrutinise most.
8. **Model selection when there is no validation signal.** If the dataset declares no `tuning` split, a
   seeded `validation_fraction` hold-out is carved from the training rows. If validation AUC is undefined
   (one class only, or no rows), the final epoch is kept and the manifest records
   `selected_by: final_epoch` instead of claiming a selection that never happened.

### Template revision

Rendered at `MEDS_model_template` `15d76f7dd96178115668447f7e4a936e0bafa6a2`, updated to
`280b31d1b91b42940e2b9810d206fd84056fa16c` via `copier update` (a `.copier-answers.yml` reconstructing the
original render was committed first, so the delta applied as a three-way merge rather than by hand). The
update brought: the `DAGS` -> `MODEL_PROFILES` rename and `ModelRepository.dag` -> `.profile_commands`; the
required `tests/test_primary_behavior.py` and its `tests/signal.py` fixture; and upstream versions of the
two fixes this port had found independently.

**Both local fixes were replaced by upstream's rather than stacked.** Upstream's `-C` fix is functionally
identical to the one carried here. Upstream's PATH fix is at the same single call site this port
established -- `meds-dev-task` only, not `meds-dev-dataset` -- so the narrow version was correct. Its
comment attributes the failure to an *ambient* MEDS-DEV install winning the name; the failure measured here
on a clean host was that `aces-cli` and `expand_shards` were **not found at all** (exit 127), because
running `.venv/bin/meds-dev-task` by absolute path never activates that virtualenv. The fix is right either
way; the rationale differs, and someone debugging a 127 from upstream's comment would look in the wrong
place.

### Deviation from the rendered scaffold

- `src/meds_model_contract/manifest.py` was replaced with the text already carried by
  `origin/model/01_01_teco` and `origin/model/02_11_duett`, which adds the optional `gpu_metrics` and
  `extra` parameters to `measured_artifact`. This is a **contract-package matter outside the model-owned
  surface**: it was raised with the master coordinator before the edit and approved as a cross-port
  consistency decision, not a gate approval. The file is byte-identical across the ports
  (`sha256:6564eeb0de306518487ad5d27d23d8c5c3b6c6f0887cc423172fc36689e92f5e`). RETAIN needs both: SPEC
  section 6's GPU block would otherwise be written as `null`, meaning "not measured", on a run that did use
  a GPU; and section 5's diagnostic coverage counts need a place in the manifest, which is the only file
  that reaches the synchronised run evidence. The other four files in `src/meds_model_contract/` are
  unchanged and were already byte-identical to both other ports. `tests/integration.py` is unmodified and
  still hashes to `cb5a2eccbafe8cb9f0c2c2329bd56c165596a54d3d2099cd9eb95d1c7753ed58`.
- `pyproject.toml` gains `numpy`, `polars`, `pyarrow` and `torch` as runtime dependencies, converging with
  the sibling ports' pins.

## Verification

### Primary behavior

- **Signal and expected behavior**: `tests/signal.py`'s `SIGNAL//POS` alone determines the label. RETAIN's
  primary behavior is to represent a medical code and let the visit attention decide where it mattered, so
  the model must reach a high held-out AUC on this cohort and must attribute the score to that code.
- **How the signal enters the model's selected codes/features**: through the learned vocabulary. This port
  has three places a code can be dropped before the model sees it -- the `min_code_count`/`max_vocab_size`
  frequency filter, the MEDS-standard-marker exclusion, and the prediction-time window -- so the test
  proves the binding at two levels rather than one: `SIGNAL//POS` is in `vocabulary.parquet`, *and* its
  code index appears in the built multi-hot sequence of exactly the positive subjects and no others
  (`np.array_equal(carries_signal, labels)`). It further asserts the signal sits at more than one visit
  index and that sequence length does not separate the classes, so a position- or length-counting model
  could not pass.
- **Example generator used or adaptation made**: `tests/signal.py` used unchanged. One model-side
  configuration change: `visit_bucket_minutes=60` instead of the port's 1440 default. The generator spaces
  its events one hour apart, so a daily bucket would collapse every subject to a single visit and the test
  would never exercise the visit attention that distinguishes RETAIN from a bag-of-codes model. That is a
  configuration choice, not a change to the fixture.
- **Positive-control result**: held-out AUC **0.937** (40 epochs, embed/alpha/beta 32, batch 32, CPU). The
  Eq. (5) attribution names `SIGNAL//POS` as the top-contributing code for **>80%** of positive keys, which
  is the interpretability claim itself under test rather than only the score. AUC saturates below 1.0 for a
  faithful reason: `alpha` is a softmax over visits and the official `* 0.5` scaling halves its logits, so
  attention cannot fully concentrate on the single visit carrying the code. Ten epochs already reach 0.907.
- **Shuffled-label negative-control result**: held-out AUC **0.460** on the identical cohort with labels
  shuffled, a gap of **+0.477**. Without this the positive result would not distinguish "the model learned
  the code" from "the fixture leaks".

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | 72 passed, 2 skipped, 0 failed. **0 `model_stub` skips** | Real three-command model profile on the synthetic fixture. 66/66 requested keys covered exactly, keys unique, prediction schema aligned by `meds_evaluation.PredictionSchema` |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | 1 passed, 53 deselected, 0 skipped, 0 failed | `meds-dev-model model=retain mode=full dataset_type=supervised` in MEDS-DEV's own isolated venv (torch 2.13.0 and `retain 0.1.0` installed there from `requirements.txt`). 13/13 `held_out` keys covered exactly |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | 1 passed, 67 deselected, 0 skipped, 0 failed | Full model profile on the MEDS-DEV MIMIC-IV demo and the pinned `mortality/in_icu/first_24h` task. 6/6 `held_out` keys covered exactly, keys unique, schema aligned |
| Full MIMIC-IV | `./slurm/submit.sh` per task, ten tasks | executed at run SHA `54a8f31` | **Results not yet in this report.** Per-task AUROC/AUPRC exist and are held by the analysis session (issue #7); they are deliberately not transcribed here — see below |

The two skips in the local run are `tests/test_meds_dev_e2e.py` and `tests/test_mimic_demo_e2e.py`, both
reported by `-rs` as `select -m meds_dev or -m real_data, or set MEDS_DEV_DIR`. They are the marker-gated
integration tiers, not stub skips; the `model_stub` skip guards are unreachable because `IS_STUB` is
`False`. On a host with no CUDA device, five tests in `tests/test_device.py` add a fifth accounted-for skip
reason, `no CUDA device on this host`; on this host they run.

**Both reported gates exercised the GPU.** `device` defaults to `auto`, so `supervised_train` and `predict`
resolved to CUDA in the local and MEDS-DEV runs, and their manifests carry real
`torch.cuda.max_memory_allocated` byte counts rather than nulls. The demo and full runs will therefore not
be the first time this port touches an accelerator. `tests/test_device.py` additionally pins the whole profile
on CPU only, on CUDA only, and trained-on-CUDA/scored-on-CPU, since a CPU-only node is an ordinary cluster
shape that a GPU-defaulting suite would otherwise never cover.

Beyond the contract's own conformance tests, this port adds:

- `tests/test_model_fidelity.py` (11 tests) -- the batched implementation against a NumPy transcription of
  the Theano equations, with and without visit durations; padding-invariance of a row's score; the
  empty-history intercept; attention summing to one over real visits only; the pinned `bias_hh`; and the
  AUC implementation against the pairwise definition.
- `tests/test_data_path.py` (17 tests) -- the MEDS-to-visits layer: static-event exclusion, training-split
  vocabulary, closed-vocabulary dropping, daily bucketing and within-bucket deduplication, per-timestamp
  bucketing, the prediction-time boundary in both settings, the non-straddling cut, `lookback_days`,
  `max_visits`, empty histories, visit gaps, and both split-map sources.
- `tests/test_device.py` (9 tests, 5 CUDA-gated) -- the same-seed determinism control, exact on CPU and
  within a stated tolerance on CUDA, its different-seeds counterpart, the whole profile on CPU and on CUDA, a CUDA checkpoint
  scored on CPU, device-independent weight initialisation, CPU/CUDA numerical agreement with dropout
  disabled, and that the pinned `bias_hh` survives a device move and GPU training.
- `tests/test_primary_behavior.py` (4 tests, required by the template and not marker-gated) -- the signal
  binding at vocabulary *and* sequence level, the fixture's position/length defences verified rather than
  assumed, the real profile's learned response with a shuffled-label control, and RETAIN's own Eq. (5)
  attribution naming the signal code.
- `tests/test_determinism.py` (5 tests) -- that a run reproduces from the content of its inputs rather
  than their file layout: permuted label row order leaves predictions bit-identical, `read_labels` imposes
  a canonical order, minibatch composition is invariant to the order keys arrive in, vocabulary selection
  breaks frequency ties on the code string, and the fixture itself rebuilds byte-identically.
- `tests/test_commands.py` (19 tests) -- exact key coverage including empty histories, split restriction,
  the Eq. (5) contributions summing back to the score they explain, artifact type/format rejection,
  predicate rejection on all three commands, mistyped-argument rejection, the preprocess-only vocabulary
  guard, atomic non-publication on failure, overwrite protection, gradient flow to every trainable
  parameter, and that training separates a synthetic separable signal.

## MIMIC-IV demo gate

Approved by the repository owner on issue #3 at commit `ae298931d3b267de94ba353f5921b74000151ca6`, for the
demo cohort and the pinned task only. Run on 2026-08-16.

### Provenance

| Item | Value |
|---|---|
| MEDS-DEV revision | `0c21a2226964181dee7ed28c9aa5aa0abdfe9765` — **verified at run time**, not assumed; printed by both preparers and confirmed in the gate output |
| Dataset | MIMIC-IV demo, `dataset_version` `3.1:0.1.0`, ETL `MEDS_transforms` 0.6.2, MEDS 0.4.1 |
| Task | `mortality/in_icu/first_24h`, `sha256:7925b00c16d2496f8a4c25068814a5fe6ad6904436eaa2f1a23c1f699727a2a2` |
| MIMIC-IV predicates | `sha256:f51853db0b44b216f81255a475893197018b569fec5fcc17e5505ffdbd5b301d` — located and asserted present by the test, **not consumed by any command** |
| Command | `uv run pytest -m real_data -rs` |

### Empty-history keys — read before any other number

A key whose subject has no qualifying event is scored at the learned intercept rather than excluded. On
this cohort that population is **empty**:

| Split | Keys | Positives | Subjects | **Empty-history keys** | Visits/key mean | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 60 | 8 | 47 | **0 (0.00%)** | 26.17 | 8 | 81 | 218 |
| tuning | 8 | 1 | 5 | **0 (0.00%)** | 9.50 | 9 | 17 | 21 |
| held_out | 6 | 0 | 4 | **0 (0.00%)** | 7.83 | 6 | 15 | 17 |

Median prior-record span per split: train 151 days, tuning 11, held_out 13.

Measured after the `MEDS_BIRTH` exclusion described below. Before it, every row carried one extra visit
(27.17 / 10.50 / 8.83) and the median history span read 23,311 days rather than 151.

No prediction on this cohort is uninformed, so no metric is diluted by intercept-only scores. This differs
from the synthetic fixture (10.0% / 7.7% / 7.7%), where the degenerate cohort was planted deliberately: the
task triggers on ICU admission, and every ICU patient in this extract has prior recorded events. The
guarantee still holds and is still what keeps key coverage exact; it simply has nothing to do here.

**That 0.00% is a property of the ICU trigger and does not transfer.** The full-data registry is made of
`"does code X occur within N days"` tasks with no ICU anchor, so nothing in those definitions guarantees
prior history the way this one does. Measured on the same demo cohort, varying only where the prediction
time falls relative to each subject's first *modelled* event (post marker-exclusion; the median modelled
record spans 256 days):

| Prediction time | Empty history, `<=` (default) | Empty history, `<` (strict) |
|---|---:|---:|
| = subject's first modelled event | **0/100 (0.0%)** | **100/100 (100.0%)** |
| 1 second after the first modelled event | 0/100 | 0/100 |
| uniform random point within the record | 0/100 | 0/100 |
| midpoint of the record | 0/100 | 0/100 |
| 1 day before the first modelled event | 100/100 | 100/100 |

Two conclusions. Empty history arises **only** for a prediction time preceding a subject's entire modelled
record, so an anchor-free task is safe as long as its trigger is some event of the subject. And the
`include_events_at_prediction_time` default is what makes that true: at a first-event-anchored trigger the
inclusive boundary yields 0% and the strict boundary yields **100%**. The strict setting reads as the
conservative one and is the dangerous one for this class of task. Pinned by
`tests/test_data_path.py::test_a_task_anchored_on_a_subjects_first_event_is_not_empty_by_default` so the
default cannot be flipped for apparent safety without that test failing.

The intercept fallback itself remains **unexercised on real data**: it is covered by unit tests and by the
synthetic fixture, and the demo produced no key that needed it.

**The empty stratum is a constant, and that is verified rather than inferred from the code.** Measured:
every empty-history key receives one identical probability equal to `sigmoid(b_output)` to within 1e-6,
against 59 distinct values among 60 non-empty keys, pinned by
`tests/test_commands.py::test_every_empty_history_key_receives_the_same_constant_prediction`. It follows
from two earlier decisions — static (`time = null`) events and the MEDS standard markers are both excluded
— which together leave **no** demographic path, so an empty history means no input at all rather than a
reduced model.

This matters for reading a counting-baseline decomposition across ports. A sibling port keeps its
demographic features flowing on an empty window, so its empty stratum is a demographics-only submodel and
can show genuine within-stratum discrimination from age and sex. Reporting one port's empty stratum as
though it were the other's kind would mislead, so both are stated. **The comparable figure across ports is
AUROC on non-empty keys**; the all-keys figure mixes in a block whose behaviour differs by port.

### Visit-count distribution — the evidence behind the 1-day granularity decision

With `visit_bucket_minutes = 1440` the training split averages **26.17 visits per key** (median 8, max 218),
spanning a median of **151 days** of prior record. Daily bucketing therefore yields a genuinely multi-visit sequence on real data rather than the
one-or-two-visit degenerate case that motivated the original question, and the long tail is exactly the
prior EHR history that the operator's rationale — keeping the EHR-versus-ICU importance distinction
visible — depends on. The decision to keep the 1-day default is supported by this measurement.

### Cohort funnel: extract to scored row

Every step from the MEDS extract down to a prediction, per split. Recorded here because it was otherwise
only reproducible by re-running against a demo build that does not survive a temp-directory rotation.

| Stage | train | tuning | held_out | total |
|---|---:|---:|---:|---:|
| Subjects in MEDS extract | 80 | 10 | 10 | **100** |
| Events | 899,463 | 76,169 | 46,923 | **1,022,555** |
| Subjects with >= 1 ICU admission | 80 | 10 | 10 | 100 |
| **ICU stays (candidate triggers)** | 113 | 15 | 12 | **140** |
| ACES gap-window exclusions | -53 | -7 | -6 | **-66** |
| **Task keys emitted by ACES** | **60** | **8** | **6** | **74** |
| Distinct subjects among keys | 47 | 5 | 4 | 56 |
| Positives | 8 | 1 | **0** | **9** |
| Code incidences fed to the model | 174,069 | 18,918 | 11,055 | 204,042 |

The largest attrition is ACES', not this port's: `mortality/in_icu/first_24h` discards 66 of 140 ICU stays
because the patient died, was discharged, or was re-admitted inside the 48-hour gap window. The headline
"100 subjects" is really **56 subjects contributing 74 prediction points**. Subject and key counts diverge
in both directions -- a subject can contribute several qualifying stays, and one held-out subject
contributes 3 of the 6 keys -- which is why coverage is checked on the `(subject_id, prediction_time)`
pair rather than on subject counts.

### Cohort scale and what it cannot support

The demo cohort is 100 subjects and 1,022,555 events, yielding **74 task keys with 9 positives**. The
evaluation split is **6 keys with 0 positives**, so AUC is undefined on `held_out` and the tuning split
carries a single positive. **This gate produces no performance result and none is claimed.** Its purpose,
per the implementation guide, is to expose pipeline, resource, serialisation and key-coverage failures at
real-data scale, which it did.

### Preprocessing on real data

| Quantity | Value |
|---|---|
| Events read | 1,022,555 |
| Static events excluded (`time = null`) | 100 |
| MEDS standard markers excluded (`MEDS_BIRTH`, `MEDS_DEATH`) | 131 |
| Out-of-vocabulary events dropped | 1,003 (0.107%) |
| Events retained | 938,528 |
| `(subject, time)` event groups | 96,666 |
| Distinct codes in dataset / vocabulary learned from train split | 7,523 / **7,085** |

The vocabulary binds to real MIMIC-IV MEDS codes across every clinical family — 2,300 `LAB`, 2,262
`MEDICATION`, 1,350 `DIAGNOSIS`, 457 `PROCEDURE`, 237 `DRG`, 157 `INFUSION_START`/`INFUSION_END`, plus
transfer, fluid-output and admission codes — at 9.18 codes per event group. This was checked rather than
assumed: a vocabulary that parsed but bound nothing would have produced the same exit status.

### Key coverage

6 requested `held_out` keys, 6 written, **0 missing, 0 extra**, keys unique, aligned by
`meds_evaluation.PredictionSchema`. Compared as key *sets*, not counts.

### Interpretability on real codes

Eq. (5) contributions were produced for every scored key. At smoke size (3 epochs) attention is
near-uniform — for an 18-visit patient, `alpha` 0.0551-0.0556 against a uniform 0.0556 — i.e. structurally
correct but not yet informative. Re-run at the official hyper-parameters (10 epochs) the same patient's
attention differentiates to `alpha = 0.0813`, about 1.46x uniform, with the largest contributions on
creatinine and chemistry panel codes. The mechanism works on real data; the magnitude is a function of how
long the model trained.

### Measurements

Both configurations were run on the same demo cohort. The smoke row is what the gate command executes; the
official row is the more useful basis for a full-data resource request.

| Configuration / command | Wall s | Peak tree PSS | Peak tree RSS | GPU allocated |
|---|---:|---:|---:|---:|
| smoke, `preprocess_data` | 0.20 | 1,140,069,376 | 1,143,742,464 | null (CPU stage) |
| smoke, `supervised_train` | 1.56 | 2,017,481,728 | 2,021,154,816 | 79,652,864 |
| smoke, `predict` | 0.11 | 2,076,791,808 | 2,080,464,896 | 76,577,280 |
| official defaults, `preprocess_data` | 0.22 | 598,925,312 | 602,435,584 | null (CPU stage) |
| official defaults, `supervised_train` (10 epochs, 1,122,050 parameters) | 1.57 | 1,965,470,720 | 1,969,143,808 | 178,265,600 |
| official defaults, `predict` | 0.11 | 2,024,526,848 | 2,028,199,936 | 85,279,232 |

Tokenising a million events takes 0.22 s and training the official architecture on this cohort takes 1.57 s,
so the demo is nowhere near a resource boundary. The resident floor of roughly 2 GB on the GPU stages is the
CUDA context and PyTorch runtime, not the model.

### Determinism on the real cohort

Environment verified clean first — no `CUBLAS_WORKSPACE_CONFIG`, no `CUDNN_DETERMINISTIC`, nothing set at
import — so these are default non-deterministic kernels. Three trials per device at the official
hyper-parameters: **CPU 0/3 pairs diverged, CUDA 0/3 pairs diverged, worst `0.000e+00`**. The CPU result is
the load-bearing one; the CUDA figure is context about this hardware at this regime (GB10, 74 keys, 10
epochs, embed 128) and is not a claim of GPU bit-reproducibility.

### The revision pin is verified rather than assumed

`provision_meds_dev` pins the MEDS-DEV revision only on the branch that clones from the URL:

    run([git, "clone", source or MEDS_DEV_URL, checkout], ...)
    if source is None:
        ref = os.environ.get("MEDS_DEV_REF", MEDS_DEV_PREDICATES_REF)
        run([git, "-C", checkout, "checkout", "--detach", ref], ...)

Supplying `MEDS_DEV_DIR` takes the other branch, so the checkout arrives at whatever the source's HEAD
happens to be and is never pinned. The assertion that follows,
`assert_meds_dev_predicates_capability`, tests that the predicates passthrough is *present*, so a newer
revision still carrying it passes — a capability check, not a revision check, despite the name. A gate run
against an unpinned MEDS-DEV is not the gate it reports being.

`tests/e2e.py::resolved_meds_dev_revision` closes that. It is anchored on **the provisioned clone**, which
is what is actually installed and executed, rather than on `MEDS_DEV_DIR`, which is only the source it was
copied from. That single anchor covers both provisioning paths; a `MEDS_DEV_DIR`-only check would be silent
on the URL path. Both preparers call it and print the SHA, so the revision appears in gate output whether or
not anyone reads the assertion. `tests/integration.py` is not edited and stays byte-identical across the
fleet.

`tests/test_meds_dev_pin.py` exercises the guard against real git checkouts rather than mocks, because a
guard that silently passes is the same class of defect as the check it replaces: a wrong HEAD is refused
with both revisions named, an explicit `MEDS_DEV_REF` override is honoured, a branch or tag is resolved
rather than string-compared, a non-git directory fails loudly instead of returning something plausible, and
with no override the expectation is `tests/integration.py`'s own pin rather than a local default.

**The guard is only as strong as its expected value.** Setting `MEDS_DEV_REF` to silence a failure rather
than to make a deliberate revision change would turn it into a rubber stamp. Nothing here can prevent that;
printing the resolved SHA into gate output is the mitigation, so the revision a run used is recoverable from
the output rather than from the assertion having passed. Every gate reported above ran at
`0c21a2226964181dee7ed28c9aa5aa0abdfe9765` with no override set.

### A defect found in the shared scaffold, not in this model

The first demo attempt failed after roughly eight minutes. `meds-dev-dataset` built the demo successfully
and `meds-dev-task` then exited 127 with `aces-cli: not found` and `expand_shards: not found`.

Cause, verified at the pinned ref rather than inferred: MEDS-DEV's `tasks/__main__.py` shells out to
`aces-cli` through `run_in_env(..., run_as_script=False)` **without passing `env`**, so the child inherits
this process's environment; invoking `.venv/bin/meds-dev-task` by absolute path never activates that
virtualenv, so its console scripts are absent from `PATH`. The dataset call site is *not* affected —
`datasets/__main__.py` wraps its build in `temp_env` and passes `env=env`, and `temp_env` prepends the
dataset venv's `bin` whenever the dataset ships a `requirements.txt`, which MIMIC-IV does.

Fixed in the model-owned `tests/e2e.py::prepare_mimic_demo` by passing `PATH` to the **task** invocation
only. Deliberately not applied to the dataset invocation: it would be harmless, since MEDS-DEV prepends its
own venv over anything passed, but it would encode a defect that does not exist. `tests/integration.py`
needed no change and remains byte-identical at
`sha256:cb5a2eccbafe8cb9f0c2c2329bd56c165596a54d3d2099cd9eb95d1c7753ed58`.

## Scaling measurement

Run at the **official model size** (embed 128, alpha/beta 128, batch 100, 10 epochs) on the real demo
cohort replicated 1x/4x/16x/64x with offset subject ids, so the event, code and visit distributions stay
authentic and only the scale changes. Replication holds the vocabulary fixed at 7,085; vocabulary growth is
treated separately below.

| Factor | Subjects | Events | Keys | `preprocess` s | `train` s | `predict` s | `preprocess` RSS | `train` RSS | GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | 100 | 1,022,555 | 74 | 0.25 | 1.62 | 0.09 | 1,656 MB | 1,918 MB | 169 MB |
| 4x | 400 | 4,090,220 | 296 | 0.57 | 0.65 | 0.13 | 4,674 MB | 3,518 MB | 168 MB |
| 16x | 1,600 | 16,360,880 | 1,184 | 1.63 | 1.47 | 0.25 | 10,181 MB | 5,537 MB | 228 MB |
| 64x | 6,400 | 65,443,520 | 4,736 | 6.47 | 5.56 | 0.81 | 28,534 MB | 10,042 MB | 325 MB |

The 1x `train` figure includes CUDA context creation; the 4x-64x rows are the ones to read for growth.

**Wall time is not a constraint.** 65.4M events tokenise in 6.5 s (~10M events/s) and 4,736 keys train for
10 epochs in 5.6 s. **GPU memory is not a constraint either**: 325 MB at 64x, because the model is
`|C| x 128` embeddings plus two 128-wide GRUs and the batch is bounded by `batch_size` visits, not by
cohort size.

### Host RAM is the binding constraint, and it is set by shard size rather than dataset size

The 28.5 GB at 64x looked alarming until it was attributed. Peak RSS is `O(largest shard)`, because both
`preprocess_data` and `build_sequences` read one shard at a time into memory. The replication above wrote a
single file per split, which is the worst case. Re-sharding the identical 16.4M-event dataset:

| Shards per split | Rows in largest shard | `preprocess` wall s | Peak RSS |
|---:|---:|---:|---:|
| 1 | 14,391,408 | 1.96 | 7,968 MB |
| 8 | 2,653,888 | 1.78 | 2,285 MB |
| 32 | 1,102,320 | 2.72 | **1,357 MB** |

Same data, same work, same wall time; a 5.9x memory reduction from sharding alone. The empirical rule is

> peak **process-tree** RSS ~ 1 GB baseline + ~0.5 KB per row of the largest shard

which held to within +/-20% across all three points above.

**A falsification recorded here in `2e36e49` is withdrawn: the comparison was invalid.** I recorded that
the rule was wrong by 3.1x on first contact with a real MIMIC-IV extract, from a `sacct` MaxRSS of
7.44 GiB against the rule's predicted 2.6 GB. Those two numbers do not measure the same quantity.
`src/meds_model_contract/manifest.py:73-88` sums `Rss` from `/proc/<pid>/smaps_rollup` across the process
tree, counting pages **mapped into the address space**; `sacct` MaxRSS under `jobacct_gather/cgroup`
reports the **cgroup's** charge, which includes page cache accumulated by a stage streaming a 19 GB
extract. All three fitted points came from that sampler. Comparing them against a cgroup figure was
apples-to-oranges, and this half of the conclusion is void **independently of what the true value is** —
that follows from my own code and needed no cluster access to establish.

**Provenance, which is the part worth keeping.** Both the 7.44 GiB and its replacement were *transcribed
to me by another session*. `/groups/...` is not reachable from this machine; I have read neither manifest.
I recorded the first as a measured falsification anyway, rewrote the rule on it, and ran two sweeps
hunting the cause of a gap that did not exist. My own provenance rule — mark every number measured, read,
or transcribed — would have caught it, and I failed to apply it to a number that arrived as bad news. An
unwelcome finding got a *lower* evidentiary bar than a welcome one would have, which is exactly backwards.

| Corrected figures — all `[T]` transcribed, none verified here | | |
|---|---|---|
| `preprocess_data` process-tree RSS | **1.55 GiB** `[T]` | what this rule models |
| `preprocess_data` cgroup MaxRSS | 7.44 GiB `[T]` | what the OOM killer enforces |
| gap between the two instruments | **4.8x** | the margin a `--mem` request must absorb |

**What the corrected number supports, and what it does not.** Against 3,206,908 rows the rule predicts
2.60 GB and the reported figure is 1.66 GB, so it **overpredicts the total by 1.56x, in the safe
direction**. But that total is flattered by a baseline term that happens to be about right. The implied
coefficient is ~207 B/row above a 1 GB baseline (~184 B/row above 1 GiB) against 500 B/row predicted, so
**the coefficient overpredicts by ~2.4x** — and ~200 B/row sits *below* the 384-526 B/row band the
replication fitted, not inside it. The sibling session's summary reads it as "comfortably inside" that
band; it is not, and the difference decides things, because *the rule is accurate on real data* and *the
rule is conservative on real data by an unmeasured factor* license different `--mem` choices. Real ICU
data is **cheaper** per row than the replicated demo, not equivalent to it.

So: **the variable is right, the direction is safe, and the coefficient is still not accurate.** Use it as
an upper bound, never as an estimate.

**Size `--mem` on the cgroup figure, not on this rule.** The rule models the algorithm's working set; the
OOM killer counts the cgroup, 4.8x larger here. Sizing `--mem` straight from the rule would have picked
~4 G for a job the scheduler measured at 7.44 GiB and killed it. That is the same practical danger
recorded before the retraction, reached by a different route and still real. `MEMORY=64G` remains
comfortable; nothing needs changing.

**What stands unchanged.** The shard-geometry result — 5.9x memory reduction from re-sharding the
identical dataset, at 1.4x wall-time variation across a 64x range — was measured here and is untouched.
So is *no cross-port resource figure is transferable*: the fleet spread on this one stage is an order of
magnitude (RETAIN 1.55 GiB to DuETT 28.86 GiB `[T]`). What I withdraw is the stronger claim I added under
the falsification — *no synthetic-fitted figure transfers to real data*. This one did transfer, in the
safe direction. The narrower statement the evidence supports: a synthetic-fitted **coefficient** transfers
only as a bound, because the generator cannot produce the data regime it is extrapolated into.

The "unresolved 2,178 B/row gap" is struck; there was no gap. The cardinality and string-width sweeps that
chased it were aimed at a phantom. Their results are retained — cardinality and string width account for
at most ~133 B/row, and event clustering moves it not at all — because they were measured here and remain
true, but they no longer explain anything that requires explaining.

**The number to check before submitting is still the row count of the largest shard, not the size of the
dataset.** If the extract has few, very large shards, the mitigation is re-sharding the input, not more
RAM.

### What this measurement does not cover

- **Vocabulary growth.** Replication holds `|C|` at 7,085. The embedding is `|C| x embed_size` float32, so
  a full extract with 100k codes costs 100,000 x 128 x 4 = **51 MB** of parameters, plus its Adadelta
  accumulators (2x) and gradient (1x) -- roughly 200 MB total, still not a constraint. `max_vocab_size`
  bounds it if needed but does not appear to be needed.
- **Visit-count tail.** The longest demo history is 218 visits. `max_visits` bounds the per-sequence cost
  if a full-data subject has an extreme record; unset by default and not obviously needed at these figures.
- **Key count at full scale.** The ten-task registry has not been read, so the number of prediction keys per
  task is unknown. `train` cost is close to linear in keys over the measured range (296 -> 4,736 keys is
  16x rows for 8.5x time), so this is the quantity that should drive the time request.

## Cluster operability

Changes made against a SLURM brief from another port on this cluster, each checked against this port
rather than adopted wholesale:

- **`supervised_train` now prints one line per epoch to stderr** — epoch, train loss, validation AUC, best
  epoch, epochs since best, optimiser steps. The loop was previously silent. Because each command writes
  its artifact and manifest only on completion, a multi-hour fit offered nothing to inspect from outside
  and a healthy run was indistinguishable from a hung one. This also makes per-epoch cost readable from a
  job log in the first minutes of a run, which is what a GPU time limit has to be sized against.
- **Every `slurm/config.sh` setting is `${VAR:-default}`**, so a cluster with different partition names or
  limits can override from the environment without a commit. `submit.sh` and `job.sbatch` are
  template-owned and unmodified.
- **`slurm-*.out` / `slurm-*.err` are gitignored.** Nothing sets `--output`, so SLURM writes job logs into
  the submit working directory, which is this worktree, and raw logs must not be committed.
- **Run output is excluded deny-by-default rather than by enumeration.** `RUN_ROOT` defaults to the
  relative path `runs` and `scripts/github-sync.sh` publishes from `runs/<id>/` by design, so the run
  directory cannot be excluded wholesale; the ignore rules are therefore load-bearing. The previous
  patterns named `logs/`, `artifacts/` and `checkpoints/` and left **`runs/<id>/command-args/` trackable**
  — written on every real submit, and containing this model's `external_meds_dir` and
  `external_labels_dir`, so a default-configured submit followed by `git add -A` would have committed the
  dataset and task paths of a real run. Now `runs/**` is excluded with an allowlist of exactly the three
  things `github-sync.sh` stages, so anything added under `runs/` later arrives ignored until it is named
  deliberately. `tests/test_run_output_is_not_committable.py` asserts both directions and derives the
  allowlist from the sync script itself so the two cannot drift; verified by mutation that a blanket
  `runs/` ignore fails the publication direction rather than passing quietly, and that the original
  enumerating rules fail the exclusion direction.

  The `git add -A` case asserts a **negative** property — no `command-args`, `artifacts`, `logs` or
  `checkpoints` staged, and no host dataset path present in staged *content* — rather than an exact staged
  set. The publishable files are trackable by design, so whether they appear depends only on whether they
  happen to be committed already; an exact-set assertion would encode this repository's commit history and
  would go red on a fresh checkout holding a completed-but-unpublished run. That is a false red of the
  kind usually resolved by loosening the assertion until it no longer means anything.

  This defect was present in **all five ports** of the pilot, each having inherited the same enumerating
  patterns, so it is a template defect rather than five independent mistakes and belongs upstream. The
  fix is recorded here because this port must not ship with the hole, not because downstream is where it
  should live.

  The invariant the test depends on is **`github-sync.sh`'s staged-path line, not that file**. The script
  is a per-model Copier render and differs across the pilot's ports in its branch name and commit message
  — five distinct hashes — while its `git add` line is character-identical in all of them. A port rendered
  for a different profile could legitimately stage different evidence, which is why the list is read from
  the script rather than restated. An earlier version of this note claimed the file itself was identical
  across ports; that was checked and is false.

- **The literal run path is a three-way coupling, and it is now asserted.** `.gitignore` excludes
  `runs/**`, `slurm/config.sh` defaults `RUN_ROOT` to `runs`, and `scripts/github-sync.sh` hardcodes
  `result_dir="runs/$run_id"` **without ever reading `RUN_ROOT`**. Those agree today by sharing a literal,
  not by construction. Overriding `RUN_ROOT` to another *relative* path puts run output back inside the
  checkout where these rules do not cover it and the `command-args` leak returns; overriding it to an
  absolute scratch path is leak-safe but makes the publisher search `./runs/` on a complete run, find
  nothing, and exit reporting "refusing to publish an incomplete run" — an error naming the run when the
  cause is the configuration. `tests/test_run_output_is_not_committable.py` asserts the three agree, so a
  change to any one of them is a failing test rather than a silent change in what can be committed.
  Mutation-checked: setting `RUN_ROOT` to `scratch` fails with "github-sync.sh publishes from 'runs' but
  submit.sh writes to 'scratch'; a complete run would fail to publish".

  **The anchor is enforced by git, not merely chosen.** `git add` refuses a path outside the repository —
  measured at exit 128 for both the absolute and the relative form, "is outside repository" — so
  publication can only ever stage paths inside the checkout, whatever `RUN_ROOT` is set to. These rules
  are therefore anchored on the one path that cannot move.

  **One residual hole that no ignore rule can close.** That refusal is not total: a symlink *inside* the
  checkout pointing at an off-checkout run is staged successfully, and its blob content is the target
  path. Staging `runs/<id>/spec.yaml -> /scratch/<id>/spec.yaml` therefore puts a host path into the
  index, and it is allowlisted precisely because a real `spec.yaml` there must be publishable. No
  `.gitignore` rule can separate the two cases — but **git's index mode does**, exactly: `120000` for a
  symlink against `100644` for a real file. So the property is asserted as an outcome rather than as an
  implementation detail: after the publisher's `git add`, no staged entry under `runs/` may have mode
  `120000`. That holds however a link arrives — `cp -s`, `rsync -l`, a helper the script shells out to, or
  a reimplementation of the publisher — where a source-text tripwire only covers today's script. Both are
  present; the mode assertion is the guard and the tripwire is the early warning. This matters because
  making the publisher follow a relocated `RUN_ROOT` by symlink is a plausible implementation of a fix
  currently under discussion upstream, so *copy, never link* is a constraint on that work.

  Only the symlinked-*file* case needs guarding. A symlinked *directory* is refused by git itself —
  `fatal: ... is beyond a symbolic link`, exit 128 — so it fails loudly and never reaches the index. Both
  cases were measured here rather than inferred from one another.

  **"Copy, never link" is not sufficient on its own**, which is why the guard is the mode assertion rather
  than the constraint. GNU `cp -R` *preserves* symlinks; only `-L` dereferences them. Measured: copying a
  `manifests/` directory whose `predict.yaml` is a link into scratch produces a link in the destination,
  and staging that gives mode `120000` with the host path as blob content — exit 0, no warning, alongside
  a real manifest at `100644`. So a run whose manifest happens to be a link into scratch, an ordinary
  cluster arrangement rather than a workaround, leaks **without anyone attempting the symlink route and
  with nothing wrong in the publisher's text**. A source-text tripwire passes on that case; the mode
  assertion fails it, and a test asserts exactly that discrimination rather than only passing on a clean
  fixture where it could not fail.

- **`RUN_ROOT` is anchored to the model directory** rather than to the caller's working directory. With
  the template's relative default, invoking `submit.sh` from the repository or worktree root writes
  `<worktree>/runs/` — a sibling of this model, which this model's `.gitignore` cannot reach. Measured
  before the fix: `git check-ignore runs/probe/command-args/x.args` from the worktree root reported it
  **trackable** while the in-model path was ignored, so the dataset paths in those files were committable
  with no misconfiguration at all — no override, no cluster, just the wrong working directory. A
  `.gitignore` at `models/<name>/` cannot cover a sibling path, so this is fixed where it originates.
  An override resolving anywhere else is refused, because **there is currently no `RUN_ROOT` that both
  stores output off-checkout and publishes**: off-checkout is leak-safe but the publisher aborts with
  "refusing to publish an incomplete run" against a run that completed, and in-checkout-but-elsewhere
  leaks. The default is the only working configuration — which is worth stating plainly because
  `DGX_AGENTIC_WORKFLOW.md:390` prescribes an off-checkout run root.

  **The anchor must also agree with the publisher's lookup, and that is the assertion that
  discriminates.** An anchor that makes `RUN_ROOT` absolute but lands anywhere other than what
  `result_dir="runs/$run_id"` resolves to fixes #9 by walking into #8 — a leak-free run that cannot be
  published. Five cases are asserted, not two: resolution from the model directory, resolution from the
  worktree root (corrected, not refused), that path being gitignored, an explicit `RUN_ROOT=runs` from the
  worktree root still being refused while the same string from the model directory is accepted, and the
  written run directory being **identical** to what the publisher would resolve. Mutation-checked: an
  anchor that is internally consistent but points at `scratch-runs` still passes several of the others and
  fails the publisher-equality case.

  The two guards do different jobs and both are present: anchoring makes the accidental route (invoking
  `submit.sh` from another directory) *correct* rather than refused, and the refusal catches a deliberate
  bad override. The guard's **message** is asserted, not merely its exit status, because the remedy here
  is counter-intuitive — the fix is to stop setting `RUN_ROOT`, not to set it somewhere better — and a
  guard that refuses correctly while misdirecting costs more than no guard. The test also asserts the
  message does not blame the working directory, since that route is anchored and therefore fine.
- **The PATH requirement is documented in `config.sh`.** Both `submit.sh` and `job.sbatch` call
  `meds-model` by bare name; without the virtualenv on PATH, `submit.sh` prints "No commands declared" on
  the login node — observed directly here while testing `--dry-run`. Since `--export=ALL` is passed,
  submit-time PATH is what the compute node inherits, so a submission made without it queues and then
  fails after the wait.

**Update the model's own worktree; never `git checkout` in the shared repository.** The first command
block sent for the full-data run said `git -C "$REPO" checkout <sha>` against
`/groups/.../Git/meds-models`. That repository is on `master` and owns five linked worktrees, one per
port, so checking out a commit there detaches the shared main working tree for no gain — this port's
worktree already exists and its branch is already at the target. The correct form is a fast-forward in the
port's own worktree:

    WT="$REPO/.claude/worktrees/01_19_retain"
    git -C "$WT" fetch --prune origin
    git -C "$WT" merge --ff-only origin/model/01_19_retain
    test -z "$(git -C "$WT" status --porcelain)"
    cd "$WT/models/01_19_retain"

It would not have corrupted the worktrees, which carry their own HEADs, but it mutates state other
sessions read. Caught by `meds-model-cluster` before execution; recorded here rather than only in that
exchange, because a runbook step that only exists in a message is a step the next person will not have.

One point from that brief needed no change: the vocabulary is built from the **data shards**, never from
`metadata/codes.parquet`. That is not a reaction to any report about that file's contents but a structural
choice — a metadata sidecar may enumerate more, fewer, or different codes than the events contain, so the
events are the authority. This port contains no reference to that file.

### A CUDA check belongs in a GPU job, not on the submit host

The runbook I first sent placed `python -c "import torch; print(torch.cuda.is_available())"` among the
login-node pre-flight checks. It returned `False` — not because the environment is wrong, but because
**the login node has no GPU at all**, so that assertion can only ever be false where it was placed. A
negative result there is indistinguishable from a real failure: the same defect as a check whose
instrument cannot observe what it asserts.

The x86_64 question it was meant to answer is better settled statically, and was: the lock carries linux
`x86_64` wheels for 28 packages with **zero aarch64-only** packages, so an aarch64-authored lock resolves
on an x86 cluster. `uv sync` then confirmed it in 122 s. CUDA availability is a separate question and must
be asserted **inside a GPU job** — in practice the first per-epoch line from `supervised_train` is that
evidence, since the model would not train without it.

### The documented full-data procedure sets a `RUN_ROOT` this port refuses

`DGX_AGENTIC_WORKFLOW.md` section 10 prescribes `MODEL_RUN_ROOT=/shared/meds-model-runs/<model>` and
exports it as `RUN_ROOT` before `./slurm/submit.sh`. That is an off-checkout run root, and this port's
guard **refuses it** — deliberately, because `scripts/github-sync.sh` hardcodes `result_dir="runs/$run_id"`
and never reads `RUN_ROOT`, so such a run would complete and then fail to publish with an error naming the
run rather than the configuration.

**Anyone following the documented procedure against this port will hit it**, and the refusal is the
correct behaviour here: it converts a publication failure discovered hours later into an explained refusal
at submit time. The message names both failure modes and states the remedy, which is to leave `RUN_ROOT`
unset.

### Which invariant this guard enforces

Stated explicitly because "the port has a `RUN_ROOT` guard" reads as agreement across the pilot and
conceals a split. Two invariants are in use, and this port implements the stricter one:

| Invariant | Off-tree absolute `RUN_ROOT` | In-tree elsewhere | Ports |
|---|---|---|---|
| **must resolve to exactly `<model>/runs`** | refused | refused | **this port**, BEHRT |
| in-tree only where `.gitignore` reaches | accepted (intended) | refused | TECO, DuETT, MEDS-EIC-AR |

Both are coherent and the choice is a real trade-off rather than a correctness question. This one is safe
by construction — run output can only land where the ignore rules reach, so no `cp -RL` and no index-mode
check is *required* for the leak to be closed (both are still present, because the mode check also covers
links carried in from a run directory built elsewhere). Its cost is that a full MIMIC-IV run's artifacts
land inside the git checkout, which for cluster work is a genuine disadvantage; the other convention keeps
them out and pays for it by needing dereferencing publication plus the mode assertion to stay safe.

The decisive factor for choosing it here is that **this port's** `scripts/github-sync.sh` — the 29-line
template version — hardcodes `result_dir="runs/$run_id"` and never reads `RUN_ROOT`. **This invariant is
the one compatible with the publisher this port actually ships.** Another port has since built a publisher
that resolves `RUN_ROOT`, dereferences on copy and skips the self-copy.

**The relaxation condition, stated as one conjunction so no part of it can be reached without the rest:**
this guard may be relaxed *only when* this port's `scripts/github-sync.sh` **both** resolves `RUN_ROOT`
**and** dereferences on copy, **and** the relaxation retains the refusal of an in-tree-elsewhere
`RUN_ROOT` (exit 4). Resolving alone is not sufficient — `git add` refuses a path outside the repository at
exit 128, so a resolve-only publisher fails at exactly the same point as the hardcode it replaced — and
dropping the in-tree refusal would be worse than either convention, since that is the case that leaks and
neither convention permits it.

Adopting that conclusion from another port's file, while this port's publisher still hardcodes the path,
would recreate exactly the failure the guard exists to prevent — a completed run that cannot be published.
So the reconciliation is coupled mechanically rather than left as a remembered intention:
`test_off_tree_acceptance_tracks_this_ports_publisher` refuses an off-tree `RUN_ROOT` while the publisher
hardcodes, and requires it to be *accepted* only once the publisher satisfies the full conjunction above.
The assertion flips itself; nobody has to remember. Its predicate is a conjunction for the same reason the
prose is: keying it on "mentions `RUN_ROOT`" would demand relaxing the guard for a resolve-only publisher
that still cannot publish — the trapdoor in executable form.

**Relaxing is not becoming permissive, and that distinction is the whole of it.** The target invariant is
three-way, not binary:

| `RUN_ROOT` | verdict | exit |
|---|---|---|
| `<model>/runs` (the default) | accepted | 0 |
| **anything else inside the work tree** | **refused, always** — would leak | **4** |
| fully outside the git work tree | refused today, **accepted** once the publisher both resolves `RUN_ROOT` and dereferences on copy | **5** |

The two refusals exit with **distinct** codes because they are different policies, and each message names
only the failure its own case causes — the in-tree message names `external_meds_dir`, the off-tree one
names the publisher aborting on a completed run. Conflating them is not hypothetical: "all five ports
refuse" was stated across this fleet while two different invariants were in use, and a survey reading exit
status alone could not have told them apart. The codes match BEHRT's so a fleet-wide survey is comparable
without reading anyone's prose.

The middle row is the dangerous one — run output inside the work tree at a path the ignore rules do not
cover, so the `command-args` files sit unignored — and no convention in the fleet permits it. Lifting a
`RUN_ROOT`-resolving publisher *without* keeping that refusal would convert today's loud failure (the
publisher aborts, having found nothing) into a silent one (it publishes happily from an unignored
location). `test_a_run_root_inside_the_work_tree_but_elsewhere_is_always_refused` asserts that row
independently of the publisher, so it cannot be lost in the relaxation.

An earlier version of this coupling was written as a single binary predicate — publisher-follows XOR
guard-refuses — which would have accepted a fully permissive guard as correctly coupled. That is recorded
because it is the same failure the rest of this section describes: a check that looks like it enforces the
property while permitting the one state nobody wants.

So the `DGX_AGENTIC_WORKFLOW.md` conflict is **not** a pilot-wide blockage: as written, section 10 is
correct under the second invariant and refused under this one. It cannot be correct for both
simultaneously, which makes it an operator decision about which convention the fleet adopts rather than a
documentation typo. It is recorded here so the full-data gate is not the moment it is discovered.

Documentation rather than five independent choices is still what propagated the underlying class — the
other instance being `docs/CLUSTER.md`'s `cp -r`, which preserves the symlinks the `100644` assertion
exists to catch.

### How the local gate is actually run

Recorded because the invocation is part of the evidence. The gate is run **unpiped, reading `$?`
directly**:

    uv run pytest -rs -q > "$LOG" 2>&1; STATUS=$?

Piping the command under test into `tail` or `head` reports the *pager's* exit status, not pytest's.
Measured here: `pytest -k <no match>` gives pipeline `0` where pytest exits `5`, and a collection error
gives pipeline `0` where pytest exits `2`. Reading the printed summary catches both, but that is vigilance
rather than a gate, and a pytest that dies without printing a summary would defeat it.

`${PIPESTATUS[0]}` is *not* the remedy used here, because it has a form that fails silently and is the
form a careful person writes. Measured:

| form | reported |
|---|---|
| `false \| tail -1` then `${PIPESTATUS[0]}` | `1`, correct |
| `out=$(false \| tail -1)` then `${PIPESTATUS[0]}` | **`0`, wrong** — it describes the assignment |
| `out=$(false \| tail -1; exit "${PIPESTATUS[0]}")` | `1`, correct |
| `false > "$f" 2>&1` then `$?` | `1`, correct |

The command substitution runs its pipeline in a subshell that no longer exists when the variable is read.
Writing to a file and reading `$?` has no positional dependency to get wrong, which is why it is the form
used: this class of error has recurred through the pilot despite being documented, so the durable fix is a
mechanical form rather than a rule to remember at the moment of writing.

### Which of this port's guards are proxies, and which way each can be wrong

Running the conditional audit against the *predicates* rather than the prose — the lesson from the
defective trigger above — makes a distinction worth stating: **a proxy that can pass while its condition
is false is dangerous; a proxy that can only fail while its condition is true is merely brittle.** The
defective trigger was the first kind. Classifying the rest of this port's file-reading assertions:

| Assertion | Kind | Backed by |
|---|---|---|
| allowlist parsed from `github-sync.sh`'s `git add` line | fails safe — a format change breaks parsing loudly | real `git add` staging test |
| `config.sh` contains the anchored `RUN_ROOT` and the dataset arguments | fails safe — reformatting fails it | dry-run from another working directory |
| publisher root and `RUN_ROOT` default name the same path | fails safe — a regex miss asserts explicitly | resolved-path equality against the publisher |
| **no `ln -s` in `github-sync.sh`** | **can pass while false** (`cp -s`, `rsync -l`) | the index-mode assertion, which is behavioural |
| **relaxation trigger: resolves `RUN_ROOT` and dereferences on copy** | **can pass while false** | **nothing — see below** |

Four of five are either fail-safe or backed by a behavioural test. The relaxation trigger is the one that
cannot be backed behaviourally **here**, because the publisher it describes does not exist in this port
yet: there is nothing to run. That is the honest limit of encoding a condition as an assertion. Two things
reduce it rather than remove it — the predicate is a conjunction rather than a proxy for one, and tripping
it produces a *demand that a human relax the guard*, carrying the full analysis in the failure message,
rather than relaxing anything automatically.

### A known flaky test that must not be re-run until green

`tests/test_operations.py::test_slurm_dry_run_is_non_mutating` snapshots `ROOT.rglob("*")` before and
after, and `ROOT` contains `runs/`. A concurrent job writing run output there changes the snapshot and
fails the test. It is template-owned and shared across the fleet, so it is deliberately **not** modified
here.

Recorded because the failure mode is the response rather than the test: a port that flakes, re-runs and
sees green learns "re-run until green", which is precisely how a real intermittent failure gets buried.
This is the same lesson as the CUDA determinism work in limitation 8, where a single passing run was shown
to be one sample rather than a measurement. If this test fails, check for a concurrent run under `runs/`
before re-running it, and record that as the cause rather than the green retry.

### Why the full-data results are not transcribed from the analysis session

The analysis session holds per-task AUROC/AUPRC for all ten tasks, measured from this run's
`artifacts/predict`. They are not copied into this report, and the reason is not caution about their
arithmetic.

That session runs a **report-reproduction check**: a model's prose report must match its artifact-derived
numbers at the precision the report prints. If this report were populated *from* that session's table, the
check would compare its own numbers against themselves and pass unconditionally. **The check has value
only where the two derivations are independent**, so transcribing would quietly convert a real
verification into a tautology — the same defect as every instrument in this report that could not observe
what it was cited for.

The artifacts live on the cluster (`/groups/...`), which is not reachable from this machine, so the
numbers will be written here once they can be derived from `predictions.parquet` and the run manifests
directly. Until then this row reads "not yet" rather than carrying figures whose provenance would be a
message.

## Patient separation: where the split boundary is, by path

Asked across the pilot on 2026-08-19. This port is `supervised`, so there is no pretrain stage. Answers
are from source, and the three files below are **byte-identical between HEAD and `54a8f31`, the commit the
full-data run executed** — so the boundary verified here is the boundary that produced that column.

| Question | Answer | Site |
|---|---|---|
| Does the fit read only `train`? | `train_rows` selects `split == TRAIN_SPLIT` | `commands.py:399` |
| Does epoch selection use only `tuning`? | `validation_rows` selects `split == TUNING_SPLIT` | `commands.py:400-402` |
| Is `held_out` touched by `predict` alone? | it is selected into neither array, so no gradient or epoch choice can see it | `commands.py:399-402` |
| Where is the vocabulary fit? | semi-join to training subjects **before** counting | `preprocess.py:92` |
| Which subjects does `min_code_count` count? | the train-only counts from that semi-join | `preprocess.py:104` |
| Quantile bins, normalisation constants? | **none exist** — `numeric_value` never appears in `src/retain/` | package-wide |
| Can anything extend past the prediction time? | window is bounded at `searchsorted(times, prediction)` | `windows.py:160` |

**`min_code_count` is a threshold on a train-only count.** Whichever value the operator picks, the subjects
counted for it are the training split and nothing else, so the open decision is about vocabulary size and
fidelity to the paper's 615 code groups — not about separation. It does not need revisiting on this axis.

**RETAIN has no transductive feature-selection surface at all**, which is worth stating plainly rather
than as an absence: the model embeds code indices and reads no values, so there is no quantile table, no
scaler, and no top-k selection to compute over the wrong population. The XGBoost defect the coordinator
described — `meds-tab-describe` reading all shards before a `max_included_codes` cut — has no analogue
here, because the only fitted-over-a-population object is the vocabulary and that one is semi-joined.

**One real gap, found by this question and now closed.** `_training_rows` is the sole place MEDS splits
become fit/validation roles, and **nothing pinned it**. Every claim above about `held_out` rested on
reading the function. Two tests now enforce it, asserting index *sets* rather than counts — a count check
still passes when a held-out row is swapped for a training one, which is the failure that matters:

- `test_held_out_rows_never_reach_the_fit_or_the_epoch_selection`
- `test_the_seeded_validation_fallback_carves_from_train_only` — the path a dataset with no `tuning`
  split takes, where a fallback could quietly reach for held-out rows to fill an empty validation set

Both were mutation-checked rather than trusted for passing: admitting `held_out` into the fit fails the
first, and carving the fallback from all rows instead of `train_rows` fails the second.

**Checked against BEHRT's control-blindness principle, and it found a second gap.** The direct question
— does either `_training_rows` test use a positive control running under a different configuration from
the negative assertion it supports? — is **no**. In both tests the control and the assertion are the same
invocation: `fit == {0, 4}` and `validation == {2}` prove selection happens at all, alongside the
disjointness assertion; `validation.size > 0` proves the fallback fires, alongside the subset assertion.
Nothing is established under one configuration and relied on under another.

Applying the principle **one level up** was not clean. Both tests hand-build the `assignments` Series that
production derives from `store.split_of(...)` (`commands.py:178`), so they are blind to split *derivation*
by construction. `split_of` is a **left join** (`windows.py:95`) and `_training_rows` indexes label rows
**positionally** against its result — so a subject listed twice in the split map expands the join, shifts
every subsequent split by one, and moves held-out rows into the fit **without any error anywhere**.

`read_split_map` already deduplicates (`preprocess.py:60`, `:71`), so the defect was not live. But that
guard was **unprotected**: deleting it passed all 114 tests. Its only control was a fixture containing no
duplicate to catch — a control blind to the condition it protects against, which is BEHRT's shape exactly,
found at the layer below the one the question pointed at. Closed with:

- `test_a_subject_listed_twice_in_the_split_map_collapses_to_one_row` — pins the dedup where it lives
- `test_split_of_refuses_a_split_map_that_would_misalign_rather_than_misassigning` — pins the consequence
- a length check in `split_of` (`windows.py:96-105`) that turns a silent misalignment into a refusal,
  since the positional contract between these two functions was real but nowhere stated

Both mutation-checked: dropping the dedup fails the first, dropping the length check fails the second.

The generalisable form, one step beyond the version received: **a test that constructs its own input is
blind to how that input is really produced.** Mutation-checking proves an assertion can fail; it does not
prove the assertion sits on the production path. Both of my `_training_rows` tests survive every mutation
to `_training_rows` and would have survived this leak untouched.

**The second shape applies here too, and it was live.** Asked whether a fixture keeping two split
sources consistent can tell which one production reads. `read_split_map` has exactly two:
`metadata/subject_splits.parquet` when present (`preprocess.py:55-60`), otherwise the shard directory
name (`:61-77`). `_write_dataset` writes **both**, always from one assignment. **Measured: forcing the
directory path unconditionally passed all 116 tests.** The sources never disagreed, so nothing could
observe which one governs.

Diverged them — subject 2's events under `data/held_out/` while the map says `train`, and the reverse for
subject 3:

| shard placement | map says | result |
|---|---|---|
| `data/train/` | `train` | `train` — positive control |
| `data/held_out/` | `train` | **`train`** — the map wins |
| `data/train/` | `held_out` | **`held_out`** — placement is not consulted |

**This port declares the split map authoritative; placement governs only in its absence** — the
opposite of a sibling port, where placement won and the map was ignored. Both choices are defensible and
neither had been written down, which made the divergence risk invisible at fleet level rather than merely
inside each port.

**The fleet-level check has since been run** `[T]` (cluster session, on the real extract; not reachable
from here): `364,627` subjects in the declared map, `364,627` by placement, in-map-only `0`,
in-placement-only `0`, **disagreements `0`**. So the cross-port comparison is like-for-like, and this port
and the placement-resolving one did train on the same cohort despite reading opposite sources.

**That agreement is the reason neither port could observe which source governs**, so it is now checked per
run rather than assumed. `_placement_agreement` (`preprocess.py:104-137`) compares the two inside the pass
that already reads every shard — no extra IO — and **refuses an extract that contradicts itself**, naming
the offending subjects. The check runs only when the directory names are split names: `data/shard_00/` is
a legal layout encoding no split, and comparing against it would refuse every such extract.

Each run now records the resolution in its manifest, which is what makes a cross-port check possible
without reading anyone's prose:

| field | meaning |
|---|---|
| `split_source` | `metadata/subject_splits.parquet` or `data/<split>/ placement` |
| `placement_encodes_splits` | whether the directory names carry split information at all |
| `placement_disagreements` | subjects the two sources assign differently; `0` required to proceed |

The honest status is the sibling session's: *it is fine now, and without this check nothing would tell you
when it stopped.*

**A third instance, one layer further down.** `preprocess` mirrors the input tree into the artifact
(`:169`), so the published payload carries `sequences/train/`, `sequences/held_out/` directories — and
`PatientStore` rglobs all of them, taking membership from the copied map (`windows.py:83`). Placement is
**inert inside the artifact**, which is correct and was asserted nowhere. Filtering shards by directory is
a plausible optimisation that would silently reassign subjects, so it is now pinned end to end through
`preprocess` into `PatientStore`.

**And the trapdoor, measured rather than reasoned about.** The fallback reads
`parts[0] if len(parts) > 1 else TRAIN_SPLIT`, so a **flat** `data/*.parquet` layout has no split
component and every subject defaults to `train`. Confirmed on a three-subject flat extract:
`[(1, 'train'), (2, 'train'), (3, 'train')]`, one distinct split. That is a vocabulary fitted over every
patient and a fit including held-out patients, reported by nothing. It is reachable only when the dataset
*also* ships no `subject_splits.parquet`, which is exactly why no fixture had ever produced it. **Now
refused outright** (`preprocess.py:71-83`), with the measurement in the comment.

Three tests, all mutation-checked: ignoring the map fails the two divergence tests, and dropping the
refusal fails the third.

**What is still not verified, and how to close it.** The above establishes that the code cannot fit on
held-out rows. It does not establish what the full-data run actually did, because `/groups/...` is
unreachable here and I have read no run manifest. `supervised_train` writes `train_split_rows`,
`tuning_split_rows` and `validation_source` into its manifest for exactly this purpose
(`commands.py:405-408`); reading those three fields against the split census confirms it empirically. That
is a concrete item for when artifact reach lands, and until then this column should be marked
code-verified rather than run-verified.

## Counting baseline: pre-registered before the number exists

`scripts/counting-baseline.py` asks how much of this model's discrimination is available from **counting
visits alone, reading no codes**. Method taken from `model/01_01_teco`'s script by reading it, not from its
description; only the unit differs — visits here, 15-minute bins there, coded events for the tabular
baseline. Built and exercised end to end; **not yet run against the full-data artifacts**, which are on the
cluster and unreachable from this machine.

The predictions below are recorded *now*, before the numbers exist, because a prediction made afterwards is
worth much less:

- **`presence` is likely degenerate on this bundle.** An empty history arises only when a prediction time
  precedes a subject's *entire* modelled record, and these tasks appear to anchor on an event of the
  subject. If so `presence` is identically 1, and the three strata collapse to one row.
- **If that happens, all-keys and non-empty will coincide — and that coincidence must not be read as
  agreement with TECO**, whose two figures do not coincide because its empty stratum is a
  demographics-only submodel rather than a constant. Identical numbers arising from an absent stratum and
  identical numbers arising from a matching stratum are different facts.
- **A degenerate `presence` makes this the strictest of the three instruments**, not the weakest: the count
  would then be pure measurement intensity with no cohort-separation component, measured without filtering
  and without needing a specially chosen dense task. If it still recovers a large fraction of the model,
  that is a stronger result on the counting question than either route currently available, and should be
  reported as such rather than as one more row. This framing is the analysis session's, recorded here
  rather than paraphrased later.

**A correction to my own claim, recorded because the corrected version is what will be written up.** I
described the existing counting evidence as *three* independent routes. It is **two**: the tabular baseline
on the bundle with empty keys filtered out, and TECO on a dense task where there are none to filter. The
degenerate-`presence` control is not a third route — it is what makes the second airtight, by ruling out a
cohort-separation component by construction rather than by measurement. Folding it in would count one task
twice. This port's result would be the genuine third, and it can still disagree; that is the reason to run
it rather than to assume it.

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicate-free confirmation, demo plan | **Approved** by the repository owner (`florian6973`), naming the gate and commit `ae298931d3b267de94ba353f5921b74000151ca6` | https://github.com/florian6973/meds-models/issues/3#issuecomment-5307590080, 2026-08-16 |
| Before full MIMIC-IV | Demo result, cohort/key coverage, measurements, full-run resources | **Requested, not yet granted** | Demo-evidence comment on issue #3 |

## Resource measurements

Local synthetic fixture, smoke hyper-parameters (`embed_size=16`, `alpha/beta_hidden=16`, `batch_size=8`,
`n_epochs=3`, `interpretation_top_k=3`), 60 subjects / 66 labelled keys / vocabulary of 17 codes / 3,842
parameters. Observations, not thresholds.

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| DGX GB10, `preprocess_data` | 0.058 | 71,398,400 | 112,173,056 | null (CPU stage) |
| DGX GB10, `supervised_train` | 1.424 | 1,601,548,288 | 1,859,211,264 | 75,762,688 allocated / 90,177,536 reserved |
| DGX GB10, `predict` | 0.116 | 1,609,777,152 | 1,867,702,272 | 75,610,624 allocated |

The ~1.6 GB resident floor on the two GPU stages is the CUDA context and the PyTorch runtime, not the
model. GPU figures come from `torch.cuda.max_memory_allocated`; `preprocess_data` records `null` because it
never touches an accelerator, which is the SPEC section 6 meaning of "not applicable".

Cohort shape at the local gate, per split:

| Split | Keys | Positives | Empty-history keys | Visits per key (mean / median / max) |
|---|---:|---:|---:|---|
| train | 40 | 10 | 4 (10.0%) | 6.35 / 6 / 13 |
| tuning | 13 | 4 | 1 (7.7%) | 9.31 / 11 / 13 |
| held_out | 13 | 5 | 1 (7.7%) | 7.46 / 7 / 13 |

No performance figure is claimed from this fixture: it is synthetic, the smoke configuration trains for 3
epochs, and 9% of its keys are scored at the intercept. The validation AUC recorded in the manifest
(0.639 at epoch 0 of 3) is evidence that the selection path executes, nothing more.

## Environment and provenance

- Model commit and dirty state: recorded in the issue #3 checkpoint comment for the exact reviewed
  revision; the working tree is committed and clean at that SHA on branch `model/01_19_retain`.
- Dataset/task identifiers: local gate uses the synthetic fixture in `tests/fixtures.py` (deterministic,
  seed 19, no real patient data). MEDS-DEV gate uses the same fixture through `dataset_name=retain-fixture`
  and `task_name=mortality_fixture`. The demo gate uses the MEDS-DEV MIMIC-IV demo
  (`dataset_version` `3.1:0.1.0`) with `mortality/in_icu/first_24h`; see *MIMIC-IV demo gate* above.
- Dataset, task, and predicate revisions: MEDS-DEV pinned at
  `0c21a2226964181dee7ed28c9aa5aa0abdfe9765` by `tests/integration.py`. No predicates file is consumed by
  any command; MEDS-DEV's MIMIC-IV `predicates.yaml`
  (`sha256:f51853db0b44b216f81255a475893197018b569fec5fcc17e5505ffdbd5b301d`) is located and asserted
  present by the demo test but read by nothing. Task
  `sha256:7925b00c16d2496f8a4c25068814a5fe6ad6904436eaa2f1a23c1f699727a2a2`.
- Full-data task-selection repository commit: not accessed. The bundle at `$FULL_MIMIC_WORK_DIR` has not
  been read and will not be before the full-run approval gate.
- Shared task-bundle identifier/path: not accessed.
- Task-bundle `tasks.yaml` SHA-256: not accessed.
- Cluster checkout commit and job IDs: no cluster job submitted.
- Structured-result branch/PR: none; `scripts/github-sync.sh` has not been run.
- Dependency/platform/hardware summary: Linux 6.17.0-1014-nvidia, aarch64, glibc 2.39; NVIDIA GB10;
  Python 3.12.13; torch 2.13.0+cu130 (CUDA available); polars 1.43.2; numpy 2.5.2; pyarrow via
  `meds-evaluation`; uv 0.12.0.

## Known limitations

1. **The visit definition is a modelling choice, not a fact about MEDS.** RETAIN attends over visits;
   MEDS has no visit concept that can be recovered without dataset knowledge. The default calendar-day
   bucket is defensible as the closest generic analogue of an admission, but on a 24-hour ICU task most of
   the sequence length comes from prior history rather than from the stay itself. `visit_bucket_minutes=60`
   or `0` are one argument away, and the visit-count distribution is reported at every gate so the choice
   can be revisited with evidence rather than by assertion.
2. **Empty-history keys are scored at the intercept, and that path is unexercised on real data.** The
   policy is deliberate, so that key coverage is exact and no exclusion policy is needed, but any metric
   computed over those keys measures the base rate. On the MIMIC-IV demo the population is empty (0.00% in
   all three splits) **because the task anchors on ICU admission** — a property of that task, not of this
   port. The full-data registry has no such anchor. Measurement above shows empty history arises only for
   a prediction time preceding a subject's entire modelled record, and that the inclusive
   prediction-time boundary is what keeps a first-event-anchored trigger at 0% rather than 100%. The
   fallback is covered by unit tests and by the synthetic fixture only; the first real-data exercise of it
   would be a full-data run, and the per-split count must be read before any metric from such a run.
3. **The closed vocabulary silently drops codes unseen in training.** The count is recorded
   (`out_of_vocabulary_events`), but a held-out cohort with substantially different coding practice would
   lose signal without failing anything.
4. **No performance result exists.** The demo cohort yields 74 task keys with 9 positives, and its
   evaluation split is 6 keys with **0 positives**, so AUC is undefined there and the tuning split carries
   a single positive. The demo is a pipeline, resource and key-coverage test, which is what the
   implementation guide asks of it; it is not a scientific result and none is claimed from it. The
   validation AUC of 1.0 recorded in the official-hyper-parameter manifest is computed over 8 rows with one
   positive and means nothing beyond "the selection path executed".
5. **The paper's appendices were not available** in the arXiv record used. Hyper-parameter defaults come
   from the official code, which is authoritative for those values, but the appendix's tuning ranges could
   not be cross-checked.
6. **`use_time` defaults to false**, matching upstream, but MEDS visit gaps are far more irregular than the
   admission sequences RETAIN was published on. Enabling it may be the better scientific choice for
   MIMIC-IV; it is one argument away and is flagged as an open question on issue #3.
7. **Single-GPU, single-process.** No distributed training, no gradient accumulation, no mixed precision.
   Whether that is sufficient for full MIMIC-IV is a question for the full-run resource review; the demo
   gate is the first evidence that would inform it.
8. **A seeded run is reproducible on a fixed device, but not across devices.** Both halves were measured
   rather than reasoned about.
   - *Fixed device, the control:* two full profile runs at the same seed produce **bit-identical**
     predictions, on CPU and again on CUDA, compared exactly rather than within a tolerance
     (`tests/test_device.py::test_two_runs_with_the_same_seed_are_bit_identical`). A companion test shows
     two *different* seeds diverge, so the seed is demonstrably wired in rather than merely unused.
     `torch.manual_seed` runs before the model is constructed; seeding after construction would leave
     weight initialisation drawing from an unseeded global generator, which is invisible in a
     cross-device comparison and is what the control exists to catch.
   - *Across devices:* CUDA draws dropout masks from its own generator, so the same `seed` yields
     different masks on CPU and on CUDA and therefore a different fit. Weight initialisation is
     byte-identical across devices (`reset_parameters` runs before the move), and with
     `keep_prob_emb=keep_prob_context=1.0` the two devices agree to `rtol=1e-4`; with RETAIN's default 0.5
     keep-probabilities the trained models differ by about 1% in predicted probability on the fixture.

   This is documented PyTorch behaviour rather than a defect here, but it has a benchmark consequence:
   **model comparisons must hold the device fixed**, and a result reproduced on different hardware will
   not match bit for bit. The forward pass itself is device-portable — a CUDA-trained checkpoint scored on
   CPU reproduces its CUDA scores to `rtol=1e-4`.

   *How far the CUDA exactness observation extends.* It was measured at every scale reachable without real
   data and held at all of them:

   Each row below states its trial count and divergence fraction; single-trial cells are not
   measurements and are not reported as such. Every run was made with a verified-clean environment (no
   `CUBLAS_WORKSPACE_CONFIG`, no `use_deterministic_algorithms`, nothing set at import), so these are
   default non-deterministic kernels rather than a suppressed baseline.

   | Configuration | Trials | CUDA diverged | CPU diverged | Worst |
   |---|---:|---:|---:|---|
   | smoke: embed 16, hidden 16, batch 8, 3 epochs, 66 keys | 5 | 0/10 pairs | 0/10 pairs | `0.000e+00` |
   | 219 keys / 200 subjects, official defaults, 8 epochs | 5 | 0/10 pairs | 0/10 pairs | `0.000e+00` |
   | 1,637 keys / 1,500 subjects, official defaults, 25 epochs | 5 | 0/10 pairs | 0/10 pairs | `0.000e+00` |
   | **MIMIC-IV demo cohort, official defaults, 10 epochs** | 3 | 0/3 pairs | 0/3 pairs | `0.000e+00` |
   | single-trial sweep: widths 16-256, epochs 3-50, 66-1,637 keys (10 configurations) | 1 each | 0/10 | 0/10 | `0.000e+00` |

   Pooling the **four** multi-trial conditions gives 14 independent comparisons with zero divergences.
   That **excludes a divergence rate comparable to the sibling ports** (`P(0/14 | rate 2/3) = 2.1e-07`)
   but leaves the 95% upper bound on a rare divergence at **19.3%** — a 5% rate would still produce this
   result 49% of the time. (This figure read 12 comparisons and 22% until the demo-cohort condition was
   added to the table above and the pooled statistic was not recomputed with it; corrected here.) The defensible claim is therefore *"RETAIN does not diverge at a rate comparable to
   the sibling ports at the conditions reachable here"*, **not** *"RETAIN is deterministic on CUDA"*.

   The 1,637-key row bears on the atomics question specifically: many rows reducing into a small
   vocabulary is the worst case for collision in `embedding_bag` backward, whose backward is an index-add
   over repeated indices. It still reproduced exactly, which makes this port a counter-case to the rule
   that a scatter/index-add backward implies nondeterminism. One possible explanation — that PyTorch uses
   a sort-based deterministic path there rather than raw atomics — is a **hypothesis that has not been
   verified here** and should be checked against the kernel rather than inferred from this result.

   Even so, **the CUDA test asserts a tolerance of `1e-5` rather than equality.** Exactness on an
   accelerator is a property of the kernels a model happens to use and the scale it runs at, not a
   guarantee: a sibling port measured a one-ULP (`5.96e-08`) disagreement in its own kernels under the
   same control. Asserting equality would promise something the hardware does not and would become an
   intermittent failure at full-data scale, where the first red run would be misread as a defect. The
   threshold sits well above a ULP and below the smallest real determinism defect observed in this port
   (`9.19e-04`, limitation 9).

   **This threshold must not be copied to another port.** It was proposed as a shared constant and that
   proposal is withdrawn: a sibling measured `1.836e-05` of genuine same-device CUDA noise, which `1e-5`
   would flag as a defect, and another measured `6.5e-3` cross-device at its published architecture. The
   generalisable rule is not a number: *a tolerance must be justified by that port's own worst measured
   divergence and must state the regime — architecture, epochs, cohort size — it was measured in.*
   More seriously, across the fleet the noise and defect populations **overlap**: a sibling's worst
   hardware noise (`3.937e-03`) exceeds the smallest real defect found in this project (`9.19e-04`) by
   4.3x. For a port in that regime no threshold separates them at all. **The CPU exact-equality control
   is the only determinism test here that carries a guarantee, and a passing CUDA tolerance is not
   evidence that a pipeline is deterministic.**

   The benchmark-design consequence, which applies whether or not this port's kernels stay exact:
   **a GPU run cannot be assumed reproducible bit-for-bit from its recorded seed alone, so cross-run and
   cross-port comparisons at the demo and full-data gates need a stated tolerance rather than equality.**
9. **Reproducibility depends on the task index's content, not its row order — by construction, after a
   fix.** The MEDS label schema gives row order no meaning and ACES writes the index through operations
   with no ordering guarantee, so the same task can be materialised with identical content in a different
   order. That difference used to reach the fit here: `len_argsort` resolved equal-length ties by row
   position, so a permuted index produced different minibatches and a different model at the same seed
   (measured: up to 9.19e-04 in predicted probability on the fixture). `read_labels` now imposes a
   canonical key order and `_length_sorted` breaks ties on `(subject_id, prediction_time)`; the same
   measurement now reads 0. Note that rebuilding a fixture twice does *not* detect this — a fixture that
   writes canonically reproduces its own layout and the check passes while the defect remains — so the
   regression test permutes row order directly.
10. **Resource behaviour is measured to 65M events, not to full-data scale.** The scaling section above
    replicates the real demo cohort to 64x. Wall time and GPU memory are comfortable there; host RAM is the
    binding constraint and is governed by the largest input shard rather than by dataset size. Two things
    remain unmeasured: vocabulary growth (replication holds `|C|` fixed, though the arithmetic puts a
    100k-code embedding at ~51 MB of parameters) and the number of prediction keys per task in the ten-task
    registry, which has not been read. The key count is what should drive the time request.
