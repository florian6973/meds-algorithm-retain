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

## DAG choice

- Selected DAG: `supervised`
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

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | 61 passed, 2 skipped, 0 failed. **0 `model_stub` skips** | Real three-command DAG on the synthetic fixture. 66/66 requested keys covered exactly, keys unique, prediction schema aligned by `meds_evaluation.PredictionSchema` |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | 1 passed, 53 deselected, 0 skipped, 0 failed | `meds-dev-model model=retain mode=full dataset_type=supervised` in MEDS-DEV's own isolated venv (torch 2.13.0 and `retain 0.1.0` installed there from `requirements.txt`). 13/13 `held_out` keys covered exactly |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | not run | Blocked pending explicit human approval on issue #3 |
| Full MIMIC-IV | not run | not run | Blocked pending a separate human approval after the demo gate |

The two skips in the local run are `tests/test_meds_dev_e2e.py` and `tests/test_mimic_demo_e2e.py`, both
reported by `-rs` as `select -m meds_dev or -m real_data, or set MEDS_DEV_DIR`. They are the marker-gated
integration tiers, not stub skips; the `model_stub` skip guards are unreachable because `IS_STUB` is
`False`. On a host with no CUDA device, five tests in `tests/test_device.py` add a fifth accounted-for skip
reason, `no CUDA device on this host`; on this host they run.

**Both reported gates exercised the GPU.** `device` defaults to `auto`, so `supervised_train` and `predict`
resolved to CUDA in the local and MEDS-DEV runs, and their manifests carry real
`torch.cuda.max_memory_allocated` byte counts rather than nulls. The demo and full runs will therefore not
be the first time this port touches an accelerator. `tests/test_device.py` additionally pins the whole DAG
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
- `tests/test_device.py` (9 tests, 5 CUDA-gated) -- the same-seed same-device determinism control on
  both devices and its different-seeds counterpart, the whole DAG on CPU and on CUDA, a CUDA checkpoint
  scored on CPU, device-independent weight initialisation, CPU/CUDA numerical agreement with dropout
  disabled, and that the pinned `bias_hh` survives a device move and GPU training.
- `tests/test_commands.py` (19 tests) -- exact key coverage including empty histories, split restriction,
  the Eq. (5) contributions summing back to the score they explain, artifact type/format rejection,
  predicate rejection on all three commands, mistyped-argument rejection, the preprocess-only vocabulary
  guard, atomic non-publication on failure, overwrite protection, gradient flow to every trainable
  parameter, and that training separates a synthetic separable signal.

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicate-free confirmation, demo plan | **Requested, not yet granted** | Checkpoint comment on issue #3, https://github.com/florian6973/meds-models/issues/3 |
| Before full MIMIC-IV | Demo result, cohort/key coverage, measurements, full-run resources | Not requested; blocked on the gate above | -- |

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
  and `task_name=mortality_fixture`.
- Dataset, task, and predicate revisions: MEDS-DEV pinned at
  `0c21a2226964181dee7ed28c9aa5aa0abdfe9765` by `tests/integration.py`. No predicates file is consumed by
  any command. The MIMIC-IV demo would use MEDS-DEV's `mortality/in_icu/first_24h` at that same revision.
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
2. **Empty-history keys are scored at the intercept.** This is deliberate, so that key coverage is exact
   and no exclusion policy is needed, but any metric computed over those keys is measuring the base rate.
   The count is reported per split at every gate.
3. **The closed vocabulary silently drops codes unseen in training.** The count is recorded
   (`out_of_vocabulary_events`), but a held-out cohort with substantially different coding practice would
   lose signal without failing anything.
4. **No performance result exists yet.** Only the synthetic fixture has been run. The demo gate is not a
   scientific result either, by the implementation guide's own instruction.
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
   - *Fixed device, the control:* two full DAG runs at the same seed produce **bit-identical**
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
   CPU reproduces its CUDA scores to `rtol=1e-4`. Determinism is verified at smoke scale (3 epochs, batch
   8, 66 keys); CUDA kernels that reduce with atomics can be nondeterministic at larger scale, so the
   control should be re-read at demo and full-data scale rather than assumed to carry over.
