# Implementation report: RETAIN

This report describes the `retain-meds` package after its conversion from the earlier `meds-model-retain`
layout to the MEDS algorithm template. The model code is unchanged; the contract around it is new. The
report of the earlier layout, with its MIMIC-IV demo, scaling and cluster evidence, is kept as
[`docs/LEGACY_IMPLEMENTATION_REPORT.md`](docs/LEGACY_IMPLEMENTATION_REPORT.md). That evidence was produced
by the earlier command layer and has not been rerun on this package.

## Source

- Paper/repository: https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html and https://github.com/mp2893/retain@9fd39c46e44b3fdae3a667a47fbdd5ffbd87bff5
- Exact revision/version:
  - Paper: Choi, Bahadori, Kulas, Schuetz, Stewart, Sun, *RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism*, NIPS 2016, pp. 3504-3512 (arXiv `1608.05745v4`).
  - Official code: `github.com/mp2893/retain` @ `9fd39c46e44b3fdae3a667a47fbdd5ffbd87bff5` (BSD-3-Clause).
    Files read in full: `retain.py`, `test_retain.py`, `process_mimic.py`, `README.md`.
- The official code is Theano 0.8 on Python 2.7 and is not installable today, so it cannot be a pinned
  dependency. The model is reimplemented in PyTorch and checked against a NumPy transcription of the
  official equations (`tests/test_model_fidelity.py`). No upstream code is copied.
- Template: `McDermottHealthAI/MEDS_algorithm_template` @ `731e759` (`v0.4.0a2-4-g731e759`), core
  `meds-algorithm==0.4.0a5`.

## Scientific assumptions and decisions

- Mode: confirmation required (default).
- Unresolved blocking questions: none. The conversion changes no scientific choice; every decision below
  was made for the earlier layout and is carried over unchanged.
- Provisional choices still requiring review before a faithful-reproduction claim: the MEDS adaptation
  (rows 1 to 4). The official code starts from tokenised visits and never decides what a visit is.

| Missing/conflicting detail | Sources consulted | Options and scientific consequences | Choice/rationale and status | User decision or source resolution | Affected code/tests/results |
|---|---|---|---|---|---|
| What a visit is in MEDS | paper section 4.1, `process_mimic.py` (admissions) | per timestamp, per fixed time bucket, per admission (needs predicates) | events bucketed on a `visit_bucket_minutes` grid, default one calendar day; provisional | carried over from the earlier layout | `_windows.build_sequences`, `tests/test_data_path.py` |
| Whether an event at the prediction time is history | MEDS-DEV task definitions (`index_timestamp: end`, `end_inclusive: True`) | include (`<=`) or exclude (`<`) | included by default, configurable; provisional | carried over | `include_events_at_prediction_time`, `tests/test_data_path.py` |
| Vocabulary control without ICD groupers | paper section 4.1, upstream README | dataset groupers (need dataset knowledge) or frequency filters | `min_code_count` and `max_vocab_size`, fitted on the training split; provisional | carried over | `_preprocess._build_vocabulary` |
| Keys with no history | none upstream (cohort is pre-filtered) | drop the key, or score it | scored with the learned intercept so every requested key is answered; provisional | carried over | `_training.collate`, `tests/test_model_fidelity.py` |
| `* 0.5` scaling of GRU outputs | `retain.py`, `test_retain.py`, paper | keep (code) or drop (paper) | kept; resolved from source | present in both upstream files | `_model.forward` |
| Second GRU bias of `nn.GRU` | `gru_layer` in `retain.py` | keep (differs from upstream) or pin to zero | pinned to zero and not optimised; resolved from source | upstream cell has one bias | `_model.py`, `tests/test_model_fidelity.py` |

## Profile choices

### supervised

- Lifecycle justification: RETAIN has no task-independent stage. `train_RETAIN` initialises the embedding,
  both attention GRUs and the risk head together and optimises them against the labelled endpoint in one
  loop; there is no pretraining objective and no frozen encoder. The steps are `preprocess_data`
  (task-independent tokenisation, no labels read), `train_supervised` (the whole model, fitted to one
  task) and `predict_supervised` (scoring a task index).
- Implemented profile methods, algorithm-owned configuration and payload formats: `RETAIN.preprocess_data`,
  `RETAIN.train_supervised`, `RETAIN.predict_supervised` in `src/retain_meds/retain_meds.py`; frozen
  `RETAINConfig` in `src/retain_meds/config.py`; payloads `retain-patients/v1`,
  `retain-supervised-model/v1`, `retain-predictions/v1`. Core `NNConfig` is not reused because RETAIN has
  two dropout rates, four separate L2 penalties and a default optimiser without a learning rate.
- Frozen/fitted parameters and permitted label access: the vocabulary is fitted on the MEDS training split
  in `preprocess_data`, which reads no labels. `train_supervised` reads the `train` labels (parameters) and
  the `tuning` labels (epoch selection) through `request.labels.read`, and refuses labels whose subjects
  are outside the matching MEDS split. `predict_supervised` reads the index only; the runner attaches
  ground truth after scoring.
- Local and MEDS-DEV evidence for this profile: see *Verification*.

## Source-element ledger

| Source element | Status | Implementation evidence | Justification/deviation |
|---|---|---|---|
| Step 1, `v_j = W_emb x_j`, linear, no bias, no activation (`emb = T.dot(x, W_emb)`) | ported | `_model.py` `Retain.forward`, `F.embedding_bag(..., mode="sum")` | `x` is binary, so summing the named rows of `W_emb` is the same product without materialising a `|C|`-wide one-hot |
| Step 2, `RNN_alpha` over reversed visits, `e_j = w_alpha g_j + b_alpha`, `alpha = softmax(e)` | ported | `_model.py`, `gru_alpha` + `w_alpha` + masked `softmax(dim=1)` | softmax normalises over real visits only; padded positions are masked to `-inf` |
| Step 3, `RNN_beta` over reversed visits, `beta_j = tanh(W_beta h_j + b_beta)` | ported | `_model.py`, `gru_beta` + `W_beta` + `tanh` | |
| Step 4, `c_i = sum_j alpha_j beta_j * v_j` | ported | `_model.py`, `(alpha.unsqueeze(-1) * beta * visits).sum(dim=1)` | |
| Step 5, `y = sigmoid(w_output c + b_output)` | ported | `_model.py`, `w_output` | binary case, as in `build_model`; the paper's softmax reduces to this for `s = 1` |
| GRU cell of `gru_layer` (`r`, `z`, `n` with the reset gate on the whole recurrent term) | ported | `nn.GRU`, verified against a NumPy transcription in `tests/test_model_fidelity.py` | `nn.GRU` is gate-for-gate identical; `bias_hh` pinned to zero, see deviation 5 |
| Reverse time ordering, then hidden states re-reversed (`gru_layer(...)[::-1]`) | ported | `_model.reverse_padded`, applied to the input and again to both hidden sequences | per-row reversal, so padding never enters a hidden state |
| `* 0.5` scaling of both GRU outputs | ported | `_model.py`, `alpha_states * 0.5`, `beta_states * 0.5` | in `retain.py` *and* `test_retain.py`; not in the paper. Kept, and called out as deviation 3 |
| Dropout on the embedding, feeding both the GRUs and the context sum (`keep_prob_emb`) | ported | `_model.embedding_dropout` applied before the split | `build_model` rebinds `emb`, so the dropped tensor is used in both places |
| Dropout on the context vector (`keep_prob_context`) | ported | `_model.context_dropout` | |
| Loss: `-(y log(y_hat + eps) + (1-y) log(1 - y_hat + eps))`, mean over the batch | ported | `_training._cross_entropy` | written out rather than `binary_cross_entropy_with_logits`, because `log_eps` sits inside both logarithms upstream |
| L2 on `w_output`, `w_alpha`, `W_beta`, `W_emb`, weights only | ported | `_model.regularization` | added to the cost, not folded into optimizer weight decay, matching `build_model` |
| Adadelta solver (rho 0.95, eps 1e-6, unit learning rate) | ported | `_training._optimizer` | the Theano block is exactly `torch.optim.Adadelta(lr=1.0, rho=0.95, eps=1e-6)` |
| Adam solver (`lr=0.0002`, `b1=0.1`, `b2=0.001`) | ported | `_training._optimizer` | upstream parameterises `b` as `1 - beta`, so this is the usual `(0.9, 0.999)`; its `fix1`/`fix2` are standard bias correction |
| Weight init `uniform(-0.1, 0.1)`, biases zero (`get_random_weight`, `init_params`) | ported | `_model.reset_parameters` | applied to the GRU weights too, replacing PyTorch's default init |
| Length-sorted minibatches (`len_argsort`) | ported | `_training._length_sorted` | |
| Reshuffled batch *order* each epoch (`random.sample(range(n_batches), n_batches)`) | ported | `_training.train`, seeded `random.Random` | |
| Model selection on best validation AUC per epoch | ported | `_training.train`, `TrainingHistory.best_epoch` | records `selected_by` so a run with no usable validation signal cannot silently claim a selection |
| Batch size 100, 10 epochs, embed/alpha/beta 128, L2 1e-3, keep-probs 0.5, `log_eps` 1e-8 | ported | `config.RETAINConfig` defaults | every default is the official `retain.py` default |
| Optional visit durations (`--time_file`, `--use_log_time`), concatenated to the GRU inputs only | ported | `config.use_time`, `_windows.visit_gap_days`, `_model.forward` | default `false`, which is the upstream default (no `--time_file`) |
| Eq. (5) per-code contributions (`test_retain.py`) | ported | `_interpret.write_contributions`, verified in `tests/test_primary_behavior.py::test_the_model_attributes_its_score_to_the_signal_code` | off by default (`interpretation_top_k=0`); NDJSON, not parquet, see *Intermediate artifacts* |
| Prediction at every timestep (`attentionStep` scanned over all prefixes) | adapted | `_model.forward` computes the final timestep only | the training graph reads `y_hat[lengths-1]`, so for sequence classification only `c_T` is used; `test_retain.py` computes exactly that. Same value, `O(T)` instead of `O(T^2)` |
| Theano 0.8 / Python 2.7 runtime | adapted | PyTorch `>= 2.2` | the original stack is not installable today; the cell and optimiser map exactly, and the fidelity test pins the equivalence |
| `sklearn.roc_auc_score` | adapted | `_training.roc_auc`, rank-based with tie correction | avoids a scikit-learn runtime dependency for one function; tested against the pairwise definition |
| `process_mimic.py` visit construction (admissions -> ICD-9 code sets) | adapted | `_preprocess.py` + `_windows.py` | MEDS has no admission concept without dataset predicates; see *Inputs* |
| 20/10/70 or `.train`/`.valid`/`.test` file split (`load_data`, `load_data_simple`) | adapted | `retain_meds.RETAIN.train_supervised` | the task's `train` labels fit and its `tuning` labels select the epoch; the seeded fallback only fires when the task supplies no `tuning` labels |
| 3-digit ICD-9 truncation / medical groupers (README, paper §4.1) | omitted | -- | both are dataset-specific vocabulary knowledge. Generic frequency filters (`min_code_count`, `max_vocab_size`) provide the same dimensionality control without encoding ICD semantics in a predicate-free port |
| Demographics and outcome markers excluded from visits (paper §4.1 "diagnosis, medication and procedure codes"; `process_mimic.py` builds sequences from `DIAGNOSES_ICD` alone and reads `PATIENTS.csv` only for the label) | ported | `_preprocess.MEDS_STANDARD_MARKERS`, `config.exclude_meds_standard_markers` | Drops `MEDS_BIRTH` and `MEDS_DEATH`. These are MEDS-standard codes, not dataset predicates, so naming them is not dataset knowledge |
| Non-binary input values (paper Eq. 5 remark) | omitted | -- | the official code is binary-only (`xvec[subseq] = 1.`); MEDS `numeric_value` is ignored |
| External pre-trained code embeddings (`--embed_file`, `--embed_finetune`) | omitted | -- | requires an external Med2Vec-style artefact that MEDS-DEV does not supply. `W_emb` is always randomly initialised and fine-tuned, which is the documented default |
| Re-training from a saved model (`--model_file`) | omitted | -- | no contract command corresponds to resuming a fit; a rerun is a fresh `train_supervised` |

## Inputs and predicates

- MEDS assumptions: only `subject_id`, `time` and `code` from `data/**/*.parquet`, plus
  `metadata/subject_splits.parquet` (or `data/<split>/` placement when it is absent). Static events
  (`time` null) and `numeric_value` are not used. `MEDS_BIRTH` and `MEDS_DEATH` are excluded by default.
- Required predicate names: none. RETAIN's input is a sequence of medical codes, which the MEDS `code`
  column already is. `external_predicates_file` is accepted and recorded by the runner, and never read.
  The earlier layout rejected the argument; the shared contract passes it to every algorithm, so it is now
  accepted and ignored.
- Predicate interpretation and derived features: not applicable. The MEDS adaptation is:

  | RETAIN concept | MEDS realisation |
  |---|---|
  | medical code | the `code` string, verbatim |
  | vocabulary | learned in `preprocess_data` from the training split only, ordered by descending frequency then code string. Closed: a code outside it is dropped and counted |
  | visit | events bucketed on an absolute grid of `visit_bucket_minutes` (default 1440). `0` keeps every distinct timestamp as its own visit |
  | visit input | binary multi-hot over the vocabulary; a code repeated inside a bucket counts once |
  | history for a key | events with `time <= prediction_time`, optionally limited by `lookback_days` and `max_visits` |
  | splits | `train` labels fit the parameters, `tuning` labels select the epoch, the requested index splits are scored |

- Temporal cutoff: events are stored per `(subject_id, time)` and are bucketed into visits only after the
  cut at the prediction time, so no visit can straddle a prediction time.
- Empty histories are scored, not excluded: the key gets one all-zero visit and the score is the learned
  intercept. Counts are written to `training.json` and `prediction_report.json`.

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| `preprocess_data` | `retain-patients/v1`: `sequences/**.parquet` of `(subject_id, time, code_index)`, `vocabulary.parquet`, `subject_splits.parquet`, `preprocess.json` (vocabulary settings), `preprocess_report.json` (event accounting) | none (reads MEDS) | `external_meds_dir`; runner manifest, `config.json`, `request.json` |
| `train_supervised` | `retain-supervised-model/v1`: `model.pt` (state dict and vocabulary size), `training.json` (history, split roles, vocabulary digest) | input payload must be `retain-patients/v1`, and its vocabulary settings must equal the configuration | `input_data_dir` manifest digest, `external_labels_dir` |
| `predict_supervised` | `retain-predictions/v1`: `predictions.parquet`, `prediction_report.json`, optional `interpretation.ndjson` | both payload formats; same algorithm and configuration as the fit (core); the prepared data's vocabulary digest must equal the one recorded at fit time | both artifact manifest digests, index path, splits and keys digest |

Eq. (5) contributions are NDJSON, not Parquet, because every Parquet file under a predictions directory is
read as predictions. `config.json` and `request.json` are written by the runner; the algorithm writes
neither.

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
8. **Model selection when there is no validation signal.** If the task supplies no `tuning` labels, a
   seeded `validation_fraction` hold-out is carved from the training rows. If validation AUC is undefined
   (one class only, or no rows), the final epoch is kept and `training.json` records
   `selected_by: final_epoch` instead of claiming a selection that never happened.

Changes made by the conversion, none of which alters a fitted parameter or a score:

- The seed is the `seed` argument of each step, not a configuration field.
- Split roles come from the task's label files (`train`, `tuning`), as the contract defines, instead of
  joining label rows to `subject_splits.parquet`. Labels whose subjects are not in the matching MEDS split
  are refused.
- Prediction uses the configuration recorded at fit time (enforced by the core), so a prediction run can no
  longer override `device`, `batch_size` or `interpretation_top_k`.
- `predicted_boolean_probability` is written as 64-bit floats (was 32-bit).

## Verification

### Primary behavior

- Signal and expected behavior: on a cohort where one code alone determines the label, the real profile
  must reach a held-out AUC above 0.85, and the same cohort with shuffled labels must stay below 0.75.
- How the signal enters the algorithm's selected codes/features: `SIGNAL//POS` is asserted to be in the
  fitted vocabulary and in the multi-hot input of exactly the positive keys, at a varying visit index
  (`test_the_signal_code_is_bound_to_the_model_input`, `test_the_signal_occupies_a_random_visit_position`).
- Example generator used or adaptation made: `meds_algorithm.conformance.signal.build_signal_dataset`,
  unchanged, with hourly visits (`visit_bucket_minutes=60`) so the fixture's hourly events form several
  visits.
- Positive-control result: held-out AUC 0.937; the Eq. (5) attribution names the signal code first for more
  than 80% of positive keys.
- Shuffled-label negative-control result: held-out AUC 0.460.

Both AUCs equal the values recorded for the earlier layout, so the conversion did not change the model.

### Temporal leakage

The shared suite of `meds-algorithm` main (`fa8cdfa`, not yet released; the pinned 0.4.0a5 does not
contain these checks) was run with:

```bash
uv run --with "meds-algorithm[conformance] @ https://github.com/McDermottHealthAI/meds-algorithm/archive/fa8cdfad1140aebf01bd0b544aee6ff580ec98d3.tar.gz" \
  pytest -rs tests/test_conformance.py
```

- Future rewrite: every held-out subject's events after its latest prediction time are replaced and the
  whole lifecycle is rerun; predictions are bit-identical. Passed.
- Null world: on a dataset whose future is independent of its past, held-out AUROC is at chance
  (`alpha = 1e-3`). Passed.
- Sensitivity: with a deliberate one-hour look-ahead added to `_windows.build_sequences` in a scratch copy,
  the future-rewrite check failed (40 of 40 keys moved). The null-world check did not detect that injected
  leak, as its documentation says it may not.

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | 49 passed, 2 skipped | the two skips are the opt-in `meds_dev` and `real_data` tiers |
| Local, core main with temporal checks | command above | 7 passed, 2 skipped | includes the future-rewrite rerun and the null-world test |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | 1 passed | `algorithms/supervised/model.yaml` run by MEDS-DEV on the signal fixture; held-out keys covered |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | not run on this package | the earlier layout's demo run is in the legacy report |
| Full target dataset (none selected for this package) | not run on this package | not applicable | the earlier layout's full-data runs are in the legacy report |

Tests of the earlier command layer (manifests, stage filter, run directories, MEDS-DEV pin, device and
determinism runs) were removed with that layer; the shared suite covers the contract. The model-fidelity,
data-path and primary-behaviour tests were kept and now run against `retain_meds`.

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicates, demo plan | not requested for the converted package | pending |
| Before full target dataset | Demo result, target-dataset/task/predicate compatibility, access, full-run resources | not requested for the converted package | pending |

## Resource measurements

Signal fixture (200 train, 40 tuning, 40 held-out subjects), `tests/e2e.py::TEST_CONFIG`, CPU, WSL2 Ubuntu
on a 12-thread x86-64 machine. Values are from each step's manifest.

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| `preprocess_data` | 0.26 | 82,886,656 | 111,808,512 | none |
| `train_supervised` (3 epochs) | 15.61 | 620,218,368 | 796,921,856 | none |
| `predict_supervised` | 0.14 | 609,399,808 | 799,780,864 | none |

## Environment and provenance

- Algorithm commit and dirty state: measured on the working tree of branch `convert-to-meds-algorithm`
  before its conversion commit; no cluster run was made from it.
- Dataset/task identifiers: synthetic signal fixture and null world of `meds-algorithm` only.
- Dataset, task, and predicate revisions: not applicable (no clinical dataset was read).
- Task-bundle kind (clinical ACES tasks / sampled code-by-horizon tasks) and the commit of the tool that built it: none used.
- Shared task-bundle identifier/path: none used.
- Digest of the bundle's task definitions: none used.
- Cluster checkout commit and job IDs: none.
- Structured-result branch/PR: none.
- Dependency/platform/hardware summary: Python 3.11, torch 2.15.0, `meds-algorithm` 0.4.0a5 (and main
  `fa8cdfa` for the temporal checks), Linux 6.6 (WSL2), 12 CPU threads, no GPU used.

## Known limitations

- The MIMIC-IV demo and full-data gates have not been run on the converted package.
- The temporal-leakage checks are not in the pinned core release, so `uv run pytest` does not run them until
  the pin moves to a release that contains them.
- `numeric_value`, static events and external pre-trained embeddings are not used.
- The MEDS adaptation (visit bucketing, vocabulary filters, empty-history scoring) has no upstream
  counterpart and is the part a reviewer should scrutinise most.
