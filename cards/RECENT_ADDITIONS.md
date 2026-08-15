# Recent additions to the structured-EHR model cards

These cards were added in the 2026-08-03 update.

---

# SurvivEHR

> **Competing-risks, time-to-event foundation model for primary-care EHRs**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Pretrain one longitudinal model to forecast many possible future clinical events and their timing, then adapt that general temporal representation to longer-horizon prognostic endpoints. The central idea is that routine primary-care records contain competing future events, so pretraining should model *what happens next and when* rather than only masked or next-token identity.

## Input

Large longitudinal primary-care event streams containing diagnoses, medications, investigations and measurements, timestamps, and numerical values when available. The accompanying FastEHR pipeline performs scalable extraction, tokenization, filtering, and leakage-aware cohort construction.

## Architecture

A Transformer-based longitudinal foundation model with a competing-risks time-to-event output formulation. Its pretraining head models future event types and timing across heterogeneous primary-care records; downstream prognostic heads are initialized from the pretrained model and optimized on labeled endpoints.

## Pretraining objective

Competing-risks, time-to-event forecasting over many future diagnoses, investigations, medications, measurements, and mortality-related events.

## Original downstream task(s)

Longer-horizon prognostic prediction for multiple long-term conditions, including evaluation in low-label and transfer settings.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The reported downstream workflow initializes task-specific prognostic models from the pretrained SurvivEHR backbone and updates them using labeled endpoint data. The pretraining hazards alone are not the primary downstream evaluation pathway.

## Source paper

- [SurvivEHR — npj Digital Medicine](https://www.nature.com/articles/s41746-026-02709-z)

## Code availability

- **Yes, official — [SurvivEHR](https://github.com/cwlgadd/SurvivEHR)**
- **Yes, official preprocessing — [FastEHR](https://github.com/cwlgadd/FastEHR)**

The paper states that the source used for preprocessing, training, evaluation, and experiment reproduction is public.

## Python package availability

- **Source-installable research software, not a verified PyPI distribution.**

The repositories are structured Python projects, but no dedicated, published `survivehr` package was verified on PyPI as of the check date.

## Interpretation notes

- SurvivEHR is conceptually close to MOTOR and later marked time-to-event objectives, but its primary-care scale, mixed event types, and competing-risks formulation distinguish it.
- The classification reflects the paper’s downstream adaptation experiments; alternative reuse of pretrained event hazards could yield a zero-shot workflow, but that is not the safest primary label.

---

# EveryQuery

> **Zero-Shot Clinical Prediction via Task-Conditioned Pretraining over Electronic Health Records**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Make the clinical prediction task itself part of the model input. Instead of pretraining a generic autoregressive model and later generating many futures, EveryQuery is pretrained across randomly sampled clinical questions so that one fixed model can answer new `(outcome code, prediction horizon)` queries directly.

## Input

A MEDS-formatted patient history up to a prediction time plus a structured query specifying an outcome code and a future duration/window. The current query language focuses on questions such as: “Will code `c` occur within `d` days?”

## Architecture

A ModernBERT-style bidirectional encoder over tokenized MEDS events, augmented with query and duration representations and a prediction head that emits occurrence and censoring probabilities.

## Pretraining objective

Task-conditioned supervised pretraining over millions of automatically generated query–context pairs. Queries and prediction times are sampled from the EHR, and labels are generated automatically from future event occurrence and censoring.

## Original downstream task(s)

Thirty-nine randomly sampled MIMIC-IV prediction tasks, including tasks held out from the pretraining query distribution. The paper also analyzes limitations on compound/disjunctive outcomes such as readmission.

## Pipeline classification

- **Primary strategy:** **4. Direct zero-shot**
  - DAG: `preprocess_data → pretrain → predict`

**Rationale.** At evaluation time, the pretrained model consumes a patient history and a new structured query and emits the requested probability in one deterministic forward pass. It uses no task-specific fine-tuning, separate probe, or generated trajectories.

## Source paper

- [EveryQuery — arXiv](https://arxiv.org/abs/2603.07900)
- [EveryQuery — OpenReview](https://openreview.net/forum?id=c2TbMN7wT0)

## Code availability

- **Yes, official — [payalchandak/EveryQuery](https://github.com/payalchandak/EveryQuery)**

The repository includes MEDS preprocessing, task generation, training, prediction, and evaluation commands.

## Python package availability

- **Yes, official — [`EveryQuery`](https://pypi.org/project/EveryQuery/)**
- Install with `pip install EveryQuery`.

## Interpretation notes

- EveryQuery is the cleanest current example of Strategy 4 in this collection.
- “Zero-shot” here has the stronger DAG meaning: direct endpoint prediction without materializing trajectories or representations.
- The query language is currently narrower than arbitrary clinical phenotypes, so support for complex logical endpoints should not be assumed.

---

# CEHR-XGPT

> **A Scalable Multi-Task Foundation Model for Electronic Health Records**  
> **Year:** 2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Use one temporally explicit autoregressive EHR model for three capabilities: reusable patient representations, task-specific adaptation, and generative forecasting. Artificial time tokens and auxiliary temporal objectives aim to preserve chronology better than plain next-token EHR language modeling.

## Input

Chronological OMOP-coded patient timelines containing diagnoses, procedures, medications, visits, demographics, and explicit time tokens between events. The published model does not cover every OMOP domain; measurements and observations are noted limitations.

## Architecture

A GPT-2-style decoder-only Transformer with a specialized patient timeline representation. In addition to next-token prediction, the model applies time-decomposition and time-to-event objectives to artificial time-token embeddings.

## Pretraining objective

Autoregressive next-event prediction plus auxiliary time decomposition and time-to-event learning over chronological OMOP sequences.

## Original downstream task(s)

Clinical prediction under linear probing and full fine-tuning; zero-shot heart-failure readmission, CABG, and heart-failure risk; external EHRSHOT evaluation; and synthetic patient/treatment-trajectory generation.

## Pipeline classification

- **Strategy 2: Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
  - The final-token patient representation and linear head are trained while all pretrained weights are updated.

- **Strategy 3: Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`
  - The backbone is frozen, final-token representations are extracted, and logistic regression is trained separately.

- **Strategy 5: Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`
  - For the paper’s zero-shot risk evaluation, CEHR-XGPT generates 50 future trajectories and aggregates whether the outcome occurs within the configured window.

**Correction.** CEHR-XGPT is **not** Strategy 4 for the reported zero-shot clinical-risk procedure. “Predict directly from prompts” in the paper means without task-specific retraining; the evaluated endpoint probability is still computed from materialized simulated futures.

## Source paper

- [CEHR-XGPT — arXiv](https://arxiv.org/abs/2509.03643)

## Code availability

- **Yes, official — [knatarajan-lab/cehrgpt](https://github.com/knatarajan-lab/cehrgpt)**
- **Yes, official preprocessing — [knatarajan-lab/cehrbert_data](https://github.com/knatarajan-lab/cehrbert_data)**
- **Tutorials — [knatarajan-lab/cehrgpt_tutorials](https://github.com/knatarajan-lab/cehrgpt_tutorials)**

## Python package availability

- **Source-installable official project; no dedicated PyPI release verified.**

The repository is organized as reusable Python code and has replication instructions, but it is best classified as a source-installed research package rather than a published package distribution.

## Interpretation notes

- This card deliberately lists three separate execution modes because the paper evaluates all three.
- Synthetic data generation is a capability, but it is not itself one of the six task-prediction DAGs unless generated data are subsequently used for prediction.

---

# CoMET

> **Cosmos Medical Event Transformer**  
> **Year:** 2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Learn a general simulator of longitudinal healthcare activity at very large scale, then answer heterogeneous clinical and operational questions by sampling plausible patient futures rather than fitting a new model for each endpoint.

## Input

Large longitudinal event streams from Epic Cosmos, including encounter headers and types, diagnoses, medications, procedures, laboratory-result events, specialties, and other discrete medical events represented as multi-token sequences.

## Architecture

A family of decoder-only autoregressive Transformers from approximately 62 million to 1 billion parameters. The paper studies scaling behavior across model size, pretraining tokens, and compute.

## Pretraining objective

Generic autoregressive next-medical-event prediction over 115 billion medical events from 118 million patients.

## Original downstream task(s)

Seventy-eight clinical and operational tasks, including diagnosis prediction, disease prognosis, laboratory thresholds, encounter utilization, and healthcare operations. Predictions are derived from repeated simulated futures.

## Pipeline classification

- **Primary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** CoMET first generates multiple future medical-event trajectories. Task probabilities, distributions, or expected quantities are then computed from those generated futures without task-specific fine-tuning.

## Source paper

- [Generative Medical Event Models Improve with Scale — arXiv](https://arxiv.org/abs/2508.12104)

## Code availability

- **No verified official public training or inference repository.**

A third-party “code replica” repository exists, but it should not be treated as the authors’ implementation. The Epic Cosmos training data are also not publicly reproducible.

## Python package availability

- **No official Python package found.**

## Interpretation notes

- CoMET is a canonical large-scale Strategy-5 example but a poor first implementation choice because the model, checkpoints, and source training data are not openly available.
- Some immediate next-event analyses may inspect token probabilities, but the paper’s broad downstream evaluation is simulation-based and therefore materialized zero-shot.

---

# Delphi-2M

> **Learning the natural history of human disease with generative transformers**  
> **Year:** 2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Model the competing progression of many diseases on a continuous age axis. Rather than building one endpoint model per disease, Delphi-2M predicts disease-specific incidence rates for more than 1,000 diagnoses simultaneously and can also sample complete future disease trajectories.

## Input

Longitudinal top-level ICD-10 diagnosis tokens plus age, sex, body-mass index, smoking, alcohol consumption, and death. Events are represented on continuous age rather than ordinary discrete token positions.

## Architecture

A modified GPT-2 causal Transformer with continuous-age sinusoidal encodings and a competing-exponential output formulation. The logits correspond to per-day event rates for 1,256 disease/death tokens; an additional time-to-next-event component is learned through the same competing-risks formulation.

## Pretraining objective

Joint autoregressive prediction of the next disease token and its waiting time under a competing exponential model.

## Original downstream task(s)

Simultaneous disease-incidence prediction for more than 1,000 diseases, external validation on Danish registry data without parameter updates, long-horizon disease-risk forecasting, and synthetic future-trajectory generation.

## Pipeline classification

- **Primary strategy:** **4. Direct zero-shot**
  - DAG: `preprocess_data → pretrain → predict`
  - A single model pass produces disease-specific log-rates. A fixed-horizon probability can be deterministically computed from a rate, for example `1 - exp(-rate × horizon)`.

- **Secondary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`
  - When the model samples future disease trajectories and downstream burdens are calculated from those samples.

**Rationale.** Delphi-2M supports both direct hazard/rate-based prediction and trajectory simulation. The card separates those two workflows rather than assigning one ambiguous “zero-shot” label.

## Source paper

- [Delphi-2M — Nature](https://www.nature.com/articles/s41586-025-09529-3)

## Code availability

- **Yes, official — [gerstung-lab/Delphi](https://github.com/gerstung-lab/Delphi)**

The repository includes training code, evaluation scripts, and trajectory-sampling notebooks.

## Python package availability

- **No published reusable Python package.**

The official release is a research repository installed through a requirements file rather than a packaged library API.

## Checkpoint availability

- The useful Delphi-2M checkpoint is available through UK Biobank controlled-access procedures.
- The repository’s openly available synthetic/demo data and model are intended for software testing, not substantive research conclusions.

## Interpretation notes

- Delphi-2M is narrower in event vocabulary than full hospital EHR models: it focuses mainly on disease progression and selected baseline/lifestyle variables.
- Its direct-risk mode makes it a stronger Strategy-4 example than trajectory-only generative models.

---

# ETHOS / ARES

> **Enhanced Transformer for Health Outcome Simulation / Adaptive Risk Estimation System**  
> **Year:** 2024/2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Represent the entire patient record as a specialized “Patient Health Timeline” language, autoregressively complete that timeline into the future, and estimate clinical risks from the frequency of target events across repeated completions. ARES adds dynamic risk estimation and patient-specific explanations around the ETHOS generator.

## Input

Tokenized chronological Patient Health Timelines derived from heterogeneous MIMIC-IV and MIMIC-IV-ED events, including admissions, demographics, diagnoses, procedures, medications, measurements, and explicit elapsed-time tokens. The current reimplementation uses MEDS as an intermediate representation.

## Architecture

A GPT-2-style decoder-only Transformer trained over the specialized health-event token language. The maintained ETHOS/ARES implementation currently uses a bias-free GPT-2 architecture and provides tokenization, training, trajectory inference, and risk-analysis commands.

## Pretraining objective

Autoregressive next-token prediction over Patient Health Timelines.

## Original downstream task(s)

Zero-shot inpatient mortality, 30-day readmission, ICU admission, prolonged length of stay, composite endpoints, and related trajectory-based risk estimates. ARES additionally provides individualized explanations of estimated risk.

## Pipeline classification

- **Primary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** ETHOS repeatedly generates future Patient Health Timelines. The probability of an endpoint is the fraction of generated futures in which the target event occurs within the specified stopping rule or horizon.

## Source paper

- [ETHOS — npj Digital Medicine](https://www.nature.com/articles/s41746-024-01235-0)
- [ARES — GigaScience](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf107/8268904)

## Code availability

- **Yes, official — [ipolharvard/ethos-ares](https://github.com/ipolharvard/ethos-ares)**
- Historical paper implementation: [ipolharvard/ethos-paper](https://github.com/ipolharvard/ethos-paper)

## Python package availability

- **Yes, source-installable official package.**

The repository has a `pyproject.toml`, installs with `pip install -e .`, and exposes commands including `ethos_tokenize`, `ethos_train`, and `ethos_infer`. No separate PyPI distribution was verified.

## Interpretation notes

- Although the papers use “zero-shot,” the concrete risk DAG is Strategy 5 because generated futures are materialized and aggregated.
- ARES is best understood as a risk-estimation and explanation layer over ETHOS rather than an independently pretrained backbone.

---

# MEDS-EIC-AR

> **MEDS “Everything-is-code” Autoregressive Model**  
> **Year:** 2025–2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Provide a reusable, MEDS-native autoregressive baseline in which heterogeneous structured EHR events—including codes, values, and timing information—are serialized into one causal token stream. The same model can be pretrained on any MEDS cohort and used to generate patient futures for zero-shot evaluation.

## Input

MEDS event streams transformed with an “everything-is-code” serialization and tensorized through the MEDS ecosystem. Inputs may include diagnoses, medications, procedures, laboratory values, visits, and explicit temporal/event tokens, depending on the source MEDS dataset and configuration.

## Architecture

A configurable causal Transformer implemented with PyTorch Lightning and Hugging Face causal-language-model components. The current generation backend wraps `LlamaForCausalLM.generate`; Hydra configurations expose model sizes, tokenization, training, and trajectory-generation settings.

## Pretraining objective

Autoregressive next-token prediction over tokenized MEDS patient timelines.

## Original downstream task(s)

Repository-packaged zero-shot generation for arbitrary MEDS task cohorts. For each labeled prediction time, the model generates multiple future trajectories in a pseudo-MEDS trajectory schema for later endpoint resolution.

## Pipeline classification

- **Current documented strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** The package’s documented workflow generates and saves multiple future trajectories before converting them to endpoint probabilities. At the check date, trajectory generation is implemented, while the repository’s built-in “resolve trajectories into predictions” stage is explicitly marked not yet implemented.

### Plausible extension, not currently packaged

A classification head could be attached to the pretrained hidden state and the backbone updated on downstream labels:

`preprocess_data → pretrain → supervised_train → predict`

That would make **MEDS-EIC-AR-FT** a Strategy-2 model, but it should not be presented as an existing supported workflow until a supervised fine-tuning command and evaluation are added.

## Source paper

- **No dedicated model paper was identified.** The primary technical specification is the official repository/package documentation.
- [Official repository](https://github.com/mmcdermott/MEDS_EIC_AR)

## Code availability

- **Yes, official — [mmcdermott/MEDS_EIC_AR](https://github.com/mmcdermott/MEDS_EIC_AR)**

## Python package availability

- **Yes, official — [MEDS-EIC-AR on PyPI](https://pypi.org/project/MEDS-EIC-AR/)**
- Install with `pip install MEDS-EIC-AR`.

## Reproducibility caveat

The repository warns that its tests establish execution and basic numerical sanity, not reliable convergence or full scientific correctness. This caveat should remain visible in benchmark documentation.

---

# ORA

> **One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Treat the EHR as a marked temporal process rather than a plain sequence of discrete codes. ORA is principally a pretraining objective: it jointly models when a future code occurs and the associated numerical value when the event is a measurement, aiming to learn representations useful across classification, survival, and regression.

## Input

Irregular structured-EHR sequences containing heterogeneous discrete events and numeric measurements, with event codes, timestamps, and values. Experiments use MIMIC-IV and Columbia University Irving Medical Center data.

## Architecture

ORA is backbone-agnostic and is evaluated with both Transformer and Mamba sequence backbones. A code-specific marked time-to-event prediction head discretizes future time and value distributions while supporting codes without numerical marks.

## Pretraining objective

A marked time-to-event objective that jointly models each medical code’s future occurrence time and associated numerical value, while handling censoring, concurrent events, and mixed event types.

## Original downstream task(s)

Seven binary classification tasks, four time-to-event tasks—including AMI, MASLD, stroke, and celiac disease where available—and four laboratory-value regression tasks.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** In the reported evaluation, pretrained models generate task-cohort embeddings and separate task-specific heads are fit: logistic regression for classification, DeepHit for time-to-event prediction, and linear regression for numerical outcomes. The downstream evaluation therefore uses frozen representations rather than end-to-end fine-tuning.

## Source paper

- [ORA — arXiv](https://arxiv.org/abs/2602.00541)
- [ORA — OpenReview](https://openreview.net/forum?id=WnxEI2t9be)

## Code availability

- **No dedicated official repository was verified in a targeted search.**

The paper is sufficiently detailed to identify the objective and evaluation pathway, but this card should not claim a public implementation until the authors link one explicitly.

## Python package availability

- **No dedicated Python package found.**

## Interpretation notes

- ORA is best viewed as an objective/family of pretrained models rather than one fixed architecture.
- If a future implementation updates the backbone with labeled downstream data, that branch would additionally qualify as Strategy 2.

---

# SurvPFN

> **Towards Foundation Models for Survival Predictions**  
> **Year:** 2026  
> **Collection:** Recent additions; tabular survival rather than longitudinal-EHR-specific  
> **Availability checked:** 2026-08-03

## Model philosophy

Amortize survival-model fitting across millions of synthetic tasks. SurvPFN is pretrained once on a prior over censored survival datasets and then solves a new dataset by reading its labeled support rows in context—without gradient-based fitting for that dataset.

## Input

A tabular survival task consisting of training/support rows with feature vectors, observed or censored event indicators, and event/censoring times, together with test/query feature rows. The released checkpoint supports a limited feature-count range and optional categorical-feature indices.

## Architecture

A prior-data fitted Transformer based on NanoTabPFN, extended with a two-route target encoder for observed versus censored rows and a survival-specific output distribution. It follows a scikit-survival-style estimator interface.

## Pretraining objective

Right-censored negative log-likelihood over millions of synthetic survival tasks generated from structural causal models with Weibull event times and non-informative censoring.

## Original downstream task(s)

Zero-shot time-to-event prediction on real-world SurvSet datasets, producing risk scores and survival functions without per-dataset gradient training.

## Pipeline classification

- **Primary strategy:** **6. PFN-style**
  - DAG: `preprocess_data → preprocess_task → predict`

**Rationale.** The pretrained model is fixed. Its `fit` method only stores/encodes the support set; it does not optimize parameters. Predictions for query rows are made in context from the supplied task data.

## Source paper

- [SurvPFN — arXiv](https://arxiv.org/abs/2606.04564)

## Code availability

- **Yes, official — [genepi-freiburg/SurvPFN](https://github.com/genepi-freiburg/SurvPFN)**
- **Pretrained checkpoint — [Hugging Face](https://huggingface.co/samuelboehm/SurvPFN)**

## Python package availability

- **Yes, source-installable official project; no PyPI release verified.**

The repository contains a `pyproject.toml` and a reusable estimator API, but installation is currently repository-based.

## Scope note

SurvPFN is not a longitudinal structured-EHR sequence model. It is included because it is the clearest recent implementation of your Strategy-6 execution semantics for health-related survival prediction. It should therefore be labeled “tabular survival / other” in analyses that restrict the main benchmark to longitudinal MEDS models.

