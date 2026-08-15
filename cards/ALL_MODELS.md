# All structured-EHR model cards

---

# TECO

> **Transformer-based Encounter-level Clinical Outcome model**  
> **Year:** 2025  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Update risk continuously as an ICU encounter evolves, using a single encounter-level sequence model rather than separate models for each prediction time.

## Input

Hourly longitudinal laboratory values, vital signs, medications, and baseline patient characteristics.

## Architecture

A Transformer encodes the evolving encounter-level time series and produces repeatedly updated mortality-risk estimates.

## Original downstream task(s)

Continuously updated ICU mortality prediction; native binary endpoint.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The published workflow learns the mortality predictor directly from labeled ICU encounters, without a separate reusable pretraining stage.

## Source paper

- [TECO paper](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091)

## Code availability

- **Not found — No public model-specific repository found**  
  Targeted search checked through 2026-07-30; this is not proof that no private or later repository exists.

## Python package availability

- **No — No dedicated Python package found**  
  A paper implementation repository was not found, so there is also no verified model-specific installable package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# STraTS-mTAND

> **Hybrid irregular-time-series model**  
> **Year:** 2025  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Combine a native sparse-observation representation with continuous-time interpolation so the model can exploit irregular measurements without committing entirely to either interpolation or non-interpolation.

## Input

Irregular multivariate clinical time series represented through STraTS-style observation tokens and mTAND-style continuous-time interpolation.

## Architecture

A hybrid of the STraTS Transformer representation and mTAND continuous-time attention/interpolation, followed by a supervised mortality head.

## Original downstream task(s)

In-hospital mortality; native binary endpoint.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The hybrid encoder and classifier are trained for the labeled mortality task; the paper table does not define an independent foundation-model pretraining phase.

## Source paper

- [STraTS-mTAND paper](https://dl.acm.org/doi/10.1145/3743689)

## Code availability

- **Not found — No model-specific public repository found**  
  Repositories for the component methods are not equivalent to an implementation of this hybrid model.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# TRANS

> **Temporal gRAph traNSformer**  
> **Year:** 2024  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent a patient history as an evolving heterogeneous graph so temporal relations and cross-type clinical relations are modeled explicitly rather than flattened into one token stream.

## Input

Longitudinal diagnosis, procedure, and medication codes organized as temporal heterogeneous patient graphs.

## Architecture

Temporal graph construction plus graph-aware Transformer layers that aggregate heterogeneous clinical events for next-visit prediction.

## Original downstream task(s)

Next-visit diagnosis prediction; an individual diagnosis can be evaluated with one sigmoid output or one-vs-rest.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The model is trained end-to-end on the downstream next-visit labels rather than adapted from a separately pretrained backbone.

## Source paper

- [TRANS paper](https://arxiv.org/abs/2405.03943)

## Code availability

- **Yes, official — [TRANS GitHub repository](https://github.com/The-Real-JerryChen/TRANS)**

## Python package availability

- **No — No dedicated pip-installable package found**  
  The official research repository is code, not a verified published Python package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# PRISM

> **Missingness-calibrated prototype patient representations**  
> **Year:** 2024  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Treat EHR sparsity as informative and uncertain, then borrow statistical strength from learned prototype patients with similar calibrated observation patterns.

## Input

Longitudinal laboratory and vital-sign features, demographics, explicit missingness-confidence features, and prototype-based patient relationships.

## Architecture

A clinical sequence encoder is coupled to learned patient prototypes and graph/message-passing mechanisms, with missing-feature confidence used in representation learning.

## Original downstream task(s)

In-hospital mortality and 30-day readmission; native binary tasks.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Prototype learning and the endpoint head are optimized from labeled task examples; there is no separate general pretraining command in the described pipeline.

## Source paper

- [PRISM paper](https://arxiv.org/abs/2309.04160)

## Code availability

- **Yes, official — [PRISM GitHub repository](https://github.com/yhzhu99/PRISM)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# GenHPF

> **General Healthcare Predictive Framework — supervised/journal configuration**  
> **Year:** 2024  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use one schema-agnostic hierarchical serialization so the same predictor can consume heterogeneous EHR tables and tasks with minimal hand-written domain mapping.

## Input

Structured events serialized hierarchically from table/event type, field names, field values, and time intervals across differing EHR schemas.

## Architecture

A hierarchical event serializer and Transformer-style sequence encoder feed task-specific heads; the framework supports multi-source and multi-task training.

## Original downstream task(s)

Twelve mortality, readmission, diagnosis, acuity, and related binary or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** This document describes the supervised configuration listed in the first table: task labels directly train the model. The same paper also studies self-supervised pretraining, documented separately in the pretrained-model folder.

## Source paper

- [GenHPF paper](https://arxiv.org/abs/2207.09858)

## Code availability

- **Yes, official — [GenHPF GitHub repository](https://github.com/hoon9405/GenHPF)**

## Python package availability

- **No — No dedicated published Python package found**  
  The repository is reusable research code, but no model-specific PyPI distribution was verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# KerPrint

> **Local-global knowledge-graph enhanced diagnosis prediction**  
> **Year:** 2023  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Enrich each patient trajectory with both local patient-specific relations and global medical knowledge, while allowing elapsed time to control which knowledge is relevant.

## Input

Visit sequences augmented with local and global medical knowledge graphs.

## Architecture

Time-aware knowledge attention combines visit representations with graph-derived local and global medical context before future-diagnosis prediction.

## Original downstream task(s)

Future diagnosis prediction; each diagnosis output can be treated as a binary sigmoid target.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The knowledge-enhanced network is trained directly on future-diagnosis labels, with no separate generic pretraining stage.

## Source paper

- [KerPrint paper](https://ojs.aaai.org/index.php/AAAI/article/view/25667/25439)

## Code availability

- **Yes, official — [KerPrint GitHub repository](https://github.com/xyxpku/KerPrint)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# Chet

> **Context-Aware Health Event Prediction via Transition Functions**  
> **Year:** 2022  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Model disease-state evolution as transitions between dynamic graphs, so the representation explicitly captures how clinical context changes from visit to visit.

## Input

Diagnosis-code histories converted into visit-specific dynamic disease graphs.

## Architecture

Graph encoders construct disease-state representations and learned transition functions propagate context across visits before prediction.

## Original downstream task(s)

Next-visit diagnosis and heart-failure prediction; heart failure is a native binary task.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The graph-transition model is optimized directly against labeled future-event or heart-failure outcomes.

## Source paper

- [Chet paper](https://arxiv.org/abs/2112.05195)

## Code availability

- **Yes, official — [Chet GitHub repository](https://github.com/LuChang-CS/Chet)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# UniHPF

> **Universal Healthcare Predictive Framework**  
> **Year:** 2022  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Avoid manually harmonizing every EHR schema by turning field names and values themselves into a universal hierarchical language of structured events.

## Input

Event types, field names, field values, and elapsed-time tokens from heterogeneous EHR systems.

## Architecture

A hierarchical Transformer encodes fields within events and events across time, with task heads for diverse clinical endpoints.

## Original downstream task(s)

Mortality, readmission, length of stay, acuity, discharge, and diagnosis; binary and multilabel outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Despite its general architecture, the table entry describes supervised task training rather than a frozen pretrained foundation model.

## Source paper

- [UniHPF paper](https://arxiv.org/abs/2211.08082)

## Code availability

- **Yes, official — [UniHPF GitHub repository](https://github.com/hoon9405/UniHPF)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# SETOR

> **Sequential diagnosis prediction with ontological representation**  
> **Year:** 2021  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Combine visit sequences with a medical ontology and continuous-time dynamics so predictions reflect both semantic code relationships and irregular disease progression.

## Input

Sequential diagnosis codes, code-ontology relations, visit structure, and time intervals.

## Architecture

A Transformer-based visit encoder is paired with ontology-aware code representations and neural-ODE/continuous-time modeling.

## Original downstream task(s)

Sequential diagnosis prediction; individual diagnoses use binary sigmoid outputs or one-vs-rest evaluation.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The complete ontology-aware temporal network is fitted on downstream diagnosis labels.

## Source paper

- [SETOR paper](https://arxiv.org/abs/2109.03069)

## Code availability

- **Yes, official — [SETOR GitHub repository](https://github.com/Xueping/SETOR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# GRASP

> **Generic framework using similar-patient knowledge**  
> **Year:** 2021  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn latent patient prototypes and propagate information among similar patients, allowing a task backbone to benefit from cohort structure rather than treating records independently.

## Input

Structured longitudinal variables suitable for a task-specific backbone, plus learned patient-similarity structure.

## Architecture

A base clinical encoder produces patient states; clustering/prototypes and graph propagation refine them before a prediction head.

## Original downstream task(s)

Patient health-status and clinical-outcome prediction; evaluated endpoints include native binary tasks.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The patient encoder, prototypes, graph module, and endpoint head are trained for the supervised task.

## Source paper

- [GRASP paper](https://ojs.aaai.org/index.php/AAAI/article/view/16152)

## Code availability

- **Yes, official — [GRASP GitHub repository](https://github.com/choczhang/GRASP)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.GRASP`. This is a toolkit implementation, not the original authors’ standalone package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# GCT

> **Graph Convolutional Transformer**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Regularize learned attention with empirical EHR co-occurrence structure, combining flexible Transformer attention with an inductive graph bias.

## Input

Sets or sequences of structured EHR concepts and empirical concept co-occurrence relationships.

## Architecture

Transformer attention is constrained or regularized by a learned graph-convolutional structure over EHR concepts.

## Original downstream task(s)

Readmission and other patient-level outcomes; native binary classification.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The graph-constrained Transformer and task head are trained end-to-end using labeled outcomes.

## Source paper

- [GCT paper](https://arxiv.org/abs/1906.04716)

## Code availability

- **Yes, third-party reproduction — [GCT reproduction repository](https://github.com/lycpaul/dl4h-gp30-gct)**  
  No authors’ repository was verified; this link is a reproduction rather than an official release.

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# StageNet

> **Stage-aware neural network for health-risk prediction**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent disease progression as latent stages and stage transitions, so rapidly changing periods receive different treatment from stable periods.

## Input

Irregular longitudinal clinical measurements and their timing.

## Architecture

A stage-aware LSTM tracks progression states; stage-adaptive convolution aggregates local temporal patterns for risk prediction.

## Original downstream task(s)

Health-risk and clinical-deterioration prediction; native binary endpoints.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Stage representations and endpoint prediction are learned directly from labeled examples.

## Source paper

- [StageNet paper](https://arxiv.org/abs/2001.10054)

## Code availability

- **Yes, official — [StageNet GitHub repository](https://github.com/v1xerunt/StageNet)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.StageNet`.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# HiTANet

> **Hierarchical time-aware attention network**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Make long-term disease-risk predictions interpretable by separately weighting codes within visits and visits across time, with explicit recency decay.

## Input

Visit-level medical-code sequences and intervals between visits.

## Architecture

Hierarchical code-level and visit-level attention with time-decay functions produces a patient representation for risk prediction.

## Original downstream task(s)

Disease-risk prediction from longitudinal records; native binary endpoints.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The hierarchical attention network is optimized directly for labeled disease-risk outcomes.

## Source paper

- [HiTANet paper](https://dl.acm.org/doi/10.1145/3394486.3403107)

## Code availability

- **Yes, official — [HiTANet GitHub repository](https://github.com/machinelearning4health/HiTANet)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# AdaCare

> **Scale-adaptive clinical feature learning**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Detect predictive patterns at multiple temporal scales and recalibrate features dynamically, supporting both accuracy and feature-level interpretability.

## Input

Multivariate clinical time series.

## Architecture

Multiscale dilated convolutions extract local patterns; scale-adaptive feature recalibration and a recurrent aggregator produce the patient state.

## Original downstream task(s)

ICU and longitudinal health-risk prediction; native binary endpoints.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The multiscale encoder and binary task head are jointly supervised on the target endpoint.

## Source paper

- [AdaCare paper](https://arxiv.org/abs/1911.12205)

## Code availability

- **Yes, official — [AdaCare GitHub repository](https://github.com/Accountable-Machine-Intelligence/AdaCare)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.AdaCare`.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# ConCare

> **Context-aware personalized clinical feature embedding**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Encode each clinical variable separately, then learn time-aware and cross-feature interactions so the model can adapt to a patient’s particular clinical context.

## Input

Dynamic clinical measurements together with static covariates.

## Architecture

Per-feature recurrent encoders feed time-aware attention and decorrelated multi-head attention, followed by an outcome head.

## Original downstream task(s)

Clinical outcome prediction; native binary endpoints.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The feature encoders, attention layers, and endpoint classifier are trained end-to-end with task labels.

## Source paper

- [ConCare paper](https://arxiv.org/abs/1911.12216)

## Code availability

- **Yes, official — [ConCare GitHub repository](https://github.com/Accountable-Machine-Intelligence/ConCare)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.ConCare`.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# SAnD

> **Simply Attend and Diagnose**  
> **Year:** 2018  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Replace recurrence with masked self-attention for clinical time series, enabling direct access to long-range dependencies and parallel training.

## Input

Regular-grid structured ICU time series with positional information.

## Architecture

Masked self-attention layers and positional encoding produce sequence features; dense interpolation converts variable-length sequences into a fixed-size representation.

## Original downstream task(s)

MIMIC-III mortality, decompensation, length of stay, and phenotyping; binary and multilabel outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The attention encoder is trained from scratch for the selected supervised clinical task.

## Source paper

- [SAnD paper](https://arxiv.org/abs/1711.03905)

## Code availability

- **Yes, third-party implementation — [SAnD GitHub repository](https://github.com/khirotaka/SAnD)**  
  This is a public implementation; an authors’ official repository was not verified.

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# Dipole

> **Attention-based bidirectional recurrent diagnosis predictor**  
> **Year:** 2017  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use both forward and backward contextualization of prior visits, then attend to the most relevant visits when forecasting the next diagnoses.

## Input

Sequences of visits represented by diagnosis or other medical-code vectors.

## Architecture

A bidirectional recurrent network encodes visits and a visit-level attention mechanism forms the context used for next-visit prediction.

## Original downstream task(s)

Next-visit diagnosis prediction; individual codes are binary sigmoid outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The bidirectional RNN, attention, and diagnosis head are trained directly on next-visit labels.

## Source paper

- [Dipole paper](https://arxiv.org/abs/1706.05764)

## Code availability

- **Yes, third-party/archived toolkit — [PyHealth repository](https://github.com/sunlabuiuc/PyHealth)**  
  A standalone original-author repository was not verified; older PyHealth releases included a Dipole implementation.

## Python package availability

- **Historical only — [Archived PyHealth implementation](https://pyhealth.readthedocs.io/en/archived/)**  
  Older PyHealth APIs exposed Dipole, but it is not listed as a current stable PyHealth 2.0 model. Pin and validate an older release before use.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# GRAM

> **Graph-based Attention Model**  
> **Year:** 2017  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Share statistical strength from clinical ontologies, especially for rare codes, by representing each code as an attention-weighted combination of itself and its ancestors.

## Input

Medical-code sequences plus ancestor relationships from a clinical ontology.

## Architecture

Ontology-aware attention builds code embeddings from ancestors; a recurrent patient-history encoder and prediction head model future diagnoses or risk.

## Original downstream task(s)

Diagnosis and heart-failure prediction; heart failure is a native binary task.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Ontology embeddings and the downstream predictor are trained jointly from labeled patient trajectories.

## Source paper

- [GRAM paper](https://arxiv.org/abs/1611.07012)

## Code availability

- **Yes, official — [GRAM GitHub repository](https://github.com/mp2893/gram)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# RETAIN

> **Reverse Time Attention Model**  
> **Year:** 2016  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Provide interpretable risk predictions through a reverse-time sequence model whose two attentions separately expose important visits and important variables.

## Input

Longitudinal visits represented by coded or structured features, processed in reverse chronological order.

## Architecture

Two recurrent attention networks estimate visit-level scalar weights and variable-level vector weights; their weighted history feeds a prediction layer.

## Original downstream task(s)

Interpretable patient-level risk prediction, including heart failure; native binary endpoints.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The attention mechanisms and risk head are trained directly for the labeled endpoint.

## Source paper

- [RETAIN paper](https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)

## Code availability

- **Yes, official — [RETAIN GitHub repository](https://github.com/mp2893/retain)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.RETAIN`.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# Doctor AI

> **Predicting clinical events with recurrent neural networks**  
> **Year:** 2016  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Treat a patient record as a clinical sequence and learn to forecast the next encounter’s content and timing in one recurrent model.

## Input

Timestamped diagnosis, medication, and procedure histories.

## Architecture

A GRU-based sequence model predicts next-visit codes and the time to the next encounter.

## Original downstream task(s)

Next-visit diagnoses, medications, and time to next encounter; individual code outputs are binary sigmoid targets.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The recurrent model is learned directly from the supervised next-event targets in the task dataset.

## Source paper

- [Doctor AI paper](https://proceedings.mlr.press/v56/Choi16.html)

## Code availability

- **Yes, official — [Doctor AI GitHub repository](https://github.com/mp2893/doctorai)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# PORTER

> **Language-grounded event representations for portable structured EHR foundation models**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Ground structured event names in natural-language descriptions so one temporal model can transfer across institutions and schemas without relying on clinical notes.

## Input

Timestamped conditions, procedures, medications, measurements, encounters, numeric values, reference ranges, and demographics; event names are encoded by a frozen text encoder.

## Architecture

A frozen text-based input pathway feeds a 115M-parameter causal temporal Transformer pretrained autoregressively on next-event prediction. Downstream tasks use extracted representations.

## Pretraining objective

Autoregressive next-event prediction; the pretraining output head is discarded for adaptation.

## Original downstream task(s)

Seventy-four binary tasks covering outcomes, laboratory abnormalities, procedures, imaging, medications, and transfusions.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The pretrained backbone and input pathway remain frozen; representations are materialized and an L2-logistic-regression probe is trained on labeled endpoints.

## Source paper

- [PORTER paper](https://arxiv.org/abs/2606.24102)

## Code availability

- **Not found — No public model-specific repository found**  
  No code link was verified from the paper page or targeted public search through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# AID-MAE

> **Augmented-Intrinsic Dual-Masked Autoencoder**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn from sparse clinical tables without imputing naturally missing observations, using both real missingness and synthetic masking as complementary learning signals.

## Input

Observed laboratory and vital-sign value–timestamp tokens from sparse, irregular tables; naturally missing entries are not first imputed.

## Architecture

A masked-autoencoder encoder reconstructs values under an intrinsic missingness mask and an additional stochastic mask; downstream heads use the learned encoder.

## Pretraining objective

Dual-masked autoencoding over naturally sparse and additionally masked observations.

## Original downstream task(s)

Mortality, acute kidney injury, and binary length-of-stay endpoints on MIMIC-IV and PhysioNet 2012.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The paper evaluates both full encoder fine-tuning with a binary MLP head (Strategy 2) and a frozen encoder followed by logistic-regression probing (Strategy 3).

## Source paper

- [AID-MAE paper](https://arxiv.org/abs/2602.15159)

## Code availability

- **Not found — No public model-specific repository found**  
  No code link was verified from the paper page or targeted public search through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# HealthFormer

> **Dual-level time-aware Transformer for structured EHR**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Pretrain at both code and event levels while explicitly modeling elapsed time, aiming to transfer across incident-disease horizons from irregular administrative EHR sequences.

## Input

ICD-10 diagnoses, procedures, ATC medications, encounter types, contextual fields, and elapsed times in irregular event sequences.

## Architecture

A dual-level time-aware Transformer represents codes within events and events across the longitudinal trajectory, then adds a sigmoid endpoint head.

## Pretraining objective

Multi-task self-supervision: masked code prediction, whole-event masking/event-type prediction, next-event-type prediction, and time-to-next-event regression.

## Original downstream task(s)

Incident colorectal and prostate cancer at 30-, 60-, and 90-day horizons.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The self-supervised Transformer and new endpoint head are updated end-to-end on labeled downstream examples.

## Source paper

- [HealthFormer paper](https://www.medrxiv.org/content/10.64898/2026.03.25.26349262v2.full-text)

## Code availability

- **Yes, official — [HealthFormer GitHub repository](https://github.com/renyi-ai/HealthFormer)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# PULSE-ICU

> **Pretrained Unified Long-Sequence Encoder for ICU**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Model long, unresampled ICU event streams directly, preserving irregular timing and event metadata while sharing one pretrained encoder across many ICU tasks.

## Input

Unresampled laboratory, vital-sign, medication, fluid, procedure, unit, value, timestamp, and event-metadata sequences.

## Architecture

A Longformer-based event encoder handles long ICU timelines; task-specific heads are attached for multi-task downstream prediction.

## Pretraining objective

Masked event and masked value prediction.

## Original downstream task(s)

Eighteen mortality, intervention-forecasting, phenotype, and other binary, multiclass, or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained encoder is adapted with supervised multi-task fine-tuning and task-specific prediction heads.

## Source paper

- [PULSE-ICU paper](https://arxiv.org/abs/2511.22199)

## Code availability

- **Yes, official — [PULSE-ICU GitHub repository](https://github.com/sejeongak/PULSE-ICU)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# BAT foundation-model configuration

> **Self-supervised foundation-model configuration for critical-care time series**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use a missingness-aware set/attention model to learn general ICU dynamics from sparse future-value forecasting before testing how much of the encoder must be updated.

## Input

Sparse multivariate ICU time series with missingness-aware set/attention representations.

## Architecture

The BAT encoder uses attention over sparse observations and missingness structure; a downstream mortality head is attached after forecasting pretraining.

## Pretraining objective

Future-value forecasting with masked mean-squared error.

## Original downstream task(s)

In-hospital mortality.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The study compares full-model fine-tuning (Strategy 2) with head-only tuning while the pretrained encoder is frozen (a Strategy 3-style probe branch).

## Source paper

- [BAT foundation-model configuration paper](https://arxiv.org/abs/2509.19885)

## Code availability

- **Yes, official project code — [YAIB repository](https://github.com/Katja-Jagd/YAIB)**  
  The study reports its implementation as an extension of the YAIB framework rather than a standalone BAT package.

## Python package availability

- **No dedicated model package — Research framework only**  
  A model-specific pip distribution was not verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# TOO-BERT

> **Trajectory Order Objective BERT**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Teach a BERT-style EHR encoder not only which codes fit locally, but also whether an entire clinical trajectory is in a plausible temporal order.

## Input

Longitudinal diagnosis, procedure, medication, and encounter-code trajectories with explicit visit and code order.

## Architecture

A Transformer encoder combines masked-code learning with a trajectory-order discriminator and is later paired with a classification head.

## Pretraining objective

Masked-code modeling plus discrimination of correctly ordered trajectories from clinically informed permutations.

## Original downstream task(s)

Heart failure, Alzheimer disease, and prolonged length of stay.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained Transformer and classification head are jointly updated during supervised downstream fine-tuning.

## Source paper

- [TOO-BERT paper](https://medinform.jmir.org/2025/1/e68138)

## Code availability

- **Yes, official — [TOO-BERT GitHub repository](https://github.com/ali-amirahmadii/TOO-Bert)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# CORE-BEHRT

> **Carefully Optimized and Rigorously Evaluated BEHRT**  
> **Year:** 2024  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Treat EHR foundation-model performance as an engineering and evaluation problem: systematically optimize masking, time representations, architecture, and training before claiming gains.

## Input

Longitudinal structured EHR sequences containing diagnoses, medications, timestamps, ages, and encounter information.

## Architecture

An optimized BEHRT-style Transformer with revised temporal encoding, masking, architecture, and training choices.

## Pretraining objective

BEHRT-style masked-event prediction with systematically optimized design and training procedure.

## Original downstream task(s)

Death, pain treatment, infection, and 25 additional predominantly binary clinical prediction tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** Pretrained weights are typically fine-tuned end-to-end with a task-specific endpoint head.

## Source paper

- [CORE-BEHRT paper](https://raw.githubusercontent.com/mlresearch/v252/main/assets/odgaard24a/odgaard24a.pdf)

## Code availability

- **Yes, official — [CORE-BEHRT repository](https://github.com/mikkelfo/CORE-BEHRT)**  
  The repository points users toward the newer BONSAI/EHR framework for continued development.

## Python package availability

- **No dedicated model package — No verified CORE-BEHRT PyPI distribution**  
  The public repositories are installable research frameworks, but a model-specific published package was not verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# EBCL

> **Event-Based Contrastive Learning**  
> **Year:** 2024  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use a clinically meaningful index event to define two views of the same patient, encouraging representations before and after that event to remain patient-specific and predictive.

## Input

Heterogeneous longitudinal measurements and coded events before and after a designated index event.

## Architecture

A sequence encoder maps pre-index and post-index contexts to representations optimized by a patient-level contrastive objective, then initializes downstream supervised models.

## Pretraining objective

Contrastive alignment of the same patient’s pre-index and post-index contexts while separating different patients.

## Original downstream task(s)

Mortality, readmission, and length of stay in heart-failure and ICU cohorts.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained encoder initializes task-specific supervised models that are trained using downstream labels.

## Source paper

- [EBCL paper](https://proceedings.mlr.press/v252/oufattole24a.html)

## Code availability

- **Yes, official — [EBCL GitHub repository](https://github.com/mit-ccrg/EBCL)**

## Python package availability

- **Yes, package implementation — [MEDS-torch](https://pypi.org/project/meds-torch/)**  
  Install with `pip install meds-torch`; the package includes an EBCL pretraining method in a general MEDS-compatible training framework.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# MOTOR

> **Many Outcome Time Oriented Representations**  
> **Year:** 2024  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Scale supervision automatically by treating many codes as time-to-event outcomes, producing reusable temporal representations and calibrated event-risk functions.

## Input

Timestamped diagnosis, procedure, medication, laboratory, encounter, and claims-event sequences.

## Architecture

A temporal Transformer-like foundation model is pretrained over many automatically generated time-to-event objectives and adapted to labeled survival endpoints.

## Pretraining objective

Large-scale self-supervised or weakly supervised time-to-event prediction over many automatically defined event outcomes.

## Original downstream task(s)

Nineteen time-to-event tasks; fixed-horizon cumulative risk can be used as a binary prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The many-outcome pretrained weights are adapted with labeled task-specific survival/time-to-event training. The output family is survival/TTE rather than an ordinary single-logit classifier.

## Source paper

- [MOTOR paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html)

## Code availability

- **Yes, official — [MOTOR code release](https://github.com/som-shahlab/motor_code_release)**  
  The maintained ecosystem is the FEMR project.

## Python package availability

- **Yes, official — [FEMR](https://pypi.org/project/femr/)**  
  Install with `pip install femr`; current FEMR documentation supports training and fine-tuning MOTOR-derived models.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# TransformEHR

> **Encoder-decoder generative EHR model**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Pretrain by generating all codes expected at a future visit, then reuse the learned longitudinal representation for low-prevalence disease outcomes.

## Input

Longitudinal visits containing ICD diagnoses and related coded EHR events.

## Architecture

A Transformer encoder-decoder predicts future-visit code sets; downstream tasks attach a disease/outcome prediction head.

## Pretraining objective

Generative prediction of all codes at a future visit from preceding visits.

## Original downstream task(s)

Pancreatic-cancer onset, intentional self-harm among patients with PTSD, and broader disease/outcome prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The encoder-decoder pretrained weights and a task-specific head are fine-tuned end-to-end on labeled outcomes.

## Source paper

- [TransformEHR paper](https://www.nature.com/articles/s41467-023-43715-z)

## Code availability

- **Yes, official — [TransformEHR GitHub repository](https://github.com/whaleloops/TransformEHR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# DuETT

> **Dual Event Time Transformer**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Respect the two-dimensional nature of sparse clinical tables by alternating attention over event types and over time rather than flattening everything into one sequence.

## Input

Sparse numeric clinical time series arranged as event-type by time, including observed values and missingness.

## Architecture

Alternating event-axis and time-axis Transformer blocks encode the 2D structure; task heads support binary and multilabel prediction.

## Pretraining objective

Masked-event modeling that predicts event presence and reconstructs values.

## Original downstream task(s)

Mortality, length of stay, phenotyping, and other MIMIC-IV and PhysioNet tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The original evaluation primarily fine-tunes the full model (Strategy 2); frozen-encoder linear probing has also been evaluated, corresponding to Strategy 3.

## Source paper

- [DuETT paper](https://proceedings.mlr.press/v219/labach23a.html)

## Code availability

- **Yes, official — [DuETT GitHub repository](https://github.com/layer6ai-labs/duett)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# TransEHR

> **Self-supervised Transformer for asynchronous clinical time series**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Design pretext tasks around the asynchronous, sparse nature of clinical measurements rather than borrowing objectives unchanged from regular language sequences.

## Input

Multi-source asynchronous clinical time series with sparse observations and irregular timestamps.

## Architecture

A Transformer encodes asynchronous events and temporal information; downstream classifiers operate after self-supervised initialization.

## Pretraining objective

Three self-supervised pretext tasks tailored to asynchronous observations and temporal structure.

## Original downstream task(s)

In-hospital mortality, phenotyping, and length-of-stay prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The Transformer is subsequently updated with supervised labels for each downstream task.

## Source paper

- [TransEHR paper](https://proceedings.mlr.press/v225/xu23a.html)

## Code availability

- **Yes, official — [TransEHR GitHub repository](https://github.com/SigmaTsing/TransEHR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# Hi-BEHRT

> **Hierarchical Transformer for very long EHR histories**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Scale BEHRT-like modeling to very long patient records using hierarchy and contrastive learning over augmented patient views.

## Input

Long records containing diagnoses, medications, measurements, age, calendar time, and other structured modalities.

## Architecture

A hierarchical Transformer compresses local segments and models long-range history; a task head is added for endpoint prediction.

## Pretraining objective

End-to-end contrastive pretraining over augmented views of a patient’s longitudinal EHR.

## Original downstream task(s)

Five-year heart failure, diabetes, chronic kidney disease, and stroke risk.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained hierarchical Transformer is fully fine-tuned for each labeled risk endpoint.

## Source paper

- [Hi-BEHRT paper](https://arxiv.org/abs/2106.11360)

## Code availability

- **Not found — No verified public authors’ repository found**  
  Targeted public search was checked through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# GenHPF with self-supervised pretraining

> **General Healthcare Predictive Framework — pretrained configuration**  
> **Year:** 2022/2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Combine schema-agnostic hierarchical event serialization with self-supervised learning so one model can transfer across EHR sources and clinical tasks.

## Input

Table names, field names, codes, values, and time intervals serialized as hierarchical structured events.

## Architecture

A hierarchical serializer and Transformer-style sequence encoder are pretrained on event prediction, then paired with task heads.

## Pretraining objective

Self-supervised prediction over serialized hierarchical EHR events.

## Original downstream task(s)

Twelve mortality, readmission, diagnosis, acuity, and related binary or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The self-supervised encoder is fine-tuned end-to-end, including in multi-task supervised configurations.

## Source paper

- [GenHPF with self-supervised pretraining paper](https://arxiv.org/abs/2207.09858)

## Code availability

- **Yes, official — [GenHPF GitHub repository](https://github.com/hoon9405/GenHPF)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# STraTS

> **Self-Supervised Transformer for sparse clinical time series**  
> **Year:** 2022  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent each observed measurement as a time-variable-value triplet, avoiding regular-grid imputation and learning useful temporal structure through forecasting.

## Input

Irregular observation triplets `(time, variable, value)`.

## Architecture

A Transformer embeds time, variable identity, and value for each observation and aggregates sparse sequences into a patient representation.

## Pretraining objective

Self-supervised future time-series/value forecasting from observed triplets.

## Original downstream task(s)

In-hospital mortality on sparse clinical time-series benchmarks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The forecasting-pretrained encoder is subsequently fine-tuned with a supervised mortality head.

## Source paper

- [STraTS paper](https://arxiv.org/abs/2107.14293)

## Code availability

- **Yes, official reimplementation — [STraTS GitHub repository](https://github.com/sindhura97/STraTS)**

## Python package availability

- **Yes, package implementation — [MEDS-torch](https://pypi.org/project/meds-torch/)**  
  Install with `pip install meds-torch`; the package supports STraTS-style value-forecasting pretraining in a general MEDS framework.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# CEHR-BERT

> **Temporally enriched BERT for structured EHR**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Adapt BERT to structured records by explicitly encoding visits, age, elapsed time, and calendar time instead of treating codes as an ordinary text sentence.

## Input

Longitudinal diagnosis, procedure, medication, and other coded concepts with visit, age, elapsed-time, and calendar-time representations.

## Architecture

A BERT encoder receives temporally transformed concept embeddings and visit-boundary information, followed by task-specific classification heads.

## Pretraining objective

Masked-code modeling plus EHR-specific visit-type prediction.

## Original downstream task(s)

Hospitalization, death, incident heart failure, and heart-failure readmission.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained BERT and a task-specific classification head are fine-tuned using labeled downstream examples.

## Source paper

- [CEHR-BERT paper](https://proceedings.mlr.press/v158/pang21a.html)

## Code availability

- **Yes, official — [CEHR-BERT GitHub repository](https://github.com/cumc-dbmi/cehrbert)**

## Python package availability

- **Yes, official — [cehrbert](https://pypi.org/project/cehrbert/)**  
  Install with `pip install cehrbert`. The companion ETL package is [`cehrbert-data`](https://pypi.org/project/cehrbert-data/).

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# CLMBR

> **Clinical Language Model-Based Representations**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn one general patient representation by forecasting future structured events, then keep that encoder fixed so many small labeled tasks can use inexpensive classical probes.

## Input

Chronological sequences of diagnoses, procedures, medications, laboratory orders/results, and encounters.

## Architecture

An autoregressive clinical sequence model produces a fixed patient representation at each prediction time; downstream models consume extracted features.

## Pretraining objective

Autoregressive clinical language modeling that predicts future structured events from patient history.

## Original downstream task(s)

Five binary outcomes in the original study, with later use on additional tasks.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** Canonical CLMBR usage freezes the pretrained encoder, materializes patient representations, and trains logistic regression or gradient-boosted probes.

## Source paper

- [CLMBR paper](https://arxiv.org/abs/2001.05295)

## Code availability

- **Yes, official — [ehr_ml GitHub repository](https://github.com/som-shahlab/ehr_ml)**

## Python package availability

- **Source-installable, official — [ehr_ml repository package](https://github.com/som-shahlab/ehr_ml)**  
  The repository is structured as an installable Python package, but a current dedicated PyPI release was not verified. Some historical FEMR versions also supported CLMBR.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# Med-BERT

> **Pretrained contextualized embeddings for structured EHR**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Transfer BERT-style contextual learning to diagnosis-code visits, with an auxiliary patient-level objective intended to improve disease-risk prediction.

## Input

Longitudinal visits represented as ICD diagnosis-code sequences or sets, plus code-order, visit, and demographic embeddings.

## Architecture

A BERT encoder contextualizes codes across visits; a classification head is added for target diseases.

## Pretraining objective

Masked-code modeling plus prediction of whether a patient ever had a prolonged hospital stay.

## Original downstream task(s)

Heart failure among patients with diabetes and pancreatic-cancer onset.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** Both the pretrained Med-BERT encoder and the newly added prediction head are updated end-to-end on labeled tasks.

## Source paper

- [Med-BERT paper](https://www.nature.com/articles/s41746-021-00455-y)

## Code availability

- **Yes, official — [Med-BERT GitHub repository](https://github.com/ZhiGroup/Med-BERT)**

## Python package availability

- **No — No dedicated pip-installable package found**  
  The official repository supplies scripts and model code rather than a verified published package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# BEHRT

> **Transformer for Electronic Health Records**  
> **Year:** 2020  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent longitudinal diagnosis histories as BERT-like sequences enriched with age and visit context, enabling one pretrained encoder to support many future-disease endpoints.

## Input

Diagnosis-code histories with visit, age, segment, and positional encodings; extensions can add medications and measurements.

## Architecture

A BERT-style Transformer contextualizes medical-code sequences and feeds a classification layer for future conditions.

## Pretraining objective

Masked diagnosis-code prediction.

## Original downstream task(s)

Future onset of 301 conditions; each condition can be treated as a binary endpoint.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained Transformer and a new classification layer are jointly fine-tuned on labeled outcomes.

## Source paper

- [BEHRT paper](https://www.nature.com/articles/s41598-020-62922-y)

## Code availability

- **Yes, official — [BEHRT GitHub repository](https://github.com/deepmedicine/BEHRT)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

---

# G-BERT

> **Graph-augmented Transformer pretraining for medication recommendation**  
> **Year:** 2019  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use medical ontologies to improve code representations and self-supervised cross-type prediction to connect diagnoses with medications before longitudinal adaptation.

## Input

Diagnosis and medication sets, visit structure, longitudinal visits, and structured code ontologies.

## Architecture

Graph neural networks encode ontology hierarchies; a Transformer-style visit encoder learns cross-type relations and is adapted over patient histories.

## Pretraining objective

Graph-based code representation learning plus single-visit self-supervised cross-type code prediction.

## Original downstream task(s)

Multilabel medication recommendation; each medication output is a binary decision.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained code and visit representations are fully fine-tuned on labeled longitudinal medication-recommendation data.

## Source paper

- [G-BERT paper](https://arxiv.org/abs/1906.00346)

## Code availability

- **Yes, official — [G-BERT GitHub repository](https://github.com/jshang123/g-bert)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.

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

