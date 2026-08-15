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
