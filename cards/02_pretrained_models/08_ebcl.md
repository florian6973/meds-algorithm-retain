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
