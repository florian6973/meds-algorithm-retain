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
