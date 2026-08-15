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
