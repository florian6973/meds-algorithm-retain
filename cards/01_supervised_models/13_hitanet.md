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
