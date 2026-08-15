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
