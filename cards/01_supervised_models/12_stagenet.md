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
