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
