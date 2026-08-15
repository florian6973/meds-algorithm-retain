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
