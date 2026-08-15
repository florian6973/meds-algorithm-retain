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
