# GCT

> **Graph Convolutional Transformer**  
> **Year:** 2020  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Regularize learned attention with empirical EHR co-occurrence structure, combining flexible Transformer attention with an inductive graph bias.

## Input

Sets or sequences of structured EHR concepts and empirical concept co-occurrence relationships.

## Architecture

Transformer attention is constrained or regularized by a learned graph-convolutional structure over EHR concepts.

## Original downstream task(s)

Readmission and other patient-level outcomes; native binary classification.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The graph-constrained Transformer and task head are trained end-to-end using labeled outcomes.

## Source paper

- [GCT paper](https://arxiv.org/abs/1906.04716)

## Code availability

- **Yes, third-party reproduction — [GCT reproduction repository](https://github.com/lycpaul/dl4h-gp30-gct)**  
  No authors’ repository was verified; this link is a reproduction rather than an official release.

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
