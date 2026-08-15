# SAnD

> **Simply Attend and Diagnose**  
> **Year:** 2018  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Replace recurrence with masked self-attention for clinical time series, enabling direct access to long-range dependencies and parallel training.

## Input

Regular-grid structured ICU time series with positional information.

## Architecture

Masked self-attention layers and positional encoding produce sequence features; dense interpolation converts variable-length sequences into a fixed-size representation.

## Original downstream task(s)

MIMIC-III mortality, decompensation, length of stay, and phenotyping; binary and multilabel outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The attention encoder is trained from scratch for the selected supervised clinical task.

## Source paper

- [SAnD paper](https://arxiv.org/abs/1711.03905)

## Code availability

- **Yes, third-party implementation — [SAnD GitHub repository](https://github.com/khirotaka/SAnD)**  
  This is a public implementation; an authors’ official repository was not verified.

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
