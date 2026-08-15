# TECO

> **Transformer-based Encounter-level Clinical Outcome model**  
> **Year:** 2025  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Update risk continuously as an ICU encounter evolves, using a single encounter-level sequence model rather than separate models for each prediction time.

## Input

Hourly longitudinal laboratory values, vital signs, medications, and baseline patient characteristics.

## Architecture

A Transformer encodes the evolving encounter-level time series and produces repeatedly updated mortality-risk estimates.

## Original downstream task(s)

Continuously updated ICU mortality prediction; native binary endpoint.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The published workflow learns the mortality predictor directly from labeled ICU encounters, without a separate reusable pretraining stage.

## Source paper

- [TECO paper](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091)

## Code availability

- **Not found — No public model-specific repository found**  
  Targeted search checked through 2026-07-30; this is not proof that no private or later repository exists.

## Python package availability

- **No — No dedicated Python package found**  
  A paper implementation repository was not found, so there is also no verified model-specific installable package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
