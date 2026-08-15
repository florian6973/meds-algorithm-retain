# Dipole

> **Attention-based bidirectional recurrent diagnosis predictor**  
> **Year:** 2017  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use both forward and backward contextualization of prior visits, then attend to the most relevant visits when forecasting the next diagnoses.

## Input

Sequences of visits represented by diagnosis or other medical-code vectors.

## Architecture

A bidirectional recurrent network encodes visits and a visit-level attention mechanism forms the context used for next-visit prediction.

## Original downstream task(s)

Next-visit diagnosis prediction; individual codes are binary sigmoid outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The bidirectional RNN, attention, and diagnosis head are trained directly on next-visit labels.

## Source paper

- [Dipole paper](https://arxiv.org/abs/1706.05764)

## Code availability

- **Yes, third-party/archived toolkit — [PyHealth repository](https://github.com/sunlabuiuc/PyHealth)**  
  A standalone original-author repository was not verified; older PyHealth releases included a Dipole implementation.

## Python package availability

- **Historical only — [Archived PyHealth implementation](https://pyhealth.readthedocs.io/en/archived/)**  
  Older PyHealth APIs exposed Dipole, but it is not listed as a current stable PyHealth 2.0 model. Pin and validate an older release before use.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
