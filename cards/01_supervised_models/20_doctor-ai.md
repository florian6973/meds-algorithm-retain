# Doctor AI

> **Predicting clinical events with recurrent neural networks**  
> **Year:** 2016  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Treat a patient record as a clinical sequence and learn to forecast the next encounter’s content and timing in one recurrent model.

## Input

Timestamped diagnosis, medication, and procedure histories.

## Architecture

A GRU-based sequence model predicts next-visit codes and the time to the next encounter.

## Original downstream task(s)

Next-visit diagnoses, medications, and time to next encounter; individual code outputs are binary sigmoid targets.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The recurrent model is learned directly from the supervised next-event targets in the task dataset.

## Source paper

- [Doctor AI paper](https://proceedings.mlr.press/v56/Choi16.html)

## Code availability

- **Yes, official — [Doctor AI GitHub repository](https://github.com/mp2893/doctorai)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
