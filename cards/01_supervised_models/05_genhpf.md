# GenHPF

> **General Healthcare Predictive Framework — supervised/journal configuration**  
> **Year:** 2024  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use one schema-agnostic hierarchical serialization so the same predictor can consume heterogeneous EHR tables and tasks with minimal hand-written domain mapping.

## Input

Structured events serialized hierarchically from table/event type, field names, field values, and time intervals across differing EHR schemas.

## Architecture

A hierarchical event serializer and Transformer-style sequence encoder feed task-specific heads; the framework supports multi-source and multi-task training.

## Original downstream task(s)

Twelve mortality, readmission, diagnosis, acuity, and related binary or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** This document describes the supervised configuration listed in the first table: task labels directly train the model. The same paper also studies self-supervised pretraining, documented separately in the pretrained-model folder.

## Source paper

- [GenHPF paper](https://arxiv.org/abs/2207.09858)

## Code availability

- **Yes, official — [GenHPF GitHub repository](https://github.com/hoon9405/GenHPF)**

## Python package availability

- **No — No dedicated published Python package found**  
  The repository is reusable research code, but no model-specific PyPI distribution was verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
