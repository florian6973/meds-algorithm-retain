# UniHPF

> **Universal Healthcare Predictive Framework**  
> **Year:** 2022  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Avoid manually harmonizing every EHR schema by turning field names and values themselves into a universal hierarchical language of structured events.

## Input

Event types, field names, field values, and elapsed-time tokens from heterogeneous EHR systems.

## Architecture

A hierarchical Transformer encodes fields within events and events across time, with task heads for diverse clinical endpoints.

## Original downstream task(s)

Mortality, readmission, length of stay, acuity, discharge, and diagnosis; binary and multilabel outputs.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Despite its general architecture, the table entry describes supervised task training rather than a frozen pretrained foundation model.

## Source paper

- [UniHPF paper](https://arxiv.org/abs/2211.08082)

## Code availability

- **Yes, official — [UniHPF GitHub repository](https://github.com/hoon9405/UniHPF)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
