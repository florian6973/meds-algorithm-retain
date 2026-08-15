# Chet

> **Context-Aware Health Event Prediction via Transition Functions**  
> **Year:** 2022  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Model disease-state evolution as transitions between dynamic graphs, so the representation explicitly captures how clinical context changes from visit to visit.

## Input

Diagnosis-code histories converted into visit-specific dynamic disease graphs.

## Architecture

Graph encoders construct disease-state representations and learned transition functions propagate context across visits before prediction.

## Original downstream task(s)

Next-visit diagnosis and heart-failure prediction; heart failure is a native binary task.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The graph-transition model is optimized directly against labeled future-event or heart-failure outcomes.

## Source paper

- [Chet paper](https://arxiv.org/abs/2112.05195)

## Code availability

- **Yes, official — [Chet GitHub repository](https://github.com/LuChang-CS/Chet)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
