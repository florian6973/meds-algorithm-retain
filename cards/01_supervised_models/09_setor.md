# SETOR

> **Sequential diagnosis prediction with ontological representation**  
> **Year:** 2021  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Combine visit sequences with a medical ontology and continuous-time dynamics so predictions reflect both semantic code relationships and irregular disease progression.

## Input

Sequential diagnosis codes, code-ontology relations, visit structure, and time intervals.

## Architecture

A Transformer-based visit encoder is paired with ontology-aware code representations and neural-ODE/continuous-time modeling.

## Original downstream task(s)

Sequential diagnosis prediction; individual diagnoses use binary sigmoid outputs or one-vs-rest evaluation.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The complete ontology-aware temporal network is fitted on downstream diagnosis labels.

## Source paper

- [SETOR paper](https://arxiv.org/abs/2109.03069)

## Code availability

- **Yes, official — [SETOR GitHub repository](https://github.com/Xueping/SETOR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
