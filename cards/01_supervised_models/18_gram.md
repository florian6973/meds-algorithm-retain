# GRAM

> **Graph-based Attention Model**  
> **Year:** 2017  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Share statistical strength from clinical ontologies, especially for rare codes, by representing each code as an attention-weighted combination of itself and its ancestors.

## Input

Medical-code sequences plus ancestor relationships from a clinical ontology.

## Architecture

Ontology-aware attention builds code embeddings from ancestors; a recurrent patient-history encoder and prediction head model future diagnoses or risk.

## Original downstream task(s)

Diagnosis and heart-failure prediction; heart failure is a native binary task.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** Ontology embeddings and the downstream predictor are trained jointly from labeled patient trajectories.

## Source paper

- [GRAM paper](https://arxiv.org/abs/1611.07012)

## Code availability

- **Yes, official — [GRAM GitHub repository](https://github.com/mp2893/gram)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
