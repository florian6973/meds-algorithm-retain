# GRASP

> **Generic framework using similar-patient knowledge**  
> **Year:** 2021  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn latent patient prototypes and propagate information among similar patients, allowing a task backbone to benefit from cohort structure rather than treating records independently.

## Input

Structured longitudinal variables suitable for a task-specific backbone, plus learned patient-similarity structure.

## Architecture

A base clinical encoder produces patient states; clustering/prototypes and graph propagation refine them before a prediction head.

## Original downstream task(s)

Patient health-status and clinical-outcome prediction; evaluated endpoints include native binary tasks.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The patient encoder, prototypes, graph module, and endpoint head are trained for the supervised task.

## Source paper

- [GRASP paper](https://ojs.aaai.org/index.php/AAAI/article/view/16152)

## Code availability

- **Yes, official — [GRASP GitHub repository](https://github.com/choczhang/GRASP)**

## Python package availability

- **Yes, third-party — [PyHealth](https://pypi.org/project/pyhealth/)**  
  Install with `pip install pyhealth`; PyHealth provides `pyhealth.models.GRASP`. This is a toolkit implementation, not the original authors’ standalone package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
