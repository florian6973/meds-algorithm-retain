# CEHR-BERT

> **Temporally enriched BERT for structured EHR**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Adapt BERT to structured records by explicitly encoding visits, age, elapsed time, and calendar time instead of treating codes as an ordinary text sentence.

## Input

Longitudinal diagnosis, procedure, medication, and other coded concepts with visit, age, elapsed-time, and calendar-time representations.

## Architecture

A BERT encoder receives temporally transformed concept embeddings and visit-boundary information, followed by task-specific classification heads.

## Pretraining objective

Masked-code modeling plus EHR-specific visit-type prediction.

## Original downstream task(s)

Hospitalization, death, incident heart failure, and heart-failure readmission.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained BERT and a task-specific classification head are fine-tuned using labeled downstream examples.

## Source paper

- [CEHR-BERT paper](https://proceedings.mlr.press/v158/pang21a.html)

## Code availability

- **Yes, official — [CEHR-BERT GitHub repository](https://github.com/cumc-dbmi/cehrbert)**

## Python package availability

- **Yes, official — [cehrbert](https://pypi.org/project/cehrbert/)**  
  Install with `pip install cehrbert`. The companion ETL package is [`cehrbert-data`](https://pypi.org/project/cehrbert-data/).

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
