# BEHRT

> **Transformer for Electronic Health Records**  
> **Year:** 2020  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent longitudinal diagnosis histories as BERT-like sequences enriched with age and visit context, enabling one pretrained encoder to support many future-disease endpoints.

## Input

Diagnosis-code histories with visit, age, segment, and positional encodings; extensions can add medications and measurements.

## Architecture

A BERT-style Transformer contextualizes medical-code sequences and feeds a classification layer for future conditions.

## Pretraining objective

Masked diagnosis-code prediction.

## Original downstream task(s)

Future onset of 301 conditions; each condition can be treated as a binary endpoint.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained Transformer and a new classification layer are jointly fine-tuned on labeled outcomes.

## Source paper

- [BEHRT paper](https://www.nature.com/articles/s41598-020-62922-y)

## Code availability

- **Yes, official — [BEHRT GitHub repository](https://github.com/deepmedicine/BEHRT)**

## Python package availability

- **No — No dedicated maintained pip package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
