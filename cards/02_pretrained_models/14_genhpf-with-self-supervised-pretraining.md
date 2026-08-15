# GenHPF with self-supervised pretraining

> **General Healthcare Predictive Framework — pretrained configuration**  
> **Year:** 2022/2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Combine schema-agnostic hierarchical event serialization with self-supervised learning so one model can transfer across EHR sources and clinical tasks.

## Input

Table names, field names, codes, values, and time intervals serialized as hierarchical structured events.

## Architecture

A hierarchical serializer and Transformer-style sequence encoder are pretrained on event prediction, then paired with task heads.

## Pretraining objective

Self-supervised prediction over serialized hierarchical EHR events.

## Original downstream task(s)

Twelve mortality, readmission, diagnosis, acuity, and related binary or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The self-supervised encoder is fine-tuned end-to-end, including in multi-task supervised configurations.

## Source paper

- [GenHPF with self-supervised pretraining paper](https://arxiv.org/abs/2207.09858)

## Code availability

- **Yes, official — [GenHPF GitHub repository](https://github.com/hoon9405/GenHPF)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
