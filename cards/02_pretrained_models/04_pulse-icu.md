# PULSE-ICU

> **Pretrained Unified Long-Sequence Encoder for ICU**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Model long, unresampled ICU event streams directly, preserving irregular timing and event metadata while sharing one pretrained encoder across many ICU tasks.

## Input

Unresampled laboratory, vital-sign, medication, fluid, procedure, unit, value, timestamp, and event-metadata sequences.

## Architecture

A Longformer-based event encoder handles long ICU timelines; task-specific heads are attached for multi-task downstream prediction.

## Pretraining objective

Masked event and masked value prediction.

## Original downstream task(s)

Eighteen mortality, intervention-forecasting, phenotype, and other binary, multiclass, or multilabel tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained encoder is adapted with supervised multi-task fine-tuning and task-specific prediction heads.

## Source paper

- [PULSE-ICU paper](https://arxiv.org/abs/2511.22199)

## Code availability

- **Yes, official — [PULSE-ICU GitHub repository](https://github.com/sejeongak/PULSE-ICU)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
