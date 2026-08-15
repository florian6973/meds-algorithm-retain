# TOO-BERT

> **Trajectory Order Objective BERT**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Teach a BERT-style EHR encoder not only which codes fit locally, but also whether an entire clinical trajectory is in a plausible temporal order.

## Input

Longitudinal diagnosis, procedure, medication, and encounter-code trajectories with explicit visit and code order.

## Architecture

A Transformer encoder combines masked-code learning with a trajectory-order discriminator and is later paired with a classification head.

## Pretraining objective

Masked-code modeling plus discrimination of correctly ordered trajectories from clinically informed permutations.

## Original downstream task(s)

Heart failure, Alzheimer disease, and prolonged length of stay.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained Transformer and classification head are jointly updated during supervised downstream fine-tuning.

## Source paper

- [TOO-BERT paper](https://medinform.jmir.org/2025/1/e68138)

## Code availability

- **Yes, official — [TOO-BERT GitHub repository](https://github.com/ali-amirahmadii/TOO-Bert)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
