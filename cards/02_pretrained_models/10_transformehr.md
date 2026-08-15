# TransformEHR

> **Encoder-decoder generative EHR model**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Pretrain by generating all codes expected at a future visit, then reuse the learned longitudinal representation for low-prevalence disease outcomes.

## Input

Longitudinal visits containing ICD diagnoses and related coded EHR events.

## Architecture

A Transformer encoder-decoder predicts future-visit code sets; downstream tasks attach a disease/outcome prediction head.

## Pretraining objective

Generative prediction of all codes at a future visit from preceding visits.

## Original downstream task(s)

Pancreatic-cancer onset, intentional self-harm among patients with PTSD, and broader disease/outcome prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The encoder-decoder pretrained weights and a task-specific head are fine-tuned end-to-end on labeled outcomes.

## Source paper

- [TransformEHR paper](https://www.nature.com/articles/s41467-023-43715-z)

## Code availability

- **Yes, official — [TransformEHR GitHub repository](https://github.com/whaleloops/TransformEHR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
