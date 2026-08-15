# CORE-BEHRT

> **Carefully Optimized and Rigorously Evaluated BEHRT**  
> **Year:** 2024  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Treat EHR foundation-model performance as an engineering and evaluation problem: systematically optimize masking, time representations, architecture, and training before claiming gains.

## Input

Longitudinal structured EHR sequences containing diagnoses, medications, timestamps, ages, and encounter information.

## Architecture

An optimized BEHRT-style Transformer with revised temporal encoding, masking, architecture, and training choices.

## Pretraining objective

BEHRT-style masked-event prediction with systematically optimized design and training procedure.

## Original downstream task(s)

Death, pain treatment, infection, and 25 additional predominantly binary clinical prediction tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** Pretrained weights are typically fine-tuned end-to-end with a task-specific endpoint head.

## Source paper

- [CORE-BEHRT paper](https://raw.githubusercontent.com/mlresearch/v252/main/assets/odgaard24a/odgaard24a.pdf)

## Code availability

- **Yes, official — [CORE-BEHRT repository](https://github.com/mikkelfo/CORE-BEHRT)**  
  The repository points users toward the newer BONSAI/EHR framework for continued development.

## Python package availability

- **No dedicated model package — No verified CORE-BEHRT PyPI distribution**  
  The public repositories are installable research frameworks, but a model-specific published package was not verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
