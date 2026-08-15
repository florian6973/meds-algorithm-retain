# Hi-BEHRT

> **Hierarchical Transformer for very long EHR histories**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Scale BEHRT-like modeling to very long patient records using hierarchy and contrastive learning over augmented patient views.

## Input

Long records containing diagnoses, medications, measurements, age, calendar time, and other structured modalities.

## Architecture

A hierarchical Transformer compresses local segments and models long-range history; a task head is added for endpoint prediction.

## Pretraining objective

End-to-end contrastive pretraining over augmented views of a patient’s longitudinal EHR.

## Original downstream task(s)

Five-year heart failure, diabetes, chronic kidney disease, and stroke risk.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained hierarchical Transformer is fully fine-tuned for each labeled risk endpoint.

## Source paper

- [Hi-BEHRT paper](https://arxiv.org/abs/2106.11360)

## Code availability

- **Not found — No verified public authors’ repository found**  
  Targeted public search was checked through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
