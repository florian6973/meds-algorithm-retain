# Med-BERT

> **Pretrained contextualized embeddings for structured EHR**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Transfer BERT-style contextual learning to diagnosis-code visits, with an auxiliary patient-level objective intended to improve disease-risk prediction.

## Input

Longitudinal visits represented as ICD diagnosis-code sequences or sets, plus code-order, visit, and demographic embeddings.

## Architecture

A BERT encoder contextualizes codes across visits; a classification head is added for target diseases.

## Pretraining objective

Masked-code modeling plus prediction of whether a patient ever had a prolonged hospital stay.

## Original downstream task(s)

Heart failure among patients with diabetes and pancreatic-cancer onset.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** Both the pretrained Med-BERT encoder and the newly added prediction head are updated end-to-end on labeled tasks.

## Source paper

- [Med-BERT paper](https://www.nature.com/articles/s41746-021-00455-y)

## Code availability

- **Yes, official — [Med-BERT GitHub repository](https://github.com/ZhiGroup/Med-BERT)**

## Python package availability

- **No — No dedicated pip-installable package found**  
  The official repository supplies scripts and model code rather than a verified published package.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
