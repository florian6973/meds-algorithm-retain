# DuETT

> **Dual Event Time Transformer**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Respect the two-dimensional nature of sparse clinical tables by alternating attention over event types and over time rather than flattening everything into one sequence.

## Input

Sparse numeric clinical time series arranged as event-type by time, including observed values and missingness.

## Architecture

Alternating event-axis and time-axis Transformer blocks encode the 2D structure; task heads support binary and multilabel prediction.

## Pretraining objective

Masked-event modeling that predicts event presence and reconstructs values.

## Original downstream task(s)

Mortality, length of stay, phenotyping, and other MIMIC-IV and PhysioNet tasks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The original evaluation primarily fine-tunes the full model (Strategy 2); frozen-encoder linear probing has also been evaluated, corresponding to Strategy 3.

## Source paper

- [DuETT paper](https://proceedings.mlr.press/v219/labach23a.html)

## Code availability

- **Yes, official — [DuETT GitHub repository](https://github.com/layer6ai-labs/duett)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
