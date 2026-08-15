# PORTER

> **Language-grounded event representations for portable structured EHR foundation models**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Ground structured event names in natural-language descriptions so one temporal model can transfer across institutions and schemas without relying on clinical notes.

## Input

Timestamped conditions, procedures, medications, measurements, encounters, numeric values, reference ranges, and demographics; event names are encoded by a frozen text encoder.

## Architecture

A frozen text-based input pathway feeds a 115M-parameter causal temporal Transformer pretrained autoregressively on next-event prediction. Downstream tasks use extracted representations.

## Pretraining objective

Autoregressive next-event prediction; the pretraining output head is discarded for adaptation.

## Original downstream task(s)

Seventy-four binary tasks covering outcomes, laboratory abnormalities, procedures, imaging, medications, and transfusions.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The pretrained backbone and input pathway remain frozen; representations are materialized and an L2-logistic-regression probe is trained on labeled endpoints.

## Source paper

- [PORTER paper](https://arxiv.org/abs/2606.24102)

## Code availability

- **Not found — No public model-specific repository found**  
  No code link was verified from the paper page or targeted public search through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
