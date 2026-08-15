# STraTS

> **Self-Supervised Transformer for sparse clinical time series**  
> **Year:** 2022  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Represent each observed measurement as a time-variable-value triplet, avoiding regular-grid imputation and learning useful temporal structure through forecasting.

## Input

Irregular observation triplets `(time, variable, value)`.

## Architecture

A Transformer embeds time, variable identity, and value for each observation and aggregates sparse sequences into a patient representation.

## Pretraining objective

Self-supervised future time-series/value forecasting from observed triplets.

## Original downstream task(s)

In-hospital mortality on sparse clinical time-series benchmarks.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The forecasting-pretrained encoder is subsequently fine-tuned with a supervised mortality head.

## Source paper

- [STraTS paper](https://arxiv.org/abs/2107.14293)

## Code availability

- **Yes, official reimplementation — [STraTS GitHub repository](https://github.com/sindhura97/STraTS)**

## Python package availability

- **Yes, package implementation — [MEDS-torch](https://pypi.org/project/meds-torch/)**  
  Install with `pip install meds-torch`; the package supports STraTS-style value-forecasting pretraining in a general MEDS framework.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
