# TransEHR

> **Self-supervised Transformer for asynchronous clinical time series**  
> **Year:** 2023  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Design pretext tasks around the asynchronous, sparse nature of clinical measurements rather than borrowing objectives unchanged from regular language sequences.

## Input

Multi-source asynchronous clinical time series with sparse observations and irregular timestamps.

## Architecture

A Transformer encodes asynchronous events and temporal information; downstream classifiers operate after self-supervised initialization.

## Pretraining objective

Three self-supervised pretext tasks tailored to asynchronous observations and temporal structure.

## Original downstream task(s)

In-hospital mortality, phenotyping, and length-of-stay prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The Transformer is subsequently updated with supervised labels for each downstream task.

## Source paper

- [TransEHR paper](https://proceedings.mlr.press/v225/xu23a.html)

## Code availability

- **Yes, official — [TransEHR GitHub repository](https://github.com/SigmaTsing/TransEHR)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
