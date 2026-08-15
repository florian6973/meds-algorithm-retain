# HealthFormer

> **Dual-level time-aware Transformer for structured EHR**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Pretrain at both code and event levels while explicitly modeling elapsed time, aiming to transfer across incident-disease horizons from irregular administrative EHR sequences.

## Input

ICD-10 diagnoses, procedures, ATC medications, encounter types, contextual fields, and elapsed times in irregular event sequences.

## Architecture

A dual-level time-aware Transformer represents codes within events and events across the longitudinal trajectory, then adds a sigmoid endpoint head.

## Pretraining objective

Multi-task self-supervision: masked code prediction, whole-event masking/event-type prediction, next-event-type prediction, and time-to-next-event regression.

## Original downstream task(s)

Incident colorectal and prostate cancer at 30-, 60-, and 90-day horizons.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The self-supervised Transformer and new endpoint head are updated end-to-end on labeled downstream examples.

## Source paper

- [HealthFormer paper](https://www.medrxiv.org/content/10.64898/2026.03.25.26349262v2.full-text)

## Code availability

- **Yes, official — [HealthFormer GitHub repository](https://github.com/renyi-ai/HealthFormer)**

## Python package availability

- **No — No dedicated published Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
