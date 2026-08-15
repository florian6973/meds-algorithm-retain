# CLMBR

> **Clinical Language Model-Based Representations**  
> **Year:** 2021  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn one general patient representation by forecasting future structured events, then keep that encoder fixed so many small labeled tasks can use inexpensive classical probes.

## Input

Chronological sequences of diagnoses, procedures, medications, laboratory orders/results, and encounters.

## Architecture

An autoregressive clinical sequence model produces a fixed patient representation at each prediction time; downstream models consume extracted features.

## Pretraining objective

Autoregressive clinical language modeling that predicts future structured events from patient history.

## Original downstream task(s)

Five binary outcomes in the original study, with later use on additional tasks.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** Canonical CLMBR usage freezes the pretrained encoder, materializes patient representations, and trains logistic regression or gradient-boosted probes.

## Source paper

- [CLMBR paper](https://arxiv.org/abs/2001.05295)

## Code availability

- **Yes, official — [ehr_ml GitHub repository](https://github.com/som-shahlab/ehr_ml)**

## Python package availability

- **Source-installable, official — [ehr_ml repository package](https://github.com/som-shahlab/ehr_ml)**  
  The repository is structured as an installable Python package, but a current dedicated PyPI release was not verified. Some historical FEMR versions also supported CLMBR.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
