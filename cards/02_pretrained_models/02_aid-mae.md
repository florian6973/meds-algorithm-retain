# AID-MAE

> **Augmented-Intrinsic Dual-Masked Autoencoder**  
> **Year:** 2026  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Learn from sparse clinical tables without imputing naturally missing observations, using both real missingness and synthetic masking as complementary learning signals.

## Input

Observed laboratory and vital-sign value–timestamp tokens from sparse, irregular tables; naturally missing entries are not first imputed.

## Architecture

A masked-autoencoder encoder reconstructs values under an intrinsic missingness mask and an additional stochastic mask; downstream heads use the learned encoder.

## Pretraining objective

Dual-masked autoencoding over naturally sparse and additionally masked observations.

## Original downstream task(s)

Mortality, acute kidney injury, and binary length-of-stay endpoints on MIMIC-IV and PhysioNet 2012.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The paper evaluates both full encoder fine-tuning with a binary MLP head (Strategy 2) and a frozen encoder followed by logistic-regression probing (Strategy 3).

## Source paper

- [AID-MAE paper](https://arxiv.org/abs/2602.15159)

## Code availability

- **Not found — No public model-specific repository found**  
  No code link was verified from the paper page or targeted public search through 2026-07-30.

## Python package availability

- **No — No dedicated Python package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
