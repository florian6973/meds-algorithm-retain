# BAT foundation-model configuration

> **Self-supervised foundation-model configuration for critical-care time series**  
> **Year:** 2025  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use a missingness-aware set/attention model to learn general ICU dynamics from sparse future-value forecasting before testing how much of the encoder must be updated.

## Input

Sparse multivariate ICU time series with missingness-aware set/attention representations.

## Architecture

The BAT encoder uses attention over sparse observations and missingness structure; a downstream mortality head is attached after forecasting pretraining.

## Pretraining objective

Future-value forecasting with masked mean-squared error.

## Original downstream task(s)

In-hospital mortality.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`
- **Also evaluated:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** The study compares full-model fine-tuning (Strategy 2) with head-only tuning while the pretrained encoder is frozen (a Strategy 3-style probe branch).

## Source paper

- [BAT foundation-model configuration paper](https://arxiv.org/abs/2509.19885)

## Code availability

- **Yes, official project code — [YAIB repository](https://github.com/Katja-Jagd/YAIB)**  
  The study reports its implementation as an extension of the YAIB framework rather than a standalone BAT package.

## Python package availability

- **No dedicated model package — Research framework only**  
  A model-specific pip distribution was not verified.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
