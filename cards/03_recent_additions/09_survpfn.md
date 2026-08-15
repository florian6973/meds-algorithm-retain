# SurvPFN

> **Towards Foundation Models for Survival Predictions**  
> **Year:** 2026  
> **Collection:** Recent additions; tabular survival rather than longitudinal-EHR-specific  
> **Availability checked:** 2026-08-03

## Model philosophy

Amortize survival-model fitting across millions of synthetic tasks. SurvPFN is pretrained once on a prior over censored survival datasets and then solves a new dataset by reading its labeled support rows in context—without gradient-based fitting for that dataset.

## Input

A tabular survival task consisting of training/support rows with feature vectors, observed or censored event indicators, and event/censoring times, together with test/query feature rows. The released checkpoint supports a limited feature-count range and optional categorical-feature indices.

## Architecture

A prior-data fitted Transformer based on NanoTabPFN, extended with a two-route target encoder for observed versus censored rows and a survival-specific output distribution. It follows a scikit-survival-style estimator interface.

## Pretraining objective

Right-censored negative log-likelihood over millions of synthetic survival tasks generated from structural causal models with Weibull event times and non-informative censoring.

## Original downstream task(s)

Zero-shot time-to-event prediction on real-world SurvSet datasets, producing risk scores and survival functions without per-dataset gradient training.

## Pipeline classification

- **Primary strategy:** **6. PFN-style**
  - DAG: `preprocess_data → preprocess_task → predict`

**Rationale.** The pretrained model is fixed. Its `fit` method only stores/encodes the support set; it does not optimize parameters. Predictions for query rows are made in context from the supplied task data.

## Source paper

- [SurvPFN — arXiv](https://arxiv.org/abs/2606.04564)

## Code availability

- **Yes, official — [genepi-freiburg/SurvPFN](https://github.com/genepi-freiburg/SurvPFN)**
- **Pretrained checkpoint — [Hugging Face](https://huggingface.co/samuelboehm/SurvPFN)**

## Python package availability

- **Yes, source-installable official project; no PyPI release verified.**

The repository contains a `pyproject.toml` and a reusable estimator API, but installation is currently repository-based.

## Scope note

SurvPFN is not a longitudinal structured-EHR sequence model. It is included because it is the clearest recent implementation of your Strategy-6 execution semantics for health-related survival prediction. It should therefore be labeled “tabular survival / other” in analyses that restrict the main benchmark to longitudinal MEDS models.
