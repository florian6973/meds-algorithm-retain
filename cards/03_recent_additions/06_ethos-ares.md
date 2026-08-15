# ETHOS / ARES

> **Enhanced Transformer for Health Outcome Simulation / Adaptive Risk Estimation System**  
> **Year:** 2024/2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Represent the entire patient record as a specialized “Patient Health Timeline” language, autoregressively complete that timeline into the future, and estimate clinical risks from the frequency of target events across repeated completions. ARES adds dynamic risk estimation and patient-specific explanations around the ETHOS generator.

## Input

Tokenized chronological Patient Health Timelines derived from heterogeneous MIMIC-IV and MIMIC-IV-ED events, including admissions, demographics, diagnoses, procedures, medications, measurements, and explicit elapsed-time tokens. The current reimplementation uses MEDS as an intermediate representation.

## Architecture

A GPT-2-style decoder-only Transformer trained over the specialized health-event token language. The maintained ETHOS/ARES implementation currently uses a bias-free GPT-2 architecture and provides tokenization, training, trajectory inference, and risk-analysis commands.

## Pretraining objective

Autoregressive next-token prediction over Patient Health Timelines.

## Original downstream task(s)

Zero-shot inpatient mortality, 30-day readmission, ICU admission, prolonged length of stay, composite endpoints, and related trajectory-based risk estimates. ARES additionally provides individualized explanations of estimated risk.

## Pipeline classification

- **Primary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** ETHOS repeatedly generates future Patient Health Timelines. The probability of an endpoint is the fraction of generated futures in which the target event occurs within the specified stopping rule or horizon.

## Source paper

- [ETHOS — npj Digital Medicine](https://www.nature.com/articles/s41746-024-01235-0)
- [ARES — GigaScience](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf107/8268904)

## Code availability

- **Yes, official — [ipolharvard/ethos-ares](https://github.com/ipolharvard/ethos-ares)**
- Historical paper implementation: [ipolharvard/ethos-paper](https://github.com/ipolharvard/ethos-paper)

## Python package availability

- **Yes, source-installable official package.**

The repository has a `pyproject.toml`, installs with `pip install -e .`, and exposes commands including `ethos_tokenize`, `ethos_train`, and `ethos_infer`. No separate PyPI distribution was verified.

## Interpretation notes

- Although the papers use “zero-shot,” the concrete risk DAG is Strategy 5 because generated futures are materialized and aggregated.
- ARES is best understood as a risk-estimation and explanation layer over ETHOS rather than an independently pretrained backbone.
