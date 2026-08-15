# CoMET

> **Cosmos Medical Event Transformer**  
> **Year:** 2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Learn a general simulator of longitudinal healthcare activity at very large scale, then answer heterogeneous clinical and operational questions by sampling plausible patient futures rather than fitting a new model for each endpoint.

## Input

Large longitudinal event streams from Epic Cosmos, including encounter headers and types, diagnoses, medications, procedures, laboratory-result events, specialties, and other discrete medical events represented as multi-token sequences.

## Architecture

A family of decoder-only autoregressive Transformers from approximately 62 million to 1 billion parameters. The paper studies scaling behavior across model size, pretraining tokens, and compute.

## Pretraining objective

Generic autoregressive next-medical-event prediction over 115 billion medical events from 118 million patients.

## Original downstream task(s)

Seventy-eight clinical and operational tasks, including diagnosis prediction, disease prognosis, laboratory thresholds, encounter utilization, and healthcare operations. Predictions are derived from repeated simulated futures.

## Pipeline classification

- **Primary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** CoMET first generates multiple future medical-event trajectories. Task probabilities, distributions, or expected quantities are then computed from those generated futures without task-specific fine-tuning.

## Source paper

- [Generative Medical Event Models Improve with Scale — arXiv](https://arxiv.org/abs/2508.12104)

## Code availability

- **No verified official public training or inference repository.**

A third-party “code replica” repository exists, but it should not be treated as the authors’ implementation. The Epic Cosmos training data are also not publicly reproducible.

## Python package availability

- **No official Python package found.**

## Interpretation notes

- CoMET is a canonical large-scale Strategy-5 example but a poor first implementation choice because the model, checkpoints, and source training data are not openly available.
- Some immediate next-event analyses may inspect token probabilities, but the paper’s broad downstream evaluation is simulation-based and therefore materialized zero-shot.
