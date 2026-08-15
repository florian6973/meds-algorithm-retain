# ORA

> **One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Treat the EHR as a marked temporal process rather than a plain sequence of discrete codes. ORA is principally a pretraining objective: it jointly models when a future code occurs and the associated numerical value when the event is a measurement, aiming to learn representations useful across classification, survival, and regression.

## Input

Irregular structured-EHR sequences containing heterogeneous discrete events and numeric measurements, with event codes, timestamps, and values. Experiments use MIMIC-IV and Columbia University Irving Medical Center data.

## Architecture

ORA is backbone-agnostic and is evaluated with both Transformer and Mamba sequence backbones. A code-specific marked time-to-event prediction head discretizes future time and value distributions while supporting codes without numerical marks.

## Pretraining objective

A marked time-to-event objective that jointly models each medical code’s future occurrence time and associated numerical value, while handling censoring, concurrent events, and mixed event types.

## Original downstream task(s)

Seven binary classification tasks, four time-to-event tasks—including AMI, MASLD, stroke, and celiac disease where available—and four laboratory-value regression tasks.

## Pipeline classification

- **Primary strategy:** **3. Pretrained probe**
  - DAG: `preprocess_data → pretrain → infer → supervised_train → predict`

**Rationale.** In the reported evaluation, pretrained models generate task-cohort embeddings and separate task-specific heads are fit: logistic regression for classification, DeepHit for time-to-event prediction, and linear regression for numerical outcomes. The downstream evaluation therefore uses frozen representations rather than end-to-end fine-tuning.

## Source paper

- [ORA — arXiv](https://arxiv.org/abs/2602.00541)
- [ORA — OpenReview](https://openreview.net/forum?id=WnxEI2t9be)

## Code availability

- **No dedicated official repository was verified in a targeted search.**

The paper is sufficiently detailed to identify the objective and evaluation pathway, but this card should not claim a public implementation until the authors link one explicitly.

## Python package availability

- **No dedicated Python package found.**

## Interpretation notes

- ORA is best viewed as an objective/family of pretrained models rather than one fixed architecture.
- If a future implementation updates the backbone with labeled downstream data, that branch would additionally qualify as Strategy 2.
