# EveryQuery

> **Zero-Shot Clinical Prediction via Task-Conditioned Pretraining over Electronic Health Records**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Make the clinical prediction task itself part of the model input. Instead of pretraining a generic autoregressive model and later generating many futures, EveryQuery is pretrained across randomly sampled clinical questions so that one fixed model can answer new `(outcome code, prediction horizon)` queries directly.

## Input

A MEDS-formatted patient history up to a prediction time plus a structured query specifying an outcome code and a future duration/window. The current query language focuses on questions such as: “Will code `c` occur within `d` days?”

## Architecture

A ModernBERT-style bidirectional encoder over tokenized MEDS events, augmented with query and duration representations and a prediction head that emits occurrence and censoring probabilities.

## Pretraining objective

Task-conditioned supervised pretraining over millions of automatically generated query–context pairs. Queries and prediction times are sampled from the EHR, and labels are generated automatically from future event occurrence and censoring.

## Original downstream task(s)

Thirty-nine randomly sampled MIMIC-IV prediction tasks, including tasks held out from the pretraining query distribution. The paper also analyzes limitations on compound/disjunctive outcomes such as readmission.

## Pipeline classification

- **Primary strategy:** **4. Direct zero-shot**
  - DAG: `preprocess_data → pretrain → predict`

**Rationale.** At evaluation time, the pretrained model consumes a patient history and a new structured query and emits the requested probability in one deterministic forward pass. It uses no task-specific fine-tuning, separate probe, or generated trajectories.

## Source paper

- [EveryQuery — arXiv](https://arxiv.org/abs/2603.07900)
- [EveryQuery — OpenReview](https://openreview.net/forum?id=c2TbMN7wT0)

## Code availability

- **Yes, official — [payalchandak/EveryQuery](https://github.com/payalchandak/EveryQuery)**

The repository includes MEDS preprocessing, task generation, training, prediction, and evaluation commands.

## Python package availability

- **Yes, official — [`EveryQuery`](https://pypi.org/project/EveryQuery/)**
- Install with `pip install EveryQuery`.

## Interpretation notes

- EveryQuery is the cleanest current example of Strategy 4 in this collection.
- “Zero-shot” here has the stronger DAG meaning: direct endpoint prediction without materializing trajectories or representations.
- The query language is currently narrower than arbitrary clinical phenotypes, so support for complex logical endpoints should not be assumed.
