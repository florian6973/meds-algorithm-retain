# SurvivEHR

> **Competing-risks, time-to-event foundation model for primary-care EHRs**  
> **Year:** 2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Pretrain one longitudinal model to forecast many possible future clinical events and their timing, then adapt that general temporal representation to longer-horizon prognostic endpoints. The central idea is that routine primary-care records contain competing future events, so pretraining should model *what happens next and when* rather than only masked or next-token identity.

## Input

Large longitudinal primary-care event streams containing diagnoses, medications, investigations and measurements, timestamps, and numerical values when available. The accompanying FastEHR pipeline performs scalable extraction, tokenization, filtering, and leakage-aware cohort construction.

## Architecture

A Transformer-based longitudinal foundation model with a competing-risks time-to-event output formulation. Its pretraining head models future event types and timing across heterogeneous primary-care records; downstream prognostic heads are initialized from the pretrained model and optimized on labeled endpoints.

## Pretraining objective

Competing-risks, time-to-event forecasting over many future diagnoses, investigations, medications, measurements, and mortality-related events.

## Original downstream task(s)

Longer-horizon prognostic prediction for multiple long-term conditions, including evaluation in low-label and transfer settings.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The reported downstream workflow initializes task-specific prognostic models from the pretrained SurvivEHR backbone and updates them using labeled endpoint data. The pretraining hazards alone are not the primary downstream evaluation pathway.

## Source paper

- [SurvivEHR — npj Digital Medicine](https://www.nature.com/articles/s41746-026-02709-z)

## Code availability

- **Yes, official — [SurvivEHR](https://github.com/cwlgadd/SurvivEHR)**
- **Yes, official preprocessing — [FastEHR](https://github.com/cwlgadd/FastEHR)**

The paper states that the source used for preprocessing, training, evaluation, and experiment reproduction is public.

## Python package availability

- **Source-installable research software, not a verified PyPI distribution.**

The repositories are structured Python projects, but no dedicated, published `survivehr` package was verified on PyPI as of the check date.

## Interpretation notes

- SurvivEHR is conceptually close to MOTOR and later marked time-to-event objectives, but its primary-care scale, mixed event types, and competing-risks formulation distinguish it.
- The classification reflects the paper’s downstream adaptation experiments; alternative reuse of pretrained event hazards could yield a zero-shot workflow, but that is not the safest primary label.
