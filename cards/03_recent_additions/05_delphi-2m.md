# Delphi-2M

> **Learning the natural history of human disease with generative transformers**  
> **Year:** 2025  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Model the competing progression of many diseases on a continuous age axis. Rather than building one endpoint model per disease, Delphi-2M predicts disease-specific incidence rates for more than 1,000 diagnoses simultaneously and can also sample complete future disease trajectories.

## Input

Longitudinal top-level ICD-10 diagnosis tokens plus age, sex, body-mass index, smoking, alcohol consumption, and death. Events are represented on continuous age rather than ordinary discrete token positions.

## Architecture

A modified GPT-2 causal Transformer with continuous-age sinusoidal encodings and a competing-exponential output formulation. The logits correspond to per-day event rates for 1,256 disease/death tokens; an additional time-to-next-event component is learned through the same competing-risks formulation.

## Pretraining objective

Joint autoregressive prediction of the next disease token and its waiting time under a competing exponential model.

## Original downstream task(s)

Simultaneous disease-incidence prediction for more than 1,000 diseases, external validation on Danish registry data without parameter updates, long-horizon disease-risk forecasting, and synthetic future-trajectory generation.

## Pipeline classification

- **Primary strategy:** **4. Direct zero-shot**
  - DAG: `preprocess_data → pretrain → predict`
  - A single model pass produces disease-specific log-rates. A fixed-horizon probability can be deterministically computed from a rate, for example `1 - exp(-rate × horizon)`.

- **Secondary strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`
  - When the model samples future disease trajectories and downstream burdens are calculated from those samples.

**Rationale.** Delphi-2M supports both direct hazard/rate-based prediction and trajectory simulation. The card separates those two workflows rather than assigning one ambiguous “zero-shot” label.

## Source paper

- [Delphi-2M — Nature](https://www.nature.com/articles/s41586-025-09529-3)

## Code availability

- **Yes, official — [gerstung-lab/Delphi](https://github.com/gerstung-lab/Delphi)**

The repository includes training code, evaluation scripts, and trajectory-sampling notebooks.

## Python package availability

- **No published reusable Python package.**

The official release is a research repository installed through a requirements file rather than a packaged library API.

## Checkpoint availability

- The useful Delphi-2M checkpoint is available through UK Biobank controlled-access procedures.
- The repository’s openly available synthetic/demo data and model are intended for software testing, not substantive research conclusions.

## Interpretation notes

- Delphi-2M is narrower in event vocabulary than full hospital EHR models: it focuses mainly on disease progression and selected baseline/lifestyle variables.
- Its direct-risk mode makes it a stronger Strategy-4 example than trajectory-only generative models.
