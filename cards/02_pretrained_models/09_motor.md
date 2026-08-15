# MOTOR

> **Many Outcome Time Oriented Representations**  
> **Year:** 2024  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Scale supervision automatically by treating many codes as time-to-event outcomes, producing reusable temporal representations and calibrated event-risk functions.

## Input

Timestamped diagnosis, procedure, medication, laboratory, encounter, and claims-event sequences.

## Architecture

A temporal Transformer-like foundation model is pretrained over many automatically generated time-to-event objectives and adapted to labeled survival endpoints.

## Pretraining objective

Large-scale self-supervised or weakly supervised time-to-event prediction over many automatically defined event outcomes.

## Original downstream task(s)

Nineteen time-to-event tasks; fixed-horizon cumulative risk can be used as a binary prediction.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The many-outcome pretrained weights are adapted with labeled task-specific survival/time-to-event training. The output family is survival/TTE rather than an ordinary single-logit classifier.

## Source paper

- [MOTOR paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html)

## Code availability

- **Yes, official — [MOTOR code release](https://github.com/som-shahlab/motor_code_release)**  
  The maintained ecosystem is the FEMR project.

## Python package availability

- **Yes, official — [FEMR](https://pypi.org/project/femr/)**  
  Install with `pip install femr`; current FEMR documentation supports training and fine-tuning MOTOR-derived models.

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
