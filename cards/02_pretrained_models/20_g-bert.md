# G-BERT

> **Graph-augmented Transformer pretraining for medication recommendation**  
> **Year:** 2019  
> **Collection:** Pretrained/adaptation models from the second table  
> **Availability checked:** 2026-07-30

## Model philosophy

Use medical ontologies to improve code representations and self-supervised cross-type prediction to connect diagnoses with medications before longitudinal adaptation.

## Input

Diagnosis and medication sets, visit structure, longitudinal visits, and structured code ontologies.

## Architecture

Graph neural networks encode ontology hierarchies; a Transformer-style visit encoder learns cross-type relations and is adapted over patient histories.

## Pretraining objective

Graph-based code representation learning plus single-visit self-supervised cross-type code prediction.

## Original downstream task(s)

Multilabel medication recommendation; each medication output is a binary decision.

## Pipeline classification

- **Primary strategy:** **2. Pretrain + fine-tune**
  - DAG: `preprocess_data → pretrain → supervised_train → predict`

**Rationale.** The pretrained code and visit representations are fully fine-tuned on labeled longitudinal medication-recommendation data.

## Source paper

- [G-BERT paper](https://arxiv.org/abs/1906.00346)

## Code availability

- **Yes, official — [G-BERT GitHub repository](https://github.com/jshang123/g-bert)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
