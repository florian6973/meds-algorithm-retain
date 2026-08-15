# MEDS-EIC-AR

> **MEDS “Everything-is-code” Autoregressive Model**  
> **Year:** 2025–2026  
> **Collection:** Recent additions identified after the original two tables  
> **Availability checked:** 2026-08-03

## Model philosophy

Provide a reusable, MEDS-native autoregressive baseline in which heterogeneous structured EHR events—including codes, values, and timing information—are serialized into one causal token stream. The same model can be pretrained on any MEDS cohort and used to generate patient futures for zero-shot evaluation.

## Input

MEDS event streams transformed with an “everything-is-code” serialization and tensorized through the MEDS ecosystem. Inputs may include diagnoses, medications, procedures, laboratory values, visits, and explicit temporal/event tokens, depending on the source MEDS dataset and configuration.

## Architecture

A configurable causal Transformer implemented with PyTorch Lightning and Hugging Face causal-language-model components. The current generation backend wraps `LlamaForCausalLM.generate`; Hydra configurations expose model sizes, tokenization, training, and trajectory-generation settings.

## Pretraining objective

Autoregressive next-token prediction over tokenized MEDS patient timelines.

## Original downstream task(s)

Repository-packaged zero-shot generation for arbitrary MEDS task cohorts. For each labeled prediction time, the model generates multiple future trajectories in a pseudo-MEDS trajectory schema for later endpoint resolution.

## Pipeline classification

- **Current documented strategy:** **5. Materialized zero-shot**
  - DAG: `preprocess_data → pretrain → infer → predict`

**Rationale.** The package’s documented workflow generates and saves multiple future trajectories before converting them to endpoint probabilities. At the check date, trajectory generation is implemented, while the repository’s built-in “resolve trajectories into predictions” stage is explicitly marked not yet implemented.

### Plausible extension, not currently packaged

A classification head could be attached to the pretrained hidden state and the backbone updated on downstream labels:

`preprocess_data → pretrain → supervised_train → predict`

That would make **MEDS-EIC-AR-FT** a Strategy-2 model, but it should not be presented as an existing supported workflow until a supervised fine-tuning command and evaluation are added.

## Source paper

- **No dedicated model paper was identified.** The primary technical specification is the official repository/package documentation.
- [Official repository](https://github.com/mmcdermott/MEDS_EIC_AR)

## Code availability

- **Yes, official — [mmcdermott/MEDS_EIC_AR](https://github.com/mmcdermott/MEDS_EIC_AR)**

## Python package availability

- **Yes, official — [MEDS-EIC-AR on PyPI](https://pypi.org/project/MEDS-EIC-AR/)**
- Install with `pip install MEDS-EIC-AR`.

## Reproducibility caveat

The repository warns that its tests establish execution and basic numerical sanity, not reliable convergence or full scientific correctness. This caveat should remain visible in benchmark documentation.
