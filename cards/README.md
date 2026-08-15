# Structured-EHR model cards

This directory contains **49 per-model Markdown documents**. The original 40 cards were derived from the two supplied tables; the `03_recent_additions/` directory adds models identified in a targeted 2025–2026 update through **2026-08-03**.

- `01_supervised_models/`: 20 primarily supervised-from-scratch models.
- `02_pretrained_models/`: 20 models from the supplied pretraining/adaptation table.
- `03_recent_additions/`: 9 recent or previously omitted models/workflows.
- `RECENT_ADDITIONS.md`: only the new cards, concatenated.
- `ALL_MODELS.md`: all cards concatenated for full-text search.
- `model_index.json`: machine-readable index.

Each card includes philosophy, input, architecture, objectives/tasks, source paper, code availability, Python-package availability, and one or more pipeline classifications.

## Classification legend

| Strategy | DAG | Meaning |
|---|---|---|
| **1. Simple supervised** | `preprocess_data → preprocess_task → supervised_train → predict` | Train a supervised model from scratch on one task. |
| **2. Pretrain + fine-tune** | `preprocess_data → pretrain → supervised_train → predict` | Initialize downstream supervised training from a pretrained model. |
| **3. Pretrained probe** | `preprocess_data → pretrain → infer → supervised_train → predict` | Freeze the pretrained model, materialize representations, and train a separate probe. |
| **4. Direct zero-shot** | `preprocess_data → pretrain → predict` | Let the pretrained model directly emit task predictions. |
| **5. Materialized zero-shot** | `preprocess_data → pretrain → infer → predict` | Materialize scores, trajectories, hazards, or similar outputs and convert them to predictions. |
| **6. PFN-style** | `preprocess_data → preprocess_task → predict` | Use a fixed pretrained model in context, with no optimizer-based training command for the new task. |

## Updated classification result

The expanded set now contains concrete examples of **all six strategies**:

- **Strategy 1:** RETAIN and the other task-specific supervised models.
- **Strategy 2:** CEHR-BERT, MOTOR, SurvivEHR, and CEHR-XGPT fine-tuning.
- **Strategy 3:** CLMBR, PORTER, ORA, and CEHR-XGPT linear probing.
- **Strategy 4:** EveryQuery and Delphi-2M’s direct rate-to-risk pathway.
- **Strategy 5:** ETHOS/ARES, CEHR-XGPT trajectory inference, CoMET, and MEDS-EIC-AR.
- **Strategy 6:** SurvPFN, with the scope caveat that it is tabular survival rather than a longitudinal EHR-sequence model.

**Important correction:** CEHR-XGPT’s reported zero-shot clinical-risk procedure is Strategy 5, because it generates 50 future trajectories and aggregates them. It is not Strategy 4 under this DAG taxonomy.

## Model index

| Family | # | Model | Year | Strategy | Python package status |
|---|---:|---|---|---|---|
| Supervised | 1 | [TECO](01_supervised_models/01_teco.md) | 2025 | 1. Simple supervised | No — No dedicated Python package found |
| Supervised | 2 | [STraTS-mTAND](01_supervised_models/02_strats-mtand.md) | 2025 | 1. Simple supervised | No — No dedicated Python package found |
| Supervised | 3 | [TRANS](01_supervised_models/03_trans.md) | 2024 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 4 | [PRISM](01_supervised_models/04_prism.md) | 2024 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 5 | [GenHPF](01_supervised_models/05_genhpf.md) | 2024 | 1. Simple supervised | No — No dedicated published Python package found |
| Supervised | 6 | [KerPrint](01_supervised_models/06_kerprint.md) | 2023 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 7 | [Chet](01_supervised_models/07_chet.md) | 2022 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 8 | [UniHPF](01_supervised_models/08_unihpf.md) | 2022 | 1. Simple supervised | No — No dedicated published Python package found |
| Supervised | 9 | [SETOR](01_supervised_models/09_setor.md) | 2021 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 10 | [GRASP](01_supervised_models/10_grasp.md) | 2021 | 1. Simple supervised | Yes, third-party — PyHealth |
| Supervised | 11 | [GCT](01_supervised_models/11_gct.md) | 2020 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 12 | [StageNet](01_supervised_models/12_stagenet.md) | 2020 | 1. Simple supervised | Yes, third-party — PyHealth |
| Supervised | 13 | [HiTANet](01_supervised_models/13_hitanet.md) | 2020 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 14 | [AdaCare](01_supervised_models/14_adacare.md) | 2020 | 1. Simple supervised | Yes, third-party — PyHealth |
| Supervised | 15 | [ConCare](01_supervised_models/15_concare.md) | 2020 | 1. Simple supervised | Yes, third-party — PyHealth |
| Supervised | 16 | [SAnD](01_supervised_models/16_sand.md) | 2018 | 1. Simple supervised | No — No dedicated pip-installable package found |
| Supervised | 17 | [Dipole](01_supervised_models/17_dipole.md) | 2017 | 1. Simple supervised | Historical only — Archived PyHealth implementation |
| Supervised | 18 | [GRAM](01_supervised_models/18_gram.md) | 2017 | 1. Simple supervised | No — No dedicated maintained pip package found |
| Supervised | 19 | [RETAIN](01_supervised_models/19_retain.md) | 2016 | 1. Simple supervised | Yes, third-party — PyHealth |
| Supervised | 20 | [Doctor AI](01_supervised_models/20_doctor-ai.md) | 2016 | 1. Simple supervised | No — No dedicated maintained pip package found |
| Pretrained | 1 | [PORTER](02_pretrained_models/01_porter.md) | 2026 | 3. Pretrained probe | No — No dedicated Python package found |
| Pretrained | 2 | [AID-MAE](02_pretrained_models/02_aid-mae.md) | 2026 | 2. Pretrain + fine-tune, 3. Pretrained probe | No — No dedicated Python package found |
| Pretrained | 3 | [HealthFormer](02_pretrained_models/03_healthformer.md) | 2026 | 2. Pretrain + fine-tune | No — No dedicated published Python package found |
| Pretrained | 4 | [PULSE-ICU](02_pretrained_models/04_pulse-icu.md) | 2025 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Pretrained | 5 | [BAT foundation-model configuration](02_pretrained_models/05_bat-foundation-model-configuration.md) | 2025 | 2. Pretrain + fine-tune, 3. Pretrained probe | No dedicated model package — Research framework only |
| Pretrained | 6 | [TOO-BERT](02_pretrained_models/06_too-bert.md) | 2025 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Pretrained | 7 | [CORE-BEHRT](02_pretrained_models/07_core-behrt.md) | 2024 | 2. Pretrain + fine-tune | No dedicated model package — No verified CORE-BEHRT PyPI distribution |
| Pretrained | 8 | [EBCL](02_pretrained_models/08_ebcl.md) | 2024 | 2. Pretrain + fine-tune | Yes, package implementation — MEDS-torch |
| Pretrained | 9 | [MOTOR](02_pretrained_models/09_motor.md) | 2024 | 2. Pretrain + fine-tune | Yes, official — FEMR |
| Pretrained | 10 | [TransformEHR](02_pretrained_models/10_transformehr.md) | 2023 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Pretrained | 11 | [DuETT](02_pretrained_models/11_duett.md) | 2023 | 2. Pretrain + fine-tune, 3. Pretrained probe | No — No dedicated published Python package found |
| Pretrained | 12 | [TransEHR](02_pretrained_models/12_transehr.md) | 2023 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Pretrained | 13 | [Hi-BEHRT](02_pretrained_models/13_hi-behrt.md) | 2023 | 2. Pretrain + fine-tune | No — No dedicated Python package found |
| Pretrained | 14 | [GenHPF with self-supervised pretraining](02_pretrained_models/14_genhpf-with-self-supervised-pretraining.md) | 2022/2023 | 2. Pretrain + fine-tune | No — No dedicated published Python package found |
| Pretrained | 15 | [STraTS](02_pretrained_models/15_strats.md) | 2022 | 2. Pretrain + fine-tune | Yes, package implementation — MEDS-torch |
| Pretrained | 16 | [CEHR-BERT](02_pretrained_models/16_cehr-bert.md) | 2021 | 2. Pretrain + fine-tune | Yes, official — cehrbert |
| Pretrained | 17 | [CLMBR](02_pretrained_models/17_clmbr.md) | 2021 | 3. Pretrained probe | Source-installable, official — ehr_ml repository package |
| Pretrained | 18 | [Med-BERT](02_pretrained_models/18_med-bert.md) | 2021 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Pretrained | 19 | [BEHRT](02_pretrained_models/19_behrt.md) | 2020 | 2. Pretrain + fine-tune | No — No dedicated maintained pip package found |
| Pretrained | 20 | [G-BERT](02_pretrained_models/20_g-bert.md) | 2019 | 2. Pretrain + fine-tune | No — No dedicated pip-installable package found |
| Recent addition | 1 | [SurvivEHR](03_recent_additions/01_survivehr.md) | 2026 | 2. Pretrain + fine-tune | Source-installable, official — SurvivEHR/FastEHR repositories |
| Recent addition | 2 | [EveryQuery](03_recent_additions/02_everyquery.md) | 2026 | 4. Direct zero-shot | Yes, official — EveryQuery on PyPI |
| Recent addition | 3 | [CEHR-XGPT](03_recent_additions/03_cehr-xgpt.md) | 2025 | 2. Pretrain + fine-tune, 3. Pretrained probe, 5. Materialized zero-shot | Source-installable, official — cehrgpt repository |
| Recent addition | 4 | [CoMET](03_recent_additions/04_comet.md) | 2025 | 5. Materialized zero-shot | No official package found |
| Recent addition | 5 | [Delphi-2M](03_recent_additions/05_delphi-2m.md) | 2025 | 4. Direct zero-shot, 5. Materialized zero-shot | No published package — official research repository and controlled checkpoint |
| Recent addition | 6 | [ETHOS / ARES](03_recent_additions/06_ethos-ares.md) | 2024/2025 | 5. Materialized zero-shot | Source-installable, official — ethos-ares package |
| Recent addition | 7 | [MEDS-EIC-AR](03_recent_additions/07_meds-eic-ar.md) | 2025–2026 | 5. Materialized zero-shot | Yes, official — MEDS-EIC-AR on PyPI |
| Recent addition | 8 | [ORA](03_recent_additions/08_ora.md) | 2026 | 3. Pretrained probe | No official package found |
| Recent addition | 9 | [SurvPFN](03_recent_additions/09_survpfn.md) | 2026 | 6. PFN-style | Source-installable, official — pretrained checkpoint on Hugging Face |

## Availability methodology

- **Official code** means an authors’ or project repository was identified.
- **Third-party code** means a reproduction or toolkit implementation was identified but not an original-author release.
- **Python package** is reserved for a PyPI distribution or a clearly reusable source-installable library with package structure/API. Scripts, notebooks, or a requirements file alone count as code, not a package.
- Availability for the new cards was checked on **2026-08-03**. Repository, checkpoint, and package status can change.
- Strategy labels describe the concrete reported workflow. A single architecture can have several cards/branches when it supports fine-tuning, probing, and trajectory-based zero-shot inference.
