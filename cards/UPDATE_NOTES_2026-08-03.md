# Update notes — 2026-08-03

## Added cards

- SurvivEHR
- EveryQuery
- CEHR-XGPT
- CoMET
- Delphi-2M
- ETHOS / ARES
- MEDS-EIC-AR
- ORA
- SurvPFN

## Taxonomy corrections

- CEHR-XGPT zero-shot clinical-risk evaluation: **Strategy 5, Materialized zero-shot**, not Strategy 4.
- Delphi-2M is split into a direct disease-rate pathway (**Strategy 4**) and a sampled-trajectory pathway (**Strategy 5**).
- MEDS-EIC-AR is classified by its current documented trajectory-generation workflow (**Strategy 5**). A supervised fine-tuning extension would be Strategy 2 but is not currently a packaged command.
- SurvPFN fills Strategy 6 but is explicitly marked as tabular survival rather than a longitudinal EHR model.
