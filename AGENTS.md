# Repository instructions

This is the development repository for **meds-model-retain**: [florian6973/meds-model-retain](https://github.com/florian6973/meds-model-retain).
Its default branch is `main`; its implementation is at the repository root.

## Development routing (effective 2026-10-08)

All further model development belongs in the corresponding private `florian6973/meds-model-*`
repository listed below. The original `florian6973/meds-models` repository is the shared reference
and coordination repository; its `model/*` branches are historical migration sources.

- Before changing model code, clone or open the matching private repository, fetch its `main`, and
  verify that the push remote names that repository. Use a feature branch there when appropriate.
- Commit and push model changes and open new model pull requests in that model's repository.
  Do not continue model implementation on `meds-models` or its historical `model/*` branches.
- Open new model development issues in the matching model repository; retain links to historical
  issues when they contain relevant decisions. Existing issues have not been transferred.
- Keep all model repositories private unless the user explicitly requests a visibility change.
- Work at the root of the matching model repository. Read its `AGENTS.md`, `SPEC.md`,
  `README.md`, and implementation guide before editing it.
- Copies of sibling models in an inherited branch are historical context. Develop each sibling in
  its own mapped repository, not in another model's copied directory.
- When preparing new clone, installation, run, or synchronization commands, use the model's new
  repository and a revision from that repository. Inspect inherited scripts for old repository or
  branch defaults before running them. Preserve historical source URLs and recorded run revisions.
- The branch-based routing in `DGX_AGENTIC_WORKFLOW.md` and inherited model documentation is
  historical and is superseded by this routing. Existing scientific validation, data-access,
  privacy, and cluster-approval requirements still apply.

The MEDS-Tab branch was migrated at its current large configuration. There was no separate
MEDS-TAB-tiny branch, so this mapping does not designate a separate tiny repository.

| Historical branch | Development repository | Model directory |
|---|---|---|
| `model/00_01_meds_tab` | [florian6973/meds-model-meds-tab-large](https://github.com/florian6973/meds-model-meds-tab-large) | `.` (repository root) |
| `model/00_02_meds_tab_lr` | [florian6973/meds-model-meds-tab-lr](https://github.com/florian6973/meds-model-meds-tab-lr) | `.` (repository root) |
| `model/01_01_teco` | [florian6973/meds-model-teco](https://github.com/florian6973/meds-model-teco) | `.` (repository root) |
| `model/01_19_retain` | [florian6973/meds-model-retain](https://github.com/florian6973/meds-model-retain) | `.` (repository root) |
| `model/01_21_icu_xgboost` | [florian6973/meds-model-icu-xgboost](https://github.com/florian6973/meds-model-icu-xgboost) | `.` (repository root) |
| `model/02_09_motor` | [florian6973/meds-model-motor](https://github.com/florian6973/meds-model-motor) | `.` (repository root) |
| `model/02_09_motor_cp` | [florian6973/meds-model-motor-cp](https://github.com/florian6973/meds-model-motor-cp) | `.` (repository root) |
| `model/02_09_motor_ft` | [florian6973/meds-model-motor-ft](https://github.com/florian6973/meds-model-motor-ft) | `.` (repository root) |
| `model/02_09_motor_lp` | [florian6973/meds-model-motor-lp](https://github.com/florian6973/meds-model-motor-lp) | `.` (repository root) |
| `model/02_11_duett` | [florian6973/meds-model-duett](https://github.com/florian6973/meds-model-duett) | `.` (repository root) |
| `model/02_19_behrt` | [florian6973/meds-model-behrt](https://github.com/florian6973/meds-model-behrt) | `.` (repository root) |
| `model/03_07_meds_eic_ar` | [florian6973/meds-model-meds-eic-ar](https://github.com/florian6973/meds-model-meds-eic-ar) | `.` (repository root) |
| `model/03_07_meds_eic_ar_sup` | [florian6973/meds-model-meds-eic-ar-sup](https://github.com/florian6973/meds-model-meds-eic-ar-sup) | `.` (repository root) |
| `model/03_08_ora` | [florian6973/meds-model-ora](https://github.com/florian6973/meds-model-ora) | `.` (repository root) |
| `model/03_10_icarefm` | [florian6973/meds-model-icarefm](https://github.com/florian6973/meds-model-icarefm) | `.` (repository root) |

## Model implementation instructions

Before editing, read `SPEC.md`, `README.md`, and `docs/IMPLEMENTATION_GUIDE.md` completely.

- `src/meds_model_contract/` is the backend-neutral contract. Do not add framework, batching,
  preprocessing-format, or model assumptions there.
- `src/retain/`, `model.yaml`, command configuration, and `tests/e2e.py` are model-owned.
- Preserve the `supervised` model profile and established command argument names.
- Intermediate payloads are opaque. Generic tests validate their envelopes, not their contents.
- MEDS-DEV owns dataset-specific predicates; the model owns their interpretation.
- Never associate outputs with prediction keys by row order.
- Keep `IS_STUB = True` until every registered command and the real end-to-end adapter work.
- Replace the intentional failure in `tests/test_primary_behavior.py`. Prove that a known signal reaches
  the model's real selected codes/features and produces the expected behavior, with a shuffled-label
  negative control; adapt the example signal dataset rather than forcing an incompatible representation.
- Run pytest with `-rs`, account for every skip, and require zero `model_stub` skips before completion.
- Follow the ordered gates in `docs/IMPLEMENTATION_GUIDE.md`. Stop for human approval before the MIMIC-IV
  demo and again before the full MIMIC-IV run; do not infer data access or cluster authorization.
- Submit cluster jobs only from the recorded SHA of a clean, committed, pushed revision. Use the approved
  external full-data `work/` task bundle, record its registry digest, and synchronize structured results.
- Report exact source revisions, commands, coverage, and measurements in `IMPLEMENTATION_REPORT.md`.
- Do not commit datasets, checkpoints, credentials, or raw cluster logs.
