# Repository instructions

This is the development repository for **meds-model-retain**: [florian6973/meds-model-retain](https://github.com/florian6973/meds-model-retain).
Its default branch is `main`; its implementation is in `models/01_19_retain`.

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
- Migration preserved the directory layout: work in the `models/...` directory listed below,
  and read its `AGENTS.md`, `SPEC.md`, `README.md`, and implementation guide before editing it.
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
| `model/00_01_meds_tab` | [florian6973/meds-model-meds-tab-large](https://github.com/florian6973/meds-model-meds-tab-large) | `models/00_01_meds_tab` |
| `model/00_02_meds_tab_lr` | [florian6973/meds-model-meds-tab-lr](https://github.com/florian6973/meds-model-meds-tab-lr) | `models/00_02_meds_tab_lr` |
| `model/01_01_teco` | [florian6973/meds-model-teco](https://github.com/florian6973/meds-model-teco) | `models/01_01_teco` |
| `model/01_19_retain` | [florian6973/meds-model-retain](https://github.com/florian6973/meds-model-retain) | `models/01_19_retain` |
| `model/01_21_icu_xgboost` | [florian6973/meds-model-icu-xgboost](https://github.com/florian6973/meds-model-icu-xgboost) | `models/01_21_icu_xgboost` |
| `model/02_09_motor` | [florian6973/meds-model-motor](https://github.com/florian6973/meds-model-motor) | `models/02_09_motor` |
| `model/02_09_motor_cp` | [florian6973/meds-model-motor-cp](https://github.com/florian6973/meds-model-motor-cp) | `models/02_09_motor_cp` |
| `model/02_09_motor_ft` | [florian6973/meds-model-motor-ft](https://github.com/florian6973/meds-model-motor-ft) | `models/02_09_motor_ft` |
| `model/02_09_motor_lp` | [florian6973/meds-model-motor-lp](https://github.com/florian6973/meds-model-motor-lp) | `models/02_09_motor_lp` |
| `model/02_11_duett` | [florian6973/meds-model-duett](https://github.com/florian6973/meds-model-duett) | `models/02_11_duett` |
| `model/02_19_behrt` | [florian6973/meds-model-behrt](https://github.com/florian6973/meds-model-behrt) | `models/02_19_behrt` |
| `model/03_07_meds_eic_ar` | [florian6973/meds-model-meds-eic-ar](https://github.com/florian6973/meds-model-meds-eic-ar) | `models/03_07_meds_eic_ar` |
| `model/03_07_meds_eic_ar_sup` | [florian6973/meds-model-meds-eic-ar-sup](https://github.com/florian6973/meds-model-meds-eic-ar-sup) | `models/03_07_meds_eic_ar_sup` |
| `model/03_08_ora` | [florian6973/meds-model-ora](https://github.com/florian6973/meds-model-ora) | `models/03_08_ora` |
| `model/03_10_icarefm` | [florian6973/meds-model-icarefm](https://github.com/florian6973/meds-model-icarefm) | `models/03_10_icarefm` |
