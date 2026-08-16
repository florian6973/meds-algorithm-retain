# Implementation report: RETAIN

## Source

- Paper/repository: cards/01_supervised_models/19_retain.md, https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html, and https://github.com/mp2893/retain
- Exact revision/version: TODO

## DAG choice

- Selected DAG: `supervised`
- Justification: TODO

## Source-element ledger

| Source element | Status: ported/adapted/omitted | Implementation evidence | Justification/deviation |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Inputs and predicates

- MEDS assumptions: TODO
- Required predicate names: TODO
- Predicate interpretation and derived features: TODO

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Scientific deviations

TODO

## Verification

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | TODO | TODO |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | TODO | TODO |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | TODO | TODO |
| Full MIMIC-IV | TODO | TODO | TODO |

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicates, demo plan | TODO | TODO |
| Before full MIMIC-IV | Demo result, cohort/key coverage, measurements, full-run resources | TODO | TODO |

## Resource measurements

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| TODO | TODO | TODO | TODO | TODO |

## Environment and provenance

- Model commit and dirty state: TODO
- Dataset/task identifiers: TODO
- Dataset, task, and predicate revisions: TODO
- Full-data task-selection repository commit: TODO
- Shared task-bundle identifier/path: TODO
- Task-bundle `tasks.yaml` SHA-256: TODO
- Cluster checkout commit and job IDs: TODO
- Structured-result branch/PR: TODO
- Dependency/platform/hardware summary: TODO

## Known limitations

TODO
