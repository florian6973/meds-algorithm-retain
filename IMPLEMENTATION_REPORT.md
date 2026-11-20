# Implementation report: RETAIN

## Source

- Paper/repository: https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html and https://github.com/mp2893/retain@9fd39c46e44b3fdae3a667a47fbdd5ffbd87bff5
- Exact revision/version: TODO

## Scientific assumptions and decisions

- Mode: confirmation required (default). If exploratory mode was explicitly enabled, record that instruction here.
- Unresolved blocking questions: TODO (or explicitly none).
- Provisional choices still requiring review before a faithful-reproduction claim: TODO (or explicitly none).

Record choices before implementing dependent behavior. Statuses: routine default, awaiting confirmation,
provisional, confirmed, or resolved from source. Link evidence and explicit decisions; do not treat silence
as approval. Revisit affected code, tests and results when a decision changes.

| Missing/conflicting detail | Sources consulted | Options and scientific consequences | Choice/rationale and status | User decision or source resolution | Affected code/tests/results |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## Profile choices

### supervised

- Lifecycle justification: TODO
- Implemented profile methods, algorithm-owned configuration and payload formats: TODO
- Frozen/fitted parameters and permitted label access: TODO
- Local and MEDS-DEV evidence for this profile: TODO



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

### Primary behavior

- Signal and expected behavior: TODO
- How the signal enters the algorithm's selected codes/features: TODO
- Example generator used or adaptation made: TODO
- Positive-control result: TODO
- Shuffled-label negative-control result: TODO

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | TODO | TODO |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | TODO | TODO |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | TODO | TODO |
| Full target dataset (name/revision: TODO) | TODO | TODO | TODO |

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicates, demo plan | TODO | TODO |
| Before full target dataset | Demo result, target-dataset/task/predicate compatibility, access, full-run resources | TODO | TODO |

## Resource measurements

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| TODO | TODO | TODO | TODO | TODO |

## Environment and provenance

- Algorithm commit and dirty state: TODO
- Dataset/task identifiers: TODO
- Dataset, task, and predicate revisions: TODO
- Task-bundle kind (clinical ACES tasks / sampled code-by-horizon tasks) and the commit of the tool that built it: TODO
- Shared task-bundle identifier/path: TODO
- Digest of the bundle's task definitions: TODO
- Cluster checkout commit and job IDs: TODO
- Structured-result branch/PR: TODO
- Dependency/platform/hardware summary: TODO

## Known limitations

TODO
