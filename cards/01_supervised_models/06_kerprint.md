# KerPrint

> **Local-global knowledge-graph enhanced diagnosis prediction**  
> **Year:** 2023  
> **Collection:** Supervised models from the first table  
> **Availability checked:** 2026-07-30

## Model philosophy

Enrich each patient trajectory with both local patient-specific relations and global medical knowledge, while allowing elapsed time to control which knowledge is relevant.

## Input

Visit sequences augmented with local and global medical knowledge graphs.

## Architecture

Time-aware knowledge attention combines visit representations with graph-derived local and global medical context before future-diagnosis prediction.

## Original downstream task(s)

Future diagnosis prediction; each diagnosis output can be treated as a binary sigmoid target.

## Pipeline classification

- **Primary strategy:** **1. Simple supervised**
  - DAG: `preprocess_data → preprocess_task → supervised_train → predict`

**Rationale.** The knowledge-enhanced network is trained directly on future-diagnosis labels, with no separate generic pretraining stage.

## Source paper

- [KerPrint paper](https://ojs.aaai.org/index.php/AAAI/article/view/25667/25439)

## Code availability

- **Yes, official — [KerPrint GitHub repository](https://github.com/xyxpku/KerPrint)**

## Python package availability

- **No — No dedicated pip-installable package found**

## Interpretation notes

- The strategy label describes the workflow reported for this table entry, not every possible way the architecture could be reused.
- “No package found” means no dedicated, publicly verifiable package was located in a targeted search as of 2026-07-30; it is not a proof of nonexistence.
- A package is counted more strongly than a repository only when it is pip-installable or clearly organized as a reusable installable Python library.
