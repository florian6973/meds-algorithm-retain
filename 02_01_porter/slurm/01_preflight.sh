#!/usr/bin/env bash
# Check a dataset before committing a GPU-day to it. Run on the login node, after env.sh is filled in:
#
#   bash slurm/01_preflight.sh
#
# Nothing here trains. It answers the questions whose wrong answers are expensive and quiet:
#
#   1. Is the cohort split-sharded?          `preprocess_data` refuses otherwise — cheap to find out now.
#   2. Do the codes have DESCRIPTIONS?       This is the one that matters. PORTER's entire input pathway
#                                            is the text of each code's description. A MEDS extract with
#                                            no `description` column still trains, still scores, and is
#                                            not PORTER — it degrades to a model over code strings.
#   3. Do the labels look like MEDS labels?  subject_id / prediction_time / boolean_value, per split.
#   4. How long are the timelines?           Picks `batch_size` against the 16,384-event budget.
#   5. How many codes carry numeric values?  That is how many get quantile bins in the composite target.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

: "${MEDS_ROOT:?set MEDS_ROOT in slurm/env.sh}"
: "${LABELS_DIR:?set LABELS_DIR in slurm/env.sh}"

$PORTER_RUN python - "$MEDS_ROOT" "$LABELS_DIR" <<'PYEOF'
import sys
from pathlib import Path

import polars as pl

from porter.model import render_event_description

meds_root, labels_dir = Path(sys.argv[1]), Path(sys.argv[2])
problems: list[str] = []
notes: list[str] = []


def section(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


section("1. Cohort layout")
if not meds_root.is_dir():
    sys.exit(f"MEDS_ROOT does not exist: {meds_root}")
shards = sorted(p for p in (meds_root / "data").rglob("*.parquet") if not p.name.startswith("."))
splits = sorted({p.relative_to(meds_root / "data").parts[0] for p in shards})
print(f"{len(shards)} shard(s) across {splits}")
if set(splits) - {"train", "tuning", "held_out"}:
    problems.append(
        f"data/ contains {splits}, but preprocess_data requires exactly train/ tuning/ held_out/. "
        "Reshard with a one-stage MEDS-transforms `reshard_to_split` pipeline first."
    )
if "train" not in splits:
    problems.append("No train/ shards: the quantile pre-pass would silently fall back to all splits.")

section("2. Code descriptions — the PORTER check")
codes_fp = meds_root / "metadata" / "codes.parquet"
if not codes_fp.is_file():
    problems.append(f"No {codes_fp}. Every code would fall back to its own identifier as its description.")
else:
    codes = pl.read_parquet(codes_fp)
    n = len(codes)
    if "description" not in codes.columns:
        problems.append(
            f"metadata/codes.parquet has no `description` column ({n} codes). PORTER's input pathway is "
            "the text of that column; without it the model reads code strings and the semantic grounding "
            "the paper is about is simply absent. Fix the ETL, or report results as the code-string arm."
        )
        described = pl.Series([None] * n)
    else:
        described = codes["description"]
        filled = int(described.is_not_null().sum())
        pct = 100 * filled / max(n, 1)
        print(f"{filled}/{n} codes have a description ({pct:.1f}%)")
        if pct < 50:
            problems.append(
                f"Only {pct:.1f}% of codes carry a description. The rest render from their identifier, "
                "so most of the vocabulary reaches the encoder as a code string."
            )
        elif pct < 95:
            notes.append(f"{100 - pct:.1f}% of codes have no description and render from the identifier.")

    print("\nWhat the frozen encoder will actually be handed (10 most frequent codes):")
    order = "code/n_occurrences" if "code/n_occurrences" in codes.columns else "code"
    top = codes.sort(order, descending=order != "code").head(10)
    desc = top["description"].to_list() if "description" in codes.columns else [None] * len(top)
    for code, description in zip(top["code"].to_list(), desc, strict=True):
        print(f"  {code[:44]:<46} -> {render_event_description(code, description)[:70]!r}")

section("3. Labels")
label_files = sorted(p for p in labels_dir.rglob("*.parquet") if not any(x.startswith(".") for x in p.parts))
if not label_files:
    problems.append(f"No parquet files under {labels_dir}.")
else:
    labels = pl.concat([pl.read_parquet(p) for p in label_files], how="diagonal_relaxed")
    print(f"{len(label_files)} file(s), {len(labels)} rows, columns: {labels.columns}")
    missing = {"subject_id", "prediction_time", "boolean_value"} - set(labels.columns)
    if missing:
        problems.append(f"Labels are missing required column(s): {sorted(missing)}")
    else:
        rate = float(labels["boolean_value"].mean() or 0)
        print(f"positive rate: {rate:.4f}  ({int(labels['boolean_value'].sum())} positives)")
        if rate < 0.001 or rate > 0.999:
            notes.append(f"Positive rate {rate:.4f} is extreme; AUROC will be unstable.")

section("4. Timeline lengths (picks batch_size)")
sample = pl.concat([pl.read_parquet(p, columns=["subject_id"]) for p in shards[:4]])
per_subject = sample.group_by("subject_id").len()["len"]
q = {p: int(per_subject.quantile(p / 100)) for p in (50, 90, 99)}
longest = int(per_subject.max())
print(f"measurements per subject (sampled from {min(4, len(shards))} shard(s)): "
      f"median {q[50]}, p90 {q[90]}, p99 {q[99]}, max {longest}")
budget = 16384
suggested = max(1, budget // max(q[99], 1))
print(f"\nmax_seq_len is 16384. p99 is {q[99]}, so `batch_size={suggested}` keeps a batch near the "
      f"paper's {budget}-event budget.")
if longest > budget:
    notes.append(f"{int((per_subject > budget).sum())} subject(s) exceed 16,384 measurements and will be "
                 "truncated to the most recent window.")
print(f"  -> add to slurm/env.sh:  export BATCH_SIZE={suggested}")

section("5. Numeric coverage (composite target)")
if codes_fp.is_file():
    codes = pl.read_parquet(codes_fp)
    if "values/n_occurrences" in codes.columns:
        valued = int((codes["values/n_occurrences"] > 0).sum())
        print(f"{valued}/{len(codes)} codes carry numeric values -> composite vocabulary will be about "
              f"{len(codes) + valued * 9} classes (head width).")
    else:
        print("Source metadata carries no value counts; preprocess_data will compute them.")

section("Verdict")
for note in notes:
    print(f"  note:    {note}")
for problem in problems:
    print(f"  PROBLEM: {problem}")
if problems:
    sys.exit(f"\n{len(problems)} blocking problem(s). Fix these before submitting.")
print("  ready — nothing blocking. Next: bash slurm/00_prefetch_encoder.sh && bash slurm/submit.sh")
PYEOF
