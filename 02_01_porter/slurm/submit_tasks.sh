#!/usr/bin/env bash
# Run every downstream task against the one pretrained backbone.
#
#   bash slurm/submit_tasks.sh                    # every task directory under $TASKS_ROOT
#   TASKS_ROOT=/path/to/labels bash slurm/submit_tasks.sh
#   MAX_CONCURRENT=8 bash slurm/submit_tasks.sh   # at most 8 array elements running at once
#   TASK_GLOB='*MORTALITY*' bash slurm/submit_tasks.sh    # only matching task names
#   LIMIT=1 bash slurm/submit_tasks.sh            # just the first task — do this once before fanning out
#   DRY_RUN=1 bash slurm/submit_tasks.sh          # list the tasks and validate, queue nothing
#
# This is the point of the probe DAG: `pretrain` is already paid, so each additional task is one forward
# pass over the backbone plus a logistic regression. Tasks run as an independent array — one failing
# leaves the rest alone.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/env.sh"
cd "$PORTER_REPO"

: "${GPU_PARTITION:?set GPU_PARTITION in slurm/env.sh}"

# Default: the directory holding the task you already ran, i.e. all its siblings.
TASKS_ROOT="${TASKS_ROOT:-$(dirname "$LABELS_DIR")}"
[[ -d "$TASKS_ROOT" ]] || { echo "TASKS_ROOT does not exist: $TASKS_ROOT" >&2; exit 1; }

if [[ ! -f "$PRETRAINED_DIR/manifest.yaml" ]]; then
  echo "No pretrained backbone at $PRETRAINED_DIR." >&2
  echo "Run the main chain first (slurm/submit.sh); this script only does the downstream half." >&2
  exit 1
fi

# A task directory is one containing label parquets. Checking rather than assuming keeps a stray
# README or a half-written directory out of the array.
TASK_LIST="$LOG_DIR/tasks.txt"
: > "$TASK_LIST"
for candidate in "$TASKS_ROOT"/*/; do
  candidate=${candidate%/}
  [[ -d "$candidate" ]] || continue
  # shellcheck disable=SC2254
  [[ -z "${TASK_GLOB:-}" || ${candidate##*/} == ${TASK_GLOB} ]] || continue
  if compgen -G "$candidate/**/*.parquet" > /dev/null || compgen -G "$candidate/*.parquet" > /dev/null; then
    echo "$candidate" >> "$TASK_LIST"
  fi
done

# `LIMIT` exists for the first run: this array script fans out N identical jobs, and finding out it is
# wrong from N failures is worse than finding out from one.
if [[ -n "${LIMIT:-}" ]]; then
  head -n "$LIMIT" "$TASK_LIST" > "$TASK_LIST.tmp" && mv "$TASK_LIST.tmp" "$TASK_LIST"
fi

n=$(wc -l < "$TASK_LIST")
(( n > 0 )) || { echo "No task directories with parquet labels under $TASKS_ROOT" >&2; exit 1; }

echo "Found $n task(s) under $TASKS_ROOT:"
sed 's|^|  |' "$TASK_LIST" | head -20
(( n > 20 )) && echo "  … and $((n - 20)) more"
echo
echo "Backbone : $PRETRAINED_DIR"
echo "Cohort   : $DATA_DIR"
echo "Probe    : ${PROBE_OVERRIDES:-<config default: trainer.max_epochs=5 — see the note below>}"
echo
if [[ -z "${PROBE_OVERRIDES:-}" ]]; then
  echo "NOTE: the probe will train for 5 epochs, which is the paper's *pretraining* epoch count, not its" >&2
  echo "      probe protocol (LBFGS to convergence, §2.7.1). Consider:" >&2
  echo "      export PROBE_OVERRIDES='trainer.max_epochs=200'" >&2
  echo >&2
fi

export TASK_LIST
array="0-$((n - 1))"
[[ -n "${MAX_CONCURRENT:-}" ]] && array="$array%$MAX_CONCURRENT"

args=(
  --output "$LOG_DIR/%x-%A_%a.out"
  --error "$LOG_DIR/%x-%A_%a.err"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$JOB_MEM"
  --export=ALL
  --time "$PROBE_TIME"
  --partition "$GPU_PARTITION"
  --gres "$GPU_GRES"
  --array "$array"
)
[[ -n "$SLURM_ACCOUNT" ]] && args+=(--account "$SLURM_ACCOUNT")

{ printf '$ sbatch'; printf ' %q' "${args[@]}"; printf ' %q\n' "$PORTER_REPO/slurm/40_task_array.sbatch"; } >&2

if [[ -n "${DRY_RUN:-}" ]]; then
  sbatch "${args[@]}" --test-only "$PORTER_REPO/slurm/40_task_array.sbatch" >&2 || true
  exit 0
fi

job=$(sbatch "${args[@]}" --parsable "$PORTER_REPO/slurm/40_task_array.sbatch")
echo "task array : $job ($n task(s))"
echo
echo "Results:  $OUTPUT_DIR/tasks/<task>/evaluation.json"
echo "Collect:  bash slurm/collect_tasks.sh"
