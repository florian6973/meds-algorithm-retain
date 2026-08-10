#!/usr/bin/env bash
# Submit the probe-epoch sweep as a job (it must not run on the login node).
#
#   bash slurm/submit_probe_sweep.sh
#   PROBE_EPOCHS="5 400" bash slurm/submit_probe_sweep.sh
#   PROBE_EXTRA="model.l2=0.01" bash slurm/submit_probe_sweep.sh
#   DRY_RUN=1 bash slurm/submit_probe_sweep.sh
#
# Answers one question: does the probe's AUROC move when it trains longer? It reuses the embeddings
# `infer` already wrote, so it runs no backbone — CPU partition, minutes.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/env.sh"
cd "$PORTER_REPO"

: "${CPU_PARTITION:?set CPU_PARTITION (or GPU_PARTITION) in slurm/env.sh}"

args=(
  --output "$LOG_DIR/%x-%j.out"
  --error "$LOG_DIR/%x-%j.err"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$JOB_MEM"
  --export=ALL
  --time "${SWEEP_TIME:-02:00:00}"
  --partition "$CPU_PARTITION"
)
[[ -n "$SLURM_ACCOUNT" ]] && args+=(--account "$SLURM_ACCOUNT")

export PROBE_EPOCHS="${PROBE_EPOCHS:-5 25 100 400}"
export PROBE_EXTRA="${PROBE_EXTRA:-}"

echo "Sweeping max_epochs = $PROBE_EPOCHS"
echo "Embeddings          = $DATA_DIR/$INFERENCE_SUBDIR"
echo "Results             = $OUTPUT_DIR/probe_sweep/"
{ printf '\n$ sbatch'; printf ' %q' "${args[@]}"; printf ' %q\n' "$PORTER_REPO/slurm/50_probe_sweep.sbatch"; } >&2

if [[ -n "${DRY_RUN:-}" ]]; then
  sbatch "${args[@]}" --test-only "$PORTER_REPO/slurm/50_probe_sweep.sbatch" >&2 || true
  exit 0
fi
job=$(sbatch "${args[@]}" --parsable "$PORTER_REPO/slurm/50_probe_sweep.sbatch")
echo "probe sweep : $job"
echo "Watch:  tail -f $LOG_DIR/porter-probe-sweep-$job.out"
