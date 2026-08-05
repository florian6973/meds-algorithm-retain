#!/usr/bin/env bash
# Submit the whole PORTER chain as three dependent SLURM jobs.
#
#   bash slurm/00_prefetch_encoder.sh    # once, on the login node (compute nodes are usually offline)
#   bash slurm/submit.sh                 # preprocess -> pretrain -> infer+probe+predict
#
# Each stage starts only if the previous one succeeded (`afterok`), so a failed pretrain does not leave a
# probe fitting on nothing. Skip stages you have already run:
#
#   SKIP_PREPROCESS=1 bash slurm/submit.sh          # reuse an existing tensorized cohort
#   SKIP_PREPROCESS=1 SKIP_PRETRAIN=1 bash slurm/submit.sh   # re-probe a trained backbone
#
# Everything else is configured in slurm/env.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/env.sh"

: "${MEDS_ROOT:?set MEDS_ROOT in slurm/env.sh (the MEDS dataset to train on)}"
: "${LABELS_DIR:?set LABELS_DIR in slurm/env.sh (the MEDS labels directory to probe against)}"
: "${GPU_PARTITION:?set GPU_PARTITION in slurm/env.sh — find it with: sinfo -o \"%P %G %l\" | grep -i gpu}"
cd "$PORTER_REPO"

# Print the resolved configuration before submitting anything.
#
# This exists because of one specific, silent mistake: `MEDS_ROOT=... bash slurm/submit.sh` works, but
#   MEDS_ROOT=/some/path
#   bash slurm/submit.sh
# does not — the second is a shell-local variable, `bash` starts a child process, and a child inherits
# only *exported* variables. env.sh's `${MEDS_ROOT:-default}` then quietly falls through to the default,
# and you queue a day of GPU against the wrong dataset. Reading four lines beats reading a log tomorrow.
banner() { printf '  %-16s %s\n' "$1" "$2"; }
echo "Submitting PORTER with:"
banner repo "$PORTER_REPO"
banner dataset "$MEDS_ROOT"
banner labels "$LABELS_DIR"
banner output "$OUTPUT_DIR"
banner account "${SLURM_ACCOUNT:-<none — SLURM default association>}"
banner partitions "gpu=$GPU_PARTITION cpu=$CPU_PARTITION gres=$GPU_GRES"
banner model "${MODEL_OVERRIDES:-<published 28x768 / 115.6M>} batch_size=$BATCH_SIZE"
banner precision "${TRAINER_OVERRIDES:-<config default>}"
for path in "$MEDS_ROOT" "$LABELS_DIR"; do
  [[ -e "$path" ]] || echo "  WARNING: does not exist from this node: $path" >&2
done
echo

sbatch_common=(
  --output "$LOG_DIR/%x-%j.out"
  --error "$LOG_DIR/%x-%j.err"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$JOB_MEM"
  --export=ALL
)
[[ -n "$SLURM_ACCOUNT" ]] && sbatch_common+=(--account "$SLURM_ACCOUNT")

submit() {  # submit <partition> <time> <extra-sbatch-args...> -- <script>
  local partition=$1 time=$2; shift 2
  local args=("${sbatch_common[@]}" --time "$time")
  [[ -n "$partition" ]] && args+=(--partition "$partition")
  while [[ $1 != "--" ]]; do args+=("$1"); shift; done
  shift
  # stderr, not stdout: this function's stdout is captured as the job id by `$(submit ...)`, so a
  # single stray line there corrupts the --dependency chain.
  { printf '\n$ sbatch'; printf ' %q' "${args[@]}"; printf ' %q\n' "$PORTER_REPO/$1"; } >&2

  if [[ -n "${DRY_RUN:-}" ]]; then
    local filtered=() skip=0
    for arg in "${args[@]}"; do
      if [[ $skip == 1 ]]; then skip=0; continue; fi
      if [[ $arg == "--dependency" ]]; then skip=1; continue; fi
      filtered+=("$arg")
    done
    args=("${filtered[@]}")
    # `--test-only` asks the controller to validate the request — account, partition, QoS, limits —
    # and report when it would start, without queueing anything. This is what to run first after an
    # "Invalid account or account/partition combination" error.
    sbatch "${args[@]}" --test-only "$PORTER_REPO/$1" >&2 || true
    echo "DRYRUN"
  else
    sbatch "${args[@]}" --parsable "$PORTER_REPO/$1"
  fi
}

dependency=()

if [[ -z "${SKIP_PREPROCESS:-}" ]]; then
  preprocess_id=$(submit "$CPU_PARTITION" "$PREPROCESS_TIME" -- slurm/10_preprocess.sbatch)
  echo "preprocess_data : $preprocess_id"
  dependency=(--dependency "afterok:$preprocess_id")
fi

if [[ -z "${SKIP_PRETRAIN:-}" ]]; then
  pretrain_id=$(submit "$GPU_PARTITION" "$PRETRAIN_TIME" --gres "$GPU_GRES" "${dependency[@]}" -- slurm/20_pretrain.sbatch)
  echo "pretrain        : $pretrain_id"
  dependency=(--dependency "afterok:$pretrain_id")
fi

probe_id=$(submit "$GPU_PARTITION" "$PROBE_TIME" --gres "$GPU_GRES" "${dependency[@]}" -- slurm/30_probe.sbatch)
echo "infer+probe     : $probe_id"

echo
echo "Logs:        $LOG_DIR"
echo "Predictions: $PREDICTIONS_DIR/predictions.parquet"
echo "Watch with:  squeue -u \$USER"
