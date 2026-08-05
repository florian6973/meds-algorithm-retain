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
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

: "${MEDS_ROOT:?set MEDS_ROOT in slurm/env.sh (the MEDS dataset to train on)}"
: "${LABELS_DIR:?set LABELS_DIR in slurm/env.sh (the MEDS labels directory to probe against)}"
: "${GPU_PARTITION:?set GPU_PARTITION in slurm/env.sh — find it with: sinfo -o \"%P %G %l\" | grep -i gpu}"

sbatch_common=(
  --output "$LOG_DIR/%x-%j.out"
  --error "$LOG_DIR/%x-%j.err"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$JOB_MEM"
  --export ALL
)
[[ -n "$SLURM_ACCOUNT" ]] && sbatch_common+=(--account "$SLURM_ACCOUNT")

submit() {  # submit <partition> <time> <extra-sbatch-args...> -- <script>
  local partition=$1 time=$2; shift 2
  local args=("${sbatch_common[@]}" --time "$time")
  [[ -n "$partition" ]] && args+=(--partition "$partition")
  while [[ $1 != "--" ]]; do args+=("$1"); shift; done
  shift
  sbatch "${args[@]}" --parsable "$1"
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
