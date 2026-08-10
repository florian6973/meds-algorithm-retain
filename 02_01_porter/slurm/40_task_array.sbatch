#!/usr/bin/env bash
#SBATCH --job-name=porter-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One downstream task per array index: infer -> supervised_train -> predict, against an ALREADY
# pretrained backbone. This is the whole economic argument for the probe DAG — `pretrain` is paid once
# and every further task costs one forward pass plus a logistic regression.
#
# Submitted by `slurm/submit_tasks.sh`, which writes $TASK_LIST and sets the array range. Each index
# reads its own line, so tasks are independent: one failing does not touch the others, and they may run
# concurrently against the same data_dir (the contract has no aggregate manifest precisely so that this
# is safe — each inference subdirectory is written once, atomically, by exactly one job).
set -euo pipefail
cd "${PORTER_REPO:?PORTER_REPO is unset — submit through slurm/submit_tasks.sh, which exports it}"
source "$PORTER_REPO/slurm/env.sh"

: "${TASK_LIST:?TASK_LIST is unset — submit through slurm/submit_tasks.sh}"

task_dir=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
[[ -n "$task_dir" ]] || { echo "No task at array index $SLURM_ARRAY_TASK_ID in $TASK_LIST" >&2; exit 1; }
task=$(basename "$task_dir")

echo "=== $task ==="
echo "  labels : $task_dir"

# Everything this task produces is namespaced by its name, so tasks never collide.
inference_subdir="inference/$task"
probe_dir="$OUTPUT_DIR/tasks/$task/probe"
predictions_dir="$OUTPUT_DIR/tasks/$task/predictions"

# `infer` is the only stage that touches the backbone: one forward pass per prediction timepoint.
$PORTER_RUN meds-model infer \
  "input_data_dir=$DATA_DIR" \
  "input_pretrained_model_dir=$PRETRAINED_DIR" \
  "external_labels_dir=$task_dir" \
  "output_inference_subdir=$inference_subdir" \
  "batch_size=$BATCH_SIZE" \
  "num_workers=$NUM_WORKERS" \
  do_overwrite=true

# Dense-feature work on an [N, 768] matrix: seconds, not hours. `PROBE_OVERRIDES` is where to put
# `trainer.max_epochs=...` or `--multirun model.l2=...` — the paper fits this to convergence (§2.7.1),
# and the config default of 5 epochs is the *pretraining* epoch count, which is far too few here.
# shellcheck disable=SC2086
$PORTER_RUN meds-model supervised_train \
  "input_data_dir=$DATA_DIR" \
  "external_labels_dir=$task_dir" \
  "input_inference_subdir=$inference_subdir" \
  "output_supervised_model_dir=$probe_dir" \
  "num_workers=$NUM_WORKERS" \
  ${PROBE_OVERRIDES:-} \
  do_overwrite=true

$PORTER_RUN meds-model predict \
  "input_data_dir=$DATA_DIR" \
  "external_labels_dir=$task_dir" \
  "input_supervised_model_dir=$probe_dir" \
  "output_predictions_dir=$predictions_dir" \
  "splits=[held_out]" \
  do_overwrite=true

# Score it here rather than leaving a manual step per task — `predict` joined `boolean_value` back on,
# so the file is directly consumable by the shared evaluation tool.
$PORTER_RUN meds-evaluation-cli \
  "predictions_path=$predictions_dir/predictions.parquet" \
  "output_file=$OUTPUT_DIR/tasks/$task/evaluation.json"

echo "=== $task done -> $OUTPUT_DIR/tasks/$task/evaluation.json ==="
