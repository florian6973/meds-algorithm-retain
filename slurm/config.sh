#!/usr/bin/env bash
# Algorithm-owned cluster settings. This file is the only SLURM file expected to need routine edits.
# Defaults; each can be overridden from the submit environment (e.g. MEMORY=96G STAGES=pretrain submit.sh), so a
# resource change does not need a commit. Size them per stage by submitting stages separately with STAGES.
PARTITION_CPU="${PARTITION_CPU:-cpu}"
PARTITION_GPU="${PARTITION_GPU:-gpu}"
TIME_CPU="${TIME_CPU:-01:00:00}"
TIME_GPU="${TIME_GPU:-04:00:00}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-16G}"
GPUS="${GPUS:-1}"
RUN_ROOT="${RUN_ROOT:-runs}"
PROFILE="${PROFILE:-supervised}"
export PROFILE

# Shared artifacts may only be reused after validating their configuration and input identities.
# Embeddings need training/tuning keys to fit a probe. Inference never reads the label values.
# TASK_PREDICATES_FILE is separate from dataset-level PREDICATES_FILE.
stage_args() {
  if [[ -n "${ALGORITHM_CONFIG_FILE:-}" ]]; then
    printf '%s\n' "config_file=$ALGORITHM_CONFIG_FILE"
  fi
  case "$1" in
    preprocess_data) printf '%s\n' "external_meds_dir=${MEDS_ROOT:?set MEDS_ROOT}" "output_data_dir=$SHARED_DIR/data" ;;
    train_supervised) printf '%s\n' "input_data_dir=$SHARED_DIR/data" "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" "output_supervised_algorithm_dir=$RUN_DIR/algorithm" ;;
    predict_supervised) printf '%s\n' "input_data_dir=$SHARED_DIR/data" "input_supervised_algorithm_dir=$RUN_DIR/algorithm" "external_index_dir=${INDEX_DIR:?set INDEX_DIR to the prediction index}" "output_predictions_dir=$RUN_DIR/predictions" "splits=[${SPLITS:-held_out}]" ;;
    *) return 2 ;;
  esac
}
