#!/usr/bin/env bash
# Model-owned cluster settings. This file is the only SLURM file expected to need routine edits.
PARTITION_CPU="cpu"
PARTITION_GPU="gpu"
TIME_CPU="04:00:00"
TIME_GPU="12:00:00"
CPUS="8"
MEMORY="64G"
GPUS="1"
RUN_ROOT="${RUN_ROOT:-runs}"

# Tokenising a full MEDS extract is a streaming pass over every shard, so `preprocess_data` is the CPU
# stage and gets the wider time limit of the two CPU-bound possibilities. Training and prediction are the
# GRU passes and take the GPU partition.

# Hyper-parameters. Empty means "every official `retain.py` default", which is what a comparison run
# should use unless there is a recorded reason to differ. Set RETAIN_MODEL_ARGS to a space-separated list
# of key=value pairs to override, and record the exact string in IMPLEMENTATION_REPORT.md.
RETAIN_MODEL_ARGS="${RETAIN_MODEL_ARGS:-}"
# Vocabulary settings are fixed when the patients artifact is built and cannot be changed later, so they
# are passed to preprocess_data only.
RETAIN_VOCABULARY_ARGS="${RETAIN_VOCABULARY_ARGS:-}"
# Which split predict scores. MEDS-DEV evaluates the held-out split.
RETAIN_PREDICT_SPLITS="${RETAIN_PREDICT_SPLITS:-[held_out]}"

# Print the implementation-specific key=value arguments for one command. Keep secrets out of this file;
# obtain them from the submit environment.
stage_args() {
  case "$1" in
    preprocess_data)
      printf '%s\n' \
        "external_meds_dir=${MEDS_ROOT:?set MEDS_ROOT}" \
        "output_data_dir=${RUN_DIR:?set RUN_DIR}/data" \
        "do_overwrite=true"
      # shellcheck disable=SC2086
      for argument in $RETAIN_VOCABULARY_ARGS; do printf '%s\n' "$argument"; done
      ;;
    supervised_train)
      printf '%s\n' \
        "input_data_dir=${RUN_DIR:?set RUN_DIR}/data" \
        "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" \
        "output_supervised_model_dir=$RUN_DIR/model" \
        "do_overwrite=true"
      # shellcheck disable=SC2086
      for argument in $RETAIN_MODEL_ARGS; do printf '%s\n' "$argument"; done
      ;;
    predict)
      # predict takes its model configuration from the trained artifact, so no hyper-parameters here.
      printf '%s\n' \
        "input_supervised_model_dir=${RUN_DIR:?set RUN_DIR}/model" \
        "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" \
        "splits=$RETAIN_PREDICT_SPLITS" \
        "output_predictions_dir=$RUN_DIR/predict" \
        "do_overwrite=true"
      ;;
    *) return 2 ;;
  esac
}
