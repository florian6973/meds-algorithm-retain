#!/usr/bin/env bash
#SBATCH --job-name=porter-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# The downstream half: materialize frozen representations once, fit the L2 logistic probe on them, score.
#
# `infer` is the expensive step (one forward pass of the backbone per prediction timepoint) and wants the
# GPU. `supervised_train` is dense-feature work on an [N, 768] matrix and is cheap; note its `batch_size`
# is the probe's minibatch, not an event budget, so it is left at the config default rather than $BATCH_SIZE.
#
# Note `predict` is given no model overrides: it rebuilds the model from the checkpoint's own
# hyperparameters, so passing them would be silently ignored rather than honoured.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm/env.sh

$PORTER_RUN meds-model infer \
  "input_data_dir=$DATA_DIR" \
  "input_pretrained_model_dir=$PRETRAINED_DIR" \
  "external_labels_dir=$LABELS_DIR" \
  "output_inference_subdir=$INFERENCE_SUBDIR" \
  "batch_size=$BATCH_SIZE" \
  "num_workers=$NUM_WORKERS" \
  do_overwrite=true

$PORTER_RUN meds-model supervised_train \
  "input_data_dir=$DATA_DIR" \
  "external_labels_dir=$LABELS_DIR" \
  "input_inference_subdir=$INFERENCE_SUBDIR" \
  "output_supervised_model_dir=$SUPERVISED_DIR" \
  "num_workers=$NUM_WORKERS" \
  do_overwrite=true

$PORTER_RUN meds-model predict \
  "input_data_dir=$DATA_DIR" \
  "external_labels_dir=$LABELS_DIR" \
  "input_supervised_model_dir=$SUPERVISED_DIR" \
  "output_predictions_dir=$PREDICTIONS_DIR" \
  "splits=[held_out]" \
  do_overwrite=true
