#!/usr/bin/env bash
#SBATCH --job-name=porter-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Next-event pretraining of the 115.6 M backbone (paper §2.4.4: five epochs, AdamW 5e-4, bf16).
# This is the only stage that loads the frozen text encoder; everything downstream reads the cached
# table out of the checkpoint.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm/env.sh

nvidia-smi || true

$PORTER_RUN meds-model pretrain \
  "input_data_dir=$DATA_DIR" \
  "output_pretrained_model_dir=$PRETRAINED_DIR" \
  "batch_size=$BATCH_SIZE" \
  "num_workers=$NUM_WORKERS" \
  $TRAINER_OVERRIDES \
  $MODEL_OVERRIDES \
  do_overwrite=true
