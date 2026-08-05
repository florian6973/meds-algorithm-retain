#!/usr/bin/env bash
#SBATCH --job-name=porter-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Tensorize the MEDS dataset. CPU only, no model involved — this is the one stage that does not need a GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm/env.sh

$PORTER_RUN meds-model preprocess_data \
  "external_meds_dir=$MEDS_ROOT" \
  "output_data_dir=$DATA_DIR" \
  do_overwrite=true
