#!/usr/bin/env bash
#SBATCH --job-name=porter-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Tensorize the MEDS dataset. CPU only, no model involved — this is the one stage that does not need a GPU.
set -euo pipefail
# Absolute, so it does not matter where sbatch was invoked from.
cd "${PORTER_REPO:?PORTER_REPO is unset — submit through slurm/submit.sh, which exports it}"
source "$PORTER_REPO/slurm/env.sh"

$PORTER_RUN meds-model preprocess_data \
  "external_meds_dir=$MEDS_ROOT" \
  "output_data_dir=$DATA_DIR" \
  do_overwrite=true
