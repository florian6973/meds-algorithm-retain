#!/usr/bin/env bash
#SBATCH --job-name=retain-meds
set -euo pipefail

: "${MEDS_ALGORITHM_COMMAND:?MEDS_ALGORITHM_COMMAND is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${MEDS_ALGORITHM_ARGS_FILE:?MEDS_ALGORITHM_ARGS_FILE is required}"

mapfile -t algorithm_args < "$MEDS_ALGORITHM_ARGS_FILE"
retain-meds "$MEDS_ALGORITHM_COMMAND" --profile "${PROFILE:?PROFILE is required}" "${algorithm_args[@]}" "$@"
