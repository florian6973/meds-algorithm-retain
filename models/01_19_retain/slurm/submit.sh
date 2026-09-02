#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

mapfile -t all_commands < <(meds-model commands)
if [[ -z "${all_commands[0]:-}" ]]; then
  echo "No commands declared" >&2
  exit 1
fi

# STAGES selects a subset, emitted in contract order, so one shared preprocess_data artifact can serve
# many task-specific supervised_train+predict chains -- see RETAIN_DATA_DIR in config.sh. Without this
# the eICU wave rebuilds a 2.67B-event extract once per task instead of once.
#
# COPIED from models/02_19_behrt with exactly one agreed change: the separator is permissive (comma OR
# space) rather than comma-only. Everything else is behrt's, deliberately. Contract-order emission and
# the duplicate-collapsing `break` are the reason it is the fleet reference: iterating the *request*
# and appending, as three other ports do, submits `predict` before `supervised_train` when asked in
# that order. Against a clean run directory that fails loudly. Against one where $RUN_DIR/model already
# exists -- the artifact-reuse case this filter is FOR -- predict runs first with no afterok
# dependency, succeeds against the stale model, and supervised_train then overwrites the model beneath
# predictions already written from the previous one. No error anywhere.
#
# Known hole, inherited and not patched here to avoid a sixth implementation: STAGES=" " (whitespace
# only) passes the -n test, parses to zero stages, and submits nothing at exit 0. Raised with the
# master session as a candidate for the fleet spec rather than fixed unilaterally.
if [[ -n "${STAGES:-}" ]]; then
  read -r -a requested <<< "${STAGES//,/ }"
  commands=()
  for wanted in "${requested[@]}"; do
    wanted="${wanted// /}"
    found=false
    for known in "${all_commands[@]}"; do
      [[ "$wanted" == "$known" ]] && found=true && break
    done
    if ! $found; then
      echo "submit.sh: STAGES names '$wanted', which is not a command of this model." >&2
      echo "  declared commands: ${all_commands[*]}" >&2
      exit 2
    fi
  done
  # Emit in contract order regardless of the order given, so dependencies chain correctly.
  for known in "${all_commands[@]}"; do
    for wanted in "${requested[@]}"; do
      [[ "${wanted// /}" == "$known" ]] && commands+=("$known") && break
    done
  done
  # An empty selection is an error, not a no-op. `STAGES=","` passes the -n test above, parses to zero
  # stages, and would submit nothing at exit 0 -- an absent result indistinguishable from a successful
  # run. `" "` you have to construct deliberately; a bare comma is a plausible slip while editing a
  # stage list. Checked on the EMITTED list rather than on `requested`, so any future path that selects
  # nothing is caught too. Fleet spec case 7; behrt does not have this yet and should copy it back.
  if [[ ${#commands[@]} -eq 0 ]]; then
    echo "submit.sh: STAGES='$STAGES' selected none of: ${all_commands[*]}" >&2
    echo "  Submitting nothing would exit 0 and be indistinguishable from a successful run." >&2
    exit 2
  fi
else
  commands=("${all_commands[@]}")
fi

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_SHA:-local}}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$run_id/artifacts}"
export RUN_DIR
dependency=""
for command in "${commands[@]}"; do
  args_dir="$RUN_ROOT/$run_id/command-args"
  args_file="$args_dir/$command.args"
  if ! $dry_run; then
    mkdir -p "$args_dir"
    stage_args "$command" > "$args_file"
  else
    args_file="$RUN_ROOT/$run_id/command-args/$command.args"
  fi
  partition="$PARTITION_GPU"
  time_limit="$TIME_GPU"
  gres=(--gres="gpu:${GPUS}")
  if [[ "$command" == "preprocess_data" ]]; then
    partition="$PARTITION_CPU"
    time_limit="$TIME_CPU"
    gres=()
  fi
  submit=(sbatch --parsable --partition="$partition" --time="$time_limit" --cpus-per-task="$CPUS" --mem="$MEMORY")
  if [[ -n "$dependency" ]]; then
    submit+=(--dependency="afterok:$dependency")
  fi
  submit+=("${gres[@]}" --export="ALL,MEDS_MODEL_COMMAND=$command,MEDS_MODEL_ARGS_FILE=$args_file,RUN_ID=$run_id" "$SCRIPT_DIR/job.sbatch")
  if $dry_run; then
    printf '%q ' "${submit[@]}"
    printf '\n'
    dependency="DRY_RUN_${command}"
  else
    dependency="$("${submit[@]}")"
    echo "$command: $dependency"
  fi
done
