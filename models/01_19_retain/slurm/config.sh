#!/usr/bin/env bash
# Model-owned cluster settings. This file is the only SLURM file expected to need routine edits;
# submit.sh and job.sbatch are template-owned. Every setting below is ${VAR:-default}, so a cluster can
# override it from the environment without a commit.
#
# BEFORE RUNNING submit.sh: put the virtualenv on PATH.
#   export PATH="$PWD/.venv/bin:$PATH"; meds-model commands   # must list the three commands
# Both submit.sh and job.sbatch invoke `meds-model` by bare name. Without it, submit.sh prints
# "No commands declared" and exits on the login node -- observed directly while testing --dry-run here.
# submit.sh passes --export=ALL, so submit-time PATH is what the compute node gets: a submission made
# without it queues successfully and then fails the same way after the wait.
#
# TIME_GPU must exceed n_epochs x per-epoch wall time. supervised_train now prints one line per epoch to
# stderr, so the per-epoch cost is readable from the job log within the first few minutes of a run rather
# than estimated. Measured here at 64x the MIMIC-IV demo (65.4M events, 4,736 keys): 5.6s for 10 epochs
# at the official architecture. Do not carry another model's per-epoch figure over to this one.
PARTITION_CPU="${PARTITION_CPU:-cpu}"
PARTITION_GPU="${PARTITION_GPU:-gpu}"
TIME_CPU="${TIME_CPU:-04:00:00}"
TIME_GPU="${TIME_GPU:-12:00:00}"
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-64G}"
GPUS="${GPUS:-1}"
# RUN_ROOT is anchored to this model directory rather than to the caller's working directory. With the
# template's relative `runs` default, invoking submit.sh from the repository or worktree root writes
# `<worktree>/runs/` -- a sibling of this model, which this model's .gitignore cannot reach, so the
# command-args files carrying external_meds_dir land TRACKABLE with no misconfiguration at all. Verified:
# `git check-ignore runs/probe/command-args/x.args` from the worktree root reports it trackable, while the
# in-model path is ignored. Anchoring makes output land where the ignore rules are, whatever the cwd.
#
# The anchor must also resolve to what scripts/github-sync.sh's `result_dir="runs/$run_id"` resolves to
# when run from this directory, or #9 is fixed by walking into #8: a leak-free run that cannot be
# published. tests/test_run_output_is_not_committable.py asserts that equality directly.
_retain_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_retain_model_root="$(dirname -- "$_retain_config_dir")"
RUN_ROOT="${RUN_ROOT:-$_retain_model_root/runs}"

# An explicit override is compared by resolved path, not by string: `RUN_ROOT=runs` is correct from this
# directory and wrong from the worktree root.
if [[ "$(realpath -m -- "$RUN_ROOT")" != "$(realpath -m -- "$_retain_model_root/runs")" ]]; then
  echo "config.sh: RUN_ROOT=$RUN_ROOT resolves outside $_retain_model_root/runs." >&2
  echo "  Inside the work tree but elsewhere: run output lands where .gitignore does not cover it, and" >&2
  echo "  the command-args files carry external_meds_dir/external_labels_dir." >&2
  echo "  Fully outside the work tree: leak-safe, but scripts/github-sync.sh hardcodes runs/<run-id> and" >&2
  echo "  would abort with 'refusing to publish an incomplete run' against a run that completed." >&2
  echo "  There is currently no RUN_ROOT that both stores output off-checkout and publishes, so the" >&2
  echo "  default is the only working configuration: unset RUN_ROOT and re-run." >&2
  echo "  (Running submit.sh from another directory is fine -- RUN_ROOT anchors to this model.)" >&2
  exit 2
fi

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
