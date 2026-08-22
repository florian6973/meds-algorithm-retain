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
# directory and wrong from the worktree root. `realpath -m` resolves a path that need not exist yet, so
# existence is never confounded with policy.
#
# OFF-TREE IS NOW THE RECOMMENDED CLUSTER SETTING, and used to be refused here with a distinct exit 5.
# It was refused because scripts/github-sync.sh hardcoded `runs/<run-id>`, so a run stored elsewhere
# completed and then could not be published. That publisher now resolves RUN_ROOT and dereferences on
# copy, so the refusal has been lifted -- exactly the precondition the old message named.
#
# Keeping run output in-tree was not free: cluster results landed inside the git worktree, where a
# routine `git clean -fdx` deletes completed work without warning, and collecting them meant reaching
# into a checkout. On the shared cluster, set something like
#   export RUN_ROOT=/groups/<group>/<user>/runs/01_19_retain
# which is where the sibling ports write. The default stays in-tree so local runs and the test suite
# need no configuration.
#
# The remaining refusal is the dangerous one and no convention in this fleet permits it: inside the work
# tree but outside the ignored model runs/, where .gitignore has no reach and the command-args files
# carrying external_meds_dir sit unignored. It keeps its distinct exit 4 so a survey reading only the
# status can tell "would leak" from anything else.
_retain_resolved="$(realpath -m -- "$RUN_ROOT")"
_retain_expected="$(realpath -m -- "$_retain_model_root/runs")"
if [[ "$_retain_resolved" != "$_retain_expected" ]]; then
  _retain_worktree_root="$(git -C "$_retain_model_root" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$_retain_worktree_root" && "$_retain_resolved" == "$_retain_worktree_root"/* ]]; then
    echo "config.sh: RUN_ROOT=$RUN_ROOT resolves to $_retain_resolved, not $_retain_expected." >&2
    echo "  This is inside the work tree but outside the ignored model runs/, so run output would land" >&2
    echo "  where .gitignore does not cover it and the command-args files carry" >&2
    echo "  external_meds_dir/external_labels_dir. No convention in this fleet permits it." >&2
    echo "  Unset RUN_ROOT to use $_retain_expected, or set a path fully outside the work tree." >&2
    exit 4
  fi
fi

# Tokenising a full MEDS extract is a streaming pass over every shard, so `preprocess_data` is the CPU
# stage and gets the wider time limit of the two CPU-bound possibilities. Training and prediction are the
# GRU passes and take the GPU partition.

# REUSING AN EXISTING preprocess_data ARTIFACT (rerunning only the two label-dependent stages).
# preprocess_data is task-independent -- its stage_args below take external_meds_dir and vocabulary
# settings, never LABELS_DIR -- so a second label set needs supervised_train and predict only.
# submit.sh has no stage filter: it submits every command `meds-model commands` declares. But
# job.sbatch runs exactly ONE command from the environment, and stage_args is a plain shell function,
# so the subset needs no change to either template-owned file:
#
#   mkdir -p "$NEW/artifacts" && ln -s "$OLD/artifacts/data" "$NEW/artifacts/data"
#   export MEDS_ROOT=... LABELS_DIR=<new task> RUN_DIR="$NEW/artifacts"
#   source slurm/config.sh
#   for c in supervised_train predict; do stage_args "$c" > "$NEW/command-args/$c.args"; done
#   sbatch ... --export=ALL,MEDS_MODEL_COMMAND=supervised_train,MEDS_MODEL_ARGS_FILE=...,RUN_ID=... job.sbatch
#
# Verified locally end to end: both stages ran against the shared artifact, the source run's model/
# and predict/ were byte-identical afterwards, and the new run's model manifest recorded
# `input_data_dir.path` as the RESOLVED source directory plus its config_digest -- so reuse is
# auditable from the manifest rather than implicit, and a mismatched vocabulary would be detectable.
#
# DO NOT instead point RUN_DIR at a shared directory for several tasks at once. RUN_DIR is honoured
# from the environment (submit.sh), so it looks like the sharing mechanism, but stage_args derives
# output_supervised_model_dir and output_predictions_dir from it too: ten concurrent tasks would race
# to overwrite one model/ and one predict/. Share the input by symlink, never the run directory.

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
