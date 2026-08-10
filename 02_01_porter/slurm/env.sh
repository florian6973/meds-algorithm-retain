# Shared settings for the SLURM jobs. Edit this, then `bash slurm/submit.sh`.
#
# Absolute path to this checkout, resolved from this file rather than from the caller's directory.
# `SLURM_SUBMIT_DIR` is wherever `sbatch` happened to be run, which is not necessarily the repo, and
# a compute node that starts in the wrong place fails minutes in with an import error.
export PORTER_REPO="${PORTER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
#
# Sourced by every script in this directory (and by `submit.sh` before it calls sbatch), so anything you
# export here reaches the jobs. Every value can also be overridden from the environment:
#   MEDS_ROOT=/other/dataset bash slurm/submit.sh

# ----------------------------------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------------------------------
# Left empty on purpose: an account or partition that does not exist on your cluster is a confusing
# failure, and a wrong one is worse. Fill in what `sacctmgr show assoc user=$USER` gives you.
# EMPTY BY DEFAULT, and that is usually correct: most clusters resolve your default association from
# your user, and passing an account you are not associated with on this partition is exactly what
# produces "Invalid account or account/partition combination specified". Leave it empty unless SLURM
# tells you an account is required. If it is, take one from
#   sacctmgr show assoc user=$USER format=account,partition,qos
export SLURM_ACCOUNT="${SLURM_ACCOUNT-}"
export GPU_PARTITION="${GPU_PARTITION:-gpu}"
export CPU_PARTITION="${CPU_PARTITION:-$GPU_PARTITION}"

# One GPU is what the paper used ("a single NVIDIA H100 or L40S GPU"). 115.6 M parameters at 16,384
# context in bf16 fits comfortably on 40 GB; the activation memory, not the weights, is what scales.
export GPU_GRES="${GPU_GRES:-gpu:1}"

export PRETRAIN_TIME="${PRETRAIN_TIME:-24:00:00}"
export PREPROCESS_TIME="${PREPROCESS_TIME:-04:00:00}"
export PROBE_TIME="${PROBE_TIME:-08:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export JOB_MEM="${JOB_MEM:-64G}"

# ----------------------------------------------------------------------------------------------------
# Data and outputs
# ----------------------------------------------------------------------------------------------------
# A MEDS dataset sharded by split (data/train/, data/tuning/, data/held_out/). PORTER is only PORTER if
# metadata/codes.parquet carries a `description` column — that text is the entire input pathway.
# Left empty rather than `${VAR:?...}`: this file is also sourced by `00_prefetch_encoder.sh`, which
# needs neither, and a `:?` here aborts the very first documented step. `submit.sh` validates instead,
# at the point where they are actually required.
export MEDS_ROOT="${MEDS_ROOT:-/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort}"

# A MEDS labels directory (subject_id, prediction_time, boolean_value), e.g. from `meds-dev-task`.
export LABELS_DIR="${LABELS_DIR:-/groups/mm6677_gp/ffp2106/Git/meds-task-selection/work/labels/SUBJECT_FLUID_OUTPUT_226559_mL_7d}"

# Everything this run writes. Use scratch, not home: the tensorized cohort and the checkpoints are large.
# Scratch, not home: the tensorized MIMIC cohort plus checkpoints is tens of GB.
export OUTPUT_DIR="${OUTPUT_DIR:-/groups/mm6677_gp/ffp2106/porter-runs/fluid_output_7d}"

export DATA_DIR="${DATA_DIR:-$OUTPUT_DIR/data}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-$OUTPUT_DIR/models/pretrained}"
export SUPERVISED_DIR="${SUPERVISED_DIR:-$OUTPUT_DIR/models/probe}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-$OUTPUT_DIR/predictions}"
export INFERENCE_SUBDIR="${INFERENCE_SUBDIR:-inference/embeddings}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/slurm-logs}"

# ----------------------------------------------------------------------------------------------------
# Model scale
# ----------------------------------------------------------------------------------------------------
# Empty means the shipped default: the published 28 x 768 / 115.6 M geometry at 16,384-event context.
# Set MODEL_OVERRIDES="model=small" for a fast plumbing check on real data before committing a GPU-day.
export MODEL_OVERRIDES="${MODEL_OVERRIDES:-}"

# The paper trains bf16 mixed precision (§2.4.4); the repository default is fp32 because the test tiers
# run on CPU. On a GPU you want this.
export TRAINER_OVERRIDES="${TRAINER_OVERRIDES:-trainer.precision=bf16-mixed}"

# `batch_size` is the event budget: `batch_size * (longest timeline in the batch)` is what has to fit.
# The config default of 1 is the faithful reading of the paper's 16,384-events-per-batch budget at full
# context. If your timelines are much shorter than 16,384 events, raise this — it is pure throughput.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-$CPUS_PER_TASK}"

# ----------------------------------------------------------------------------------------------------
# Caches — the part that most often breaks on a cluster
# ----------------------------------------------------------------------------------------------------
# `pretrain` loads the frozen text encoder from HuggingFace. Many clusters give compute nodes no outbound
# network, so `slurm/00_prefetch_encoder.sh` downloads it on the login node and the jobs run offline.
export HF_HOME="${HF_HOME:-$OUTPUT_DIR/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# torch.compile / Triton write generated kernels to disk. Home directories are often small, slow, or
# read-only from compute nodes; point these at scratch or the sliding-window attention path falls back
# to a dense mask.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/inductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUTPUT_DIR/triton-cache}"

# Required by the contract for deterministic CUDA matmuls; harmless otherwise.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Prefix that puts this package's environment in front of a command. It is used before BOTH a console
# script (`$PORTER_RUN meds-model pretrain ...`) and the interpreter (`$PORTER_RUN python -c ...`), so
# it must work for both — `uv run` does. On a module/conda cluster, activate the environment first and
# set this to empty rather than inventing a wrapper:
#   module load python/3.12 cuda/13.0 && source /path/to/venv/bin/activate && export PORTER_RUN=""
export PORTER_RUN="${PORTER_RUN-uv run}"

mkdir -p "$LOG_DIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || {
  echo "slurm/env.sh: cannot create directories under OUTPUT_DIR=$OUTPUT_DIR" >&2
  echo "  Point OUTPUT_DIR at somewhere writable (scratch, not home)." >&2
  return 1 2>/dev/null || exit 1
}
