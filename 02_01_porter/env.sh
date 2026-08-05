# Convenience environment variables for running PORTER locally.
# Usage: edit the paths, then `source env.sh` before invoking `meds-model`.

# Raw MEDS dataset (contains data/ and metadata/), sharded by split.
# PORTER is only meaningfully PORTER when metadata/codes.parquet carries a `description` column: that
# text is the model's entire input pathway. Without it every code falls back to its own identifier.
export MEDS_ROOT="${MEDS_ROOT:-/path/to/meds/dataset}"

# Where preprocessed tensors / run artifacts / predictions go.
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs}"

# A labels/index directory (subject_id, prediction_time [, boolean_value]) — e.g. produced by MEDS-DEV/ACES.
export LABELS_DIR="${LABELS_DIR:-/path/to/labels}"

# Where the frozen text encoder is cached. `pretrain` downloads FremyCompany/BioLORD-2023 (~0.4 GB) on
# first use and encodes every code description once; `infer` and `predict` never load it, because the
# encoded table travels inside the pretrained checkpoint. Point this at shared storage on a cluster, or
# run `meds-model pretrain model.text_encoder=random` for the paper's offline control arm (§2.6.1).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
