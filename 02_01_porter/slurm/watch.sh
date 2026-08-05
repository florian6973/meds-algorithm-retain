#!/usr/bin/env bash
# Follow a running PORTER chain.
#
#   bash slurm/watch.sh        # one-shot: queue state, artifacts so far, tail of the active stage
#   bash slurm/watch.sh -f     # same, then follow the active stage's log
#
# `squeue --me` alone is not very informative here: all three job names start with "porter-p", so the
# default 8-character NAME column truncates them to the same string. This widens it and adds the two
# things that actually tell you where the run is — which artifacts exist, and what the live log says.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/env.sh"

follow=""
[[ "${1:-}" == "-f" ]] && follow=1

echo "── queue ───────────────────────────────────────────────────────────────────"
squeue --me --format='%.10i %.16j %.9P %.2t %.11M %.6D %R' || true

echo
echo "── artifacts (this is the real progress) ───────────────────────────────────"
report() {  # report <label> <path> [file that means "finished"]
  local label=$1 path=$2 marker=${3:-manifest.yaml}
  local dir base staging size
  dir=$(dirname "$path"); base=$(basename "$path")

  if [[ -f "$path/$marker" ]]; then
    printf '  %-22s done       %6s  %s\n' "$label" "$(du -sh "$path" 2>/dev/null | cut -f1)" "$path"
    return
  fi

  # A partial artifact is NEVER visible at its final path: `write_artifact` stages into a dot-prefixed
  # sibling (`mkdtemp(prefix=".<name>.", dir=<parent>)`) and renames it into place only once the
  # manifest is written. So "pending" at the final path can mean "tens of GB already written" — look
  # for the staging directory, or you will think a working job is doing nothing. Training additionally
  # scratches in `<output_dir>.work`.
  staging=$(ls -d "$dir/.$base."* "$path.work" 2>/dev/null | head -1 || true)
  if [[ -n "$staging" ]]; then
    size=$(du -sh "$staging" 2>/dev/null | cut -f1)
    printf '  %-22s writing    %6s  %s\n' "$label" "${size:-?}" "${staging##*/}"
  else
    printf '  %-22s pending\n' "$label"
  fi
}
report "preprocess_data" "$DATA_DIR/patients"
report "pretrain" "$PRETRAINED_DIR"
report "infer (embeddings)" "$DATA_DIR/$INFERENCE_SUBDIR"
report "supervised_train" "$SUPERVISED_DIR"
report "predict" "$PREDICTIONS_DIR"

# The published artifacts carry their own numbers; show them rather than making you go find them.
if [[ -f "$PRETRAINED_DIR/manifest.yaml" ]]; then
  echo
  echo "── pretrained model ────────────────────────────────────────────────────────"
  grep -E "^ *(score|monitor|selection|epochs|global_step|seed):" "$PRETRAINED_DIR/manifest.yaml" || true
fi
if [[ -f "$PREDICTIONS_DIR/manifest.yaml" ]]; then
  echo
  echo "── predictions ─────────────────────────────────────────────────────────────"
  sed -n '/^coverage:/,/^[a-z]/p' "$PREDICTIONS_DIR/manifest.yaml" | head -8
fi

latest=$(ls -t "$LOG_DIR"/*.out 2>/dev/null | head -1 || true)
if [[ -z "$latest" ]]; then
  echo
  echo "No logs yet under $LOG_DIR"
  exit 0
fi

echo
echo "── $latest ──"
# Both streams matter: Lightning writes progress to stderr, the commands log to stdout.
err="${latest%.out}.err"
if [[ -n "$follow" ]]; then
  tail -n 40 -f "$latest" "$err" 2>/dev/null
else
  tail -n 25 "$latest" 2>/dev/null || true
  [[ -s "$err" ]] && { echo "── ${err##*/} ──"; tail -n 15 "$err"; }
fi
