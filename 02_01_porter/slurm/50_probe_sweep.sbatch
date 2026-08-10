#!/usr/bin/env bash
#SBATCH --job-name=porter-probe-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Does training the probe longer change anything?
#
# The shipped `trainer.max_epochs=5` is inherited from the paper's *pretraining* epoch count (§2.4.4), not
# from its probe protocol (§2.7.1 fits to convergence with LBFGS). That makes 5 arbitrary here rather than
# wrong — the probe is a 769-parameter convex logistic regression, and a few thousand AdamW steps may
# already be at the optimum. This settles it by measurement instead of argument.
#
# Reuses the embeddings `infer` already materialized, so it runs no backbone and costs minutes. Submit
# with `bash slurm/submit_probe_sweep.sh`.
set -euo pipefail
cd "${PORTER_REPO:?PORTER_REPO is unset — submit through slurm/submit_probe_sweep.sh}"
source "$PORTER_REPO/slurm/env.sh"

PROBE_EPOCHS="${PROBE_EPOCHS:-5 25 100 400}"
SWEEP_DIR="$OUTPUT_DIR/probe_sweep"
inference_dir="$DATA_DIR/$INFERENCE_SUBDIR"

[[ -f "$inference_dir/manifest.yaml" ]] || {
  echo "No inference artifact at $inference_dir — run the main chain first." >&2
  exit 1
}
echo "Embeddings : $inference_dir"
echo "Epochs     : $PROBE_EPOCHS"
echo "Extra      : ${PROBE_EXTRA:-<none>}"
echo

for epochs in $PROBE_EPOCHS; do
  run_dir="$SWEEP_DIR/epochs_$epochs"
  echo "───────────── max_epochs=$epochs ─────────────"

  # Dense-feature work on an [N, 768] matrix; the backbone is not involved, so CPU is plenty and leaves
  # the GPU free. `predict` reloads the probe from its own checkpoint and scores the same embeddings.
  # shellcheck disable=SC2086
  $PORTER_RUN meds-model supervised_train \
    "input_data_dir=$DATA_DIR" \
    "external_labels_dir=$LABELS_DIR" \
    "input_inference_subdir=$INFERENCE_SUBDIR" \
    "output_supervised_model_dir=$run_dir/probe" \
    "trainer.max_epochs=$epochs" \
    trainer.accelerator=cpu \
    "num_workers=$NUM_WORKERS" \
    ${PROBE_EXTRA:-} \
    do_overwrite=true

  $PORTER_RUN meds-model predict \
    "input_data_dir=$DATA_DIR" \
    "external_labels_dir=$LABELS_DIR" \
    "input_supervised_model_dir=$run_dir/probe" \
    "output_predictions_dir=$run_dir/predictions" \
    "splits=[held_out]" \
    do_overwrite=true

  $PORTER_RUN meds-evaluation-cli \
    "predictions_path=$run_dir/predictions/predictions.parquet" \
    "output_file=$run_dir/evaluation.json"
done

echo
echo "═════════════════════ result ═════════════════════"
$PORTER_RUN python - "$SWEEP_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

sweep = Path(sys.argv[1])
rows = []
for run in sorted(sweep.glob("epochs_*"), key=lambda p: int(p.name.split("_")[1])):
    result = run / "evaluation.json"
    if not result.is_file():
        continue
    payload = json.loads(result.read_text())
    flat = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}/" if prefix else f"{key}/")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix.rstrip("/")] = float(node)

    walk(payload)
    pick = lambda frag: next((v for k, v in flat.items() if frag in k.lower()), float("nan"))  # noqa: E731
    rows.append((int(run.name.split("_")[1]), pick("roc_auc"), pick("average_precision"), pick("brier")))

if not rows:
    sys.exit("No evaluation.json produced.")

print(f"{'max_epochs':>10}  {'AUROC':>8}  {'AUPRC':>8}  {'Brier':>8}")
for epochs, auroc, auprc, brier in rows:
    print(f"{epochs:>10}  {auroc:8.5f}  {auprc:8.5f}  {brier:8.5f}")

aurocs = [r[1] for r in rows if r[1] == r[1]]
if len(aurocs) > 1:
    spread = max(aurocs) - min(aurocs)
    print(f"\nAUROC spread across the sweep: {spread:.5f}")
    if spread < 0.001:
        print("→ Training longer changes nothing. 5 epochs was already converged; drop PROBE_OVERRIDES.")
    elif spread < 0.01:
        print("→ Marginal. Worth using the larger value, but 5 was not badly wrong.")
    else:
        print("→ Substantial. The shipped default WAS under-training the probe; set PROBE_OVERRIDES.")
PYEOF
