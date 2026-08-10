#!/usr/bin/env bash
# Gather every finished task's metrics into one table (and a CSV).
#
#   bash slurm/collect_tasks.sh
#
# PORTER's headline result is a *mean over tasks* (§3.1: "Across 74 clinical prediction tasks ... matched
# the mean AUROC"), so the per-task JSON files are only half the story. This prints the table and the
# mean, and says how many tasks are still missing rather than quietly averaging over whatever finished.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/env.sh"

tasks_dir="$OUTPUT_DIR/tasks"
[[ -d "$tasks_dir" ]] || { echo "No $tasks_dir — run slurm/submit_tasks.sh first." >&2; exit 1; }

$PORTER_RUN python - "$tasks_dir" <<'PYEOF'
import json
import sys
from pathlib import Path

tasks_dir = Path(sys.argv[1])
rows, missing = [], []

for task in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
    result = task / "evaluation.json"
    if not result.is_file():
        missing.append(task.name)
        continue
    try:
        payload = json.loads(result.read_text())
    except json.JSONDecodeError:
        missing.append(f"{task.name} (unreadable json)")
        continue

    # meds-evaluation nests its metrics; find the scalars wherever they sit rather than assuming a shape.
    flat: dict[str, float] = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}/" if prefix else f"{key}/")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix.rstrip("/")] = float(node)

    walk(payload)
    auroc = next(
        (v for k, v in flat.items() if "auroc" in k.lower() or "roc_auc" in k.lower()), float("nan")
    )
    auprc = next(
        (v for k, v in flat.items() if "auprc" in k.lower() or "average_precision" in k.lower()),
        float("nan"),
    )
    rows.append((task.name, auroc, auprc))

if not rows:
    sys.exit(f"No evaluation.json found under {tasks_dir}. Are the array jobs still running?")

width = max(len(name) for name, _, _ in rows)
print(f"{'task'.ljust(width)}   AUROC    AUPRC")
print("─" * (width + 18))
for name, auroc, auprc in sorted(rows, key=lambda r: -r[1] if r[1] == r[1] else 0):
    print(f"{name.ljust(width)}  {auroc:6.4f}   {auprc:6.4f}")

scored = [a for _, a, _ in rows if a == a]
if scored:
    print("─" * (width + 18))
    print(f"{'mean over ' + str(len(scored)) + ' task(s)':<{width}}  {sum(scored) / len(scored):6.4f}")

if missing:
    print(f"\n{len(missing)} task(s) with no result yet: {', '.join(missing[:10])}"
          + (" …" if len(missing) > 10 else ""))
    print("The mean above covers only the finished ones.")

csv = tasks_dir / "summary.csv"
csv.write_text("task,auroc,auprc\n" + "".join(f"{n},{a},{p}\n" for n, a, p in rows))
print(f"\nWrote {csv}")
PYEOF
