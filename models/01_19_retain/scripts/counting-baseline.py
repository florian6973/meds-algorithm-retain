"""How much of a run's discrimination is available from counting visits, reading no codes?

    uv run python scripts/counting-baseline.py <run-artifacts-dir> <labels-dir> [split]

Reports three predictors on the *same* held-out keys the model scored:

  presence   1 if the model's input history contained any visit at all, else 0
  count      the number of visits in that history -- RETAIN's own input unit
  model      the published predictions

Method follows `model/01_01_teco`'s `scripts/counting-baseline.py`, read rather than paraphrased, so the
numbers are comparable across ports. What must match: the same task, the same split, the same key set the
model scored, the count taken from the model's *realised* input rebuilt from the run's own manifest rather
than recomputed from raw data, and AUROC **and** AUPRC reported with prevalence. What legitimately differs
is only the unit -- 15-minute bins there, visits here, coded events for the tabular baseline. Ports
agreeing in method but differing in unit are comparable; ports with different key sets are not.

The stratification is not optional for this port. An empty history leaves RETAIN with **no input at all**:
`v = 0`, so `c = 0`, and every such key receives `sigmoid(b_output)` -- one identical constant. An overall
AUROC therefore includes credit for separating patients who have a history from patients who do not, which
is a property of the cohort rather than of the architecture. Note this differs from TECO, whose empty
stratum still carries demographic features and is a demographics-only submodel rather than a constant; the
cross-port comparable figure is AUROC on the non-empty stratum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retain.config import RetainConfig  # noqa: E402
from retain.labels import prediction_keys, read_labels  # noqa: E402
from retain.training import roc_auc  # noqa: E402
from retain.windows import PatientStore, build_sequences  # noqa: E402

KEYS = ["subject_id", "prediction_time"]
PROBABILITY = "predicted_boolean_probability"


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision, matching `sklearn.metrics.average_precision_score`.

    sum_n (R_n - R_{n-1}) * P_n over descending score thresholds, with ties collapsed into one threshold.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    distinct = np.flatnonzero(np.diff(scores))
    threshold = np.r_[distinct, labels.size - 1]
    true_positive = np.cumsum(labels)[threshold]
    predicted_positive = threshold + 1.0
    precision = true_positive / predicted_positive
    recall = true_positive / max(labels.sum(), 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def main(run_dir: Path, labels_dir: Path, split: str) -> int:
    manifest = yaml.safe_load((run_dir / "model" / "manifest.yaml").read_text())
    data_dir = Path(manifest["inputs"]["input_data_dir"]["path"])
    if not data_dir.is_dir():
        raise SystemExit(f"the patients artifact {data_dir} recorded by the run is not readable here")

    predictions = pl.read_parquet(run_dir / "predict" / "predictions.parquet")
    labels = read_labels(labels_dir / split if (labels_dir / split).is_dir() else labels_dir)
    joined = predictions.select(*KEYS, PROBABILITY).join(
        labels.select(*KEYS, "boolean_value"), on=KEYS, how="inner"
    )
    if joined.height != predictions.height:
        print(f"WARNING: {predictions.height - joined.height} predictions had no matching label")

    # Rebuild exactly the sequences the model consumed, for exactly the keys it scored -- the run's own
    # config, not defaults, and the prediction file's keys, not the label file's superset.
    store = PatientStore(data_dir)
    config = RetainConfig(**json.loads((data_dir / "config.json").read_text()))
    batch = build_sequences(store, prediction_keys(joined), config)
    counts = batch.lengths.astype(float)

    y = joined.get_column("boolean_value").to_numpy().astype(bool)
    model = joined.get_column(PROBABILITY).to_numpy().astype(float)
    if y.sum() in (0, len(y)):
        print("only one class present; discrimination undefined")
        return 1
    prevalence = float(y.mean())

    print(f"task {labels_dir.name}   split {split}   bucket {config.visit_bucket_minutes} min")
    print(f"keys {len(y):,}   positives {int(y.sum()):,}   prevalence {prevalence:.4f}")
    print(f"empty histories {int((counts == 0).sum()):,} ({(counts == 0).mean():.1%})   "
          f"visits/key median {np.median(counts):.0f} max {counts.max():.0f}")

    print(f"\n  {'predictor':28s}{'AUROC':>9s}{'AUPRC':>9s}{'AUPRC lift':>12s}")
    rows = {"presence (0/1)": (counts > 0).astype(float), "count (visits)": counts, "model": model}
    scores = {}
    for name, values in rows.items():
        auroc, auprc = roc_auc(y, values), average_precision(y, values)
        scores[name] = (auroc, auprc)
        print(f"  {name:28s}{auroc or float('nan'):9.4f}{auprc:9.4f}{auprc / prevalence:11.2f}x")

    (count_auroc, count_auprc), (model_auroc, model_auprc) = scores["count (visits)"], scores["model"]
    if count_auroc and model_auroc:
        print(f"\n  counting recovers {count_auroc / model_auroc:.0%} of the model's AUROC "
              f"and {count_auprc / model_auprc:.0%} of its AUPRC")
    print("  AUPRC is the discriminating comparison at low prevalence; AUROC flatters both.")

    empty = counts == 0
    print(f"\n  {'stratum':28s}{'keys':>9s}{'prev':>8s}{'AUROC':>9s}{'AUPRC':>9s}")
    for name, mask in (("all keys", np.ones_like(empty)), ("non-empty history", ~empty),
                       ("empty history", empty)):
        n = int(mask.sum())
        if not n:
            print(f"  {name:28s}{0:>9}{'-':>8}{'-':>9}{'-':>9}")
            continue
        sy, sm = y[mask], model[mask]
        if sy.sum() in (0, len(sy)):
            print(f"  {name:28s}{n:>9,}{sy.mean():8.4f}{'one class':>9s}{'-':>9s}")
            continue
        print(f"  {name:28s}{n:>9,}{sy.mean():8.4f}{roc_auc(sy, sm):9.4f}{average_precision(sy, sm):9.4f}")
    if empty.any():
        distinct = len(np.unique(model[empty]))
        print(f"\n  the {int(empty.sum()):,} empty-history keys carry {distinct} distinct prediction(s); "
              "this port emits a single constant there, so their AUROC contribution is the stratum split "
              "alone. Report the non-empty row for cross-port comparison.")
    if empty.any() and (~empty).any():
        print(f"  prevalence differs {y[empty].mean():.4f} (empty) vs {y[~empty].mean():.4f} (non-empty).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
                          sys.argv[3] if len(sys.argv) == 4 else "held_out"))
