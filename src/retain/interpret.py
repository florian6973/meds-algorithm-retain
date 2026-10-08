"""Per-code contributions, Eq. (5) of the paper and `test_retain.py` of the official repository.

    omega(y, x_{j,k}) = alpha_j * w_output . (beta_j * W_emb[:, k]) * x_{j,k}

`x` is binary here, so the coefficient *is* the contribution, which is the case the paper calls out
explicitly. This is the interpretability claim that motivates RETAIN, so it is implemented rather than
described; it is off by default because it costs a second pass and writes one record per scored key.

The output is NDJSON on purpose. `tests/integration.py::assert_prediction_keys` concatenates every
`*.parquet` under the predictions directory, so a second keyed parquet there would look like duplicate
prediction keys and fail the contract's own coverage check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import RetainConfig
from .model import Retain
from .training import collate
from .windows import SequenceBatch


def _timestamp(microseconds: int) -> str:
    return np.datetime64(int(microseconds), "us").astype("datetime64[us]").item().isoformat()


def write_contributions(
    path: Path,
    *,
    model: Retain,
    batch: SequenceBatch,
    config: RetainConfig,
    device: torch.device,
    code_strings: list[str],
    top_k: int,
) -> int:
    """Write one NDJSON record per key and return the number of records written."""
    model.eval()
    written = 0
    with Path(path).open("w") as handle:
        for start in range(0, batch.n_keys, config.batch_size):
            rows = np.arange(start, min(start + config.batch_size, batch.n_keys), dtype=np.int64)
            tensors = collate(batch, rows, config, device)
            with torch.no_grad():
                output = model(
                    tensors.codes,
                    tensors.offsets,
                    tensors.slots,
                    tensors.lengths,
                    tensors.gaps,
                    n_rows=tensors.n_rows,
                    n_visits=tensors.n_visits,
                    return_attention=True,
                )
                probabilities = torch.sigmoid(output.logits).cpu().numpy()
                alpha = output.alpha.cpu().numpy()
                beta = output.beta.cpu().numpy()
                weights = model.W_emb.detach()
                w_output = model.w_output.weight.detach().squeeze(0)

            for position, row in enumerate(rows.tolist()):
                first, last = int(batch.row_ptr[row]), int(batch.row_ptr[row + 1])
                entries = []
                for visit in range(last - first):
                    index = first + visit
                    span = slice(int(batch.visit_ptr[index]), int(batch.visit_ptr[index + 1]))
                    codes = batch.codes[span].astype(np.int64)
                    if not codes.size:
                        continue
                    embeddings = weights[torch.from_numpy(codes).to(weights.device)]
                    scaled = embeddings * torch.from_numpy(beta[position, visit]).to(weights.device)
                    contributions = (scaled @ w_output).cpu().numpy() * float(alpha[position, visit])
                    for code, contribution in zip(codes.tolist(), contributions.tolist()):
                        entries.append(
                            {
                                "visit_index": visit,
                                "visit_time": _timestamp(batch.visit_time[index]),
                                "code": code_strings[code],
                                "alpha": float(alpha[position, visit]),
                                "contribution": float(contribution),
                            }
                        )
                entries.sort(key=lambda entry: abs(entry["contribution"]), reverse=True)
                handle.write(
                    json.dumps(
                        {
                            "subject_id": int(batch.keys[row, 0]),
                            "prediction_time": _timestamp(batch.keys[row, 1]),
                            "predicted_boolean_probability": float(probabilities[position]),
                            "n_visits": last - first,
                            "top_contributions": entries[:top_k],
                        }
                    )
                    + "\n"
                )
                written += 1
    return written
