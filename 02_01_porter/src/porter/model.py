"""PORTER — language-grounded event representations for portable structured EHR foundation models.

Reimplemented from **arXiv:2606.24102** (Guo, Yan, Vettese & Sung, 2026). There is no public repository
and no Python package for PORTER, so nothing here wraps an upstream implementation: every element is
built from the paper's description, and :doc:`../IMPLEMENTATION_REPORT` records for each one whether it
was ported, adapted or omitted. Section numbers in the comments below refer to that paper.

The three components of §2.4, and where each one lives:

1. **Frozen text encoder** (§2.4.1). Every vocabulary entry's event description is rendered by
   :func:`render_event_description`, encoded *once* by a frozen sentence encoder in
   :func:`build_text_table`, and cached as the ``text_table`` buffer. Nothing about the model's input
   depends on a learned per-code embedding — that is the whole point of PORTER, and it is why this module
   does **not** use ``meds_model_base.lightning.modules.CodeEmbedder``.
2. **Numeric pathway + FiLM** (§2.4.2). :meth:`Model.numeric_features` builds the paper's four-dimensional
   feature vector (one scalar, three indicators); a 128-128 GELU MLP and two zero-initialised heads produce
   ``gamma`` and ``beta``; the fusion is ``h = (1 + gamma) * h_text + beta``.
3. **Temporal backbone** (§2.4.3). A causal decoder-only Transformer (SwiGLU, RMSNorm, RoPE over patient
   age in days, alternating global and 128-wide local attention) trained by next-event prediction (§2.4.4).

The DAG this repository implements calls exactly three things:

- ``compute_loss(batch)`` — used by ``pretrain`` **only**. Unlike the ``finetune`` profile, this DAG's
  ``supervised_train`` is ``ProbeTrainCommand``, which builds :class:`Probe` from
  ``configs/model/probe.yaml`` and never touches :class:`Model`. There is therefore no ``has_labels``
  branch here; a labelled batch is an error rather than a second training mode.
- ``encode(batch)`` — the representation ``infer`` materialises as embeddings (§2.7.1: "hidden states were
  extracted at prediction time").
- :class:`Probe` — the L2-regularised logistic regression ``supervised_train`` fits on those frozen
  embeddings and ``predict`` scores with (§2.7.1).

The pretraining head (:attr:`Model.head`) is discarded downstream by construction: :meth:`Model.encode`
never calls it, and ``infer`` calls only ``encode`` (§2.4.4, "the output head was therefore discarded
before downstream representation extraction").
"""

import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from meds_model_base.lightning.modules import BaseLightningModule, masked_mean, padding_mask
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

PAD_INDEX = MEDSTorchBatch.PAD_INDEX

# ----------------------------------------------------------------------------------------------------
# Published constants. Everything in this block is a number the paper states (or that its own parameter
# arithmetic pins); nothing here is tuned.
# ----------------------------------------------------------------------------------------------------

#: §2.4.3 and Supplementary Table S4: 28 layers, hidden 768, 12 heads, intermediate 768 (equal to hidden),
#: SwiGLU, alternating global/local attention starting global, local window 128.
#: :func:`backbone_parameter_count` checks this reproduces the paper's stated 115.6 M backbone.
#: **This is the shipped default** (``configs/model/default.yaml``); running PORTER means running this.
PAPER_BACKBONE: dict[str, Any] = {
    "hidden_size": 768,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "intermediate_size": 768,
    "local_attention_window": 128,
}

#: What :class:`Model` falls back to when ``backbone`` is not given: the published geometry, so a model
#: built with no configuration at all is PORTER rather than a toy.
DEFAULT_BACKBONE: dict[str, Any] = PAPER_BACKBONE

#: The published *shape* at 1/6 the width and 1/7 the depth — every ratio the paper fixes is preserved
#: (intermediate == hidden, head_dim = hidden / heads, alternating global/local starting global,
#: window 128), but small enough to train on CPU in seconds. This is what the conformance and
#: learnability tiers run (``tests/conftest.py`` selects it through ``PORTER_*`` environment variables)
#: and what ``model=small`` selects. It is a test fixture, not a model: no result should be reported
#: from it.
SMALL_BACKBONE: dict[str, Any] = {
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "local_attention_window": 128,
}

#: §2.4.3: "The maximum effective context window was therefore 16,384 clinical events per patient."
PAPER_MAX_SEQ_LEN = 16_384

#: Supplementary Table S4: "Max clinical events per batch 16,384", "Min patients per batch 1".
PAPER_EVENT_BUDGET = 16_384

#: §2.4.1: the primary text encoder, selected over BGE-M3 and Qwen3-Embedding-8B on pretraining validation
#: loss. 768-dimensional, which is exactly the published backbone width.
PAPER_TEXT_ENCODER = "FremyCompany/BioLORD-2023"
PAPER_TEXT_DIM = 768

#: §2.6.1's non-semantic control arm ("PORTER-Random"): each unique rendered description gets a fixed
#: standard-normal vector, L2-normalised, deterministically keyed by the string. Needs no network.
RANDOM_TEXT_ENCODER = "random"

#: §2.4.2, verbatim: ``log(1 + |v|) / 7`` clipped to ``[0, 1.5]``; the divisor maps ``log(1 + 1000) ≈ 6.9``
#: to about 1.0.
LOG_FALLBACK_DIVISOR = 7.0
LOG_FALLBACK_CLIP = (0.0, 1.5)

#: §2.4.2, verbatim: position within the reference range (0 = lower limit, 1 = upper), clipped to [-2, 3].
REF_RANGE_CLIP = (-2.0, 3.0)

#: §2.4.4 and Supplementary Table S4. Used only when no ``optimizer`` factory is supplied, so that a
#: ``Model`` built outside Hydra still trains with the paper's settings rather than torch defaults.
PAPER_LR = 5e-4
PAPER_WEIGHT_DECAY = 0.05
PAPER_BETAS = (0.9, 0.95)

#: How ``[B, L, H]`` becomes the ``[B, H]`` vector ``infer`` materialises.
POOLINGS = ("last", "mean")

#: MEDS code prefixes → the paper's nine event-description categories (Supplementary Table S3). MEDS has
#: already flattened OMOP's per-domain tables into a single ``code`` column, so the paper's slot mapping
#: (``measurement_concept_name``, ``drug_route_concept_name``, …) has no direct counterpart; this table is
#: the documented substitute. Unlisted prefixes are title-cased as-is.
CATEGORY_ALIASES: dict[str, str] = {
    "MEAS": "Measurement",
    "MEASUREMENT": "Measurement",
    "LAB": "Measurement",
    "LABS": "Measurement",
    "VITAL": "Measurement",
    "VITALS": "Measurement",
    "OBS": "Observation",
    "OBSERVATION": "Observation",
    "COND": "Condition",
    "CONDITION": "Condition",
    "DIAGNOSIS": "Condition",
    "DX": "Condition",
    "ICD": "Condition",
    "PROC": "Procedure",
    "PROCEDURE": "Procedure",
    "CPT": "Procedure",
    "DRUG": "Drug",
    "MED": "Drug",
    "MEDICATION": "Drug",
    "RX": "Drug",
    "NOTE": "Note",
    "SPECIMEN": "Specimen",
    "VISIT": "Visit",
    "ADMISSION": "Visit",
    "DISCHARGE": "Visit",
    "TRANSFER": "Visit",
    "ICU": "Visit",
    "GENDER": "Patient Sex",
    "SEX": "Patient Sex",
    "RACE": "Patient Race",
    "ETHNICITY": "Patient Ethnicity",
}

#: Codes the paper renders as a fixed string with no slot at all (Supplementary Table S3, demographics).
FIXED_DESCRIPTIONS: dict[str, str] = {
    "MEDS_BIRTH": "Patient Birth Date",
    "MEDS_DEATH": "Patient Death Date",
}

#: Column pairs a MEDS code-metadata file may use to carry a per-code reference range. MEDS does not
#: standardise one, so the numeric pathway's reference-range branch (§2.4.2) is reachable only when a
#: dataset supplies one of these; otherwise every value takes the log-magnitude fallback.
REFERENCE_RANGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("reference_low", "reference_high"),
    ("values/reference_low", "values/reference_high"),
    ("reference_range_low", "reference_range_high"),
    ("normal_range_low", "normal_range_high"),
)


# ----------------------------------------------------------------------------------------------------
# The input pathway: MEDS code metadata -> rendered event descriptions -> a frozen embedding table.
# ----------------------------------------------------------------------------------------------------


def render_event_description(code: str, description: str | None = None) -> str:
    """Render one vocabulary entry as the natural-language event description PORTER encodes (§2.4.1).

    The paper builds descriptions from OMOP columns with nine per-domain templates
    (Supplementary Table S3): ``"Measurement: {concept name}"``, ``"Drug: {name}. Route: {route}"``, and so
    on. MEDS has already flattened those tables into ``code`` plus an optional ``description``, so this is
    the documented adaptation of that table: the first ``//``-segment picks the category (via
    :data:`CATEGORY_ALIASES`) and the description — or the remaining segments — fills the concept-name slot.

    A ``description`` equal to the code carries no information beyond the code itself, so it is treated as
    absent; ``meds_model_base.testing.synthetic`` writes exactly that, and so do MEDS ETLs with no concept
    dictionary.

    Examples:
        The paper's own worked example (§2.4.1) and its fixed demographic strings:

        >>> render_event_description("LAB//HGB", "Hemoglobin [Mass/volume] in Blood")
        'Measurement: Hemoglobin [Mass/volume] in Blood'
        >>> render_event_description("MEDS_BIRTH")
        'Patient Birth Date'
        >>> render_event_description("MEDS_DEATH")
        'Patient Death Date'
        >>> render_event_description("GENDER//F")
        'Patient Sex: F'

        Categories are recognised case-insensitively, and an unknown prefix is title-cased rather than
        dropped, so no vocabulary entry ever loses its domain:

        >>> render_event_description("drug//aspirin", "Aspirin 81 MG Oral Tablet")
        'Drug: Aspirin 81 MG Oral Tablet'
        >>> render_event_description("LOINC//718-7")
        'Loinc: 718-7'

        A description that merely repeats the code is treated as absent:

        >>> render_event_description("SIGNAL//POS", "SIGNAL//POS")
        'Signal: POS'

        A code with no separator takes its description as-is, and falls back to its own domain when
        there is none — both shapes occur in real MEDS vocabularies:

        >>> render_event_description("HR", "Heart rate")
        'Heart rate'
        >>> render_event_description("DISCHARGE")
        'Visit: Discharge'
        >>> render_event_description("HEIGHT")
        'HEIGHT'
    """
    code = (code or "").strip()
    if code in FIXED_DESCRIPTIONS:
        return FIXED_DESCRIPTIONS[code]

    text = (description or "").strip()
    if text == code:  # a description that repeats the code adds nothing; treat it as missing
        text = ""

    segments = [seg for seg in code.split("//") if seg]
    if len(segments) < 2:
        # A single-segment code still has a domain if the whole code names one — MEDS writes bare
        # `DISCHARGE` and `VISIT` as often as it writes `ADMISSION//CARDIAC`.
        category = CATEGORY_ALIASES.get(code.upper())
        if category and not text:
            return f"{category}: {code.replace('_', ' ').title()}"
        return text or code or "Unknown event"

    prefix = segments[0]
    category = CATEGORY_ALIASES.get(prefix.upper(), prefix.replace("_", " ").title())
    name = text or " ".join(seg.replace("_", " ") for seg in segments[1:])
    return f"{category}: {name}" if name else category


def _random_embeddings(descriptions: Sequence[str], dim: int) -> Tensor:
    """§2.6.1's PORTER-Random arm: a fixed unit-norm normal vector keyed by the description string.

    "Vectors were deterministically keyed by the rendered event-description string, so identical
    event-description strings received identical vectors across vocabularies, whereas semantically
    equivalent but textually distinct event-description strings received uncorrelated vectors."

    Examples:
        >>> import torch
        >>> a = _random_embeddings(["Measurement: Hemoglobin", "Condition: Asthma"], 8)
        >>> a.shape
        torch.Size([2, 8])
        >>> bool(torch.allclose(a.norm(dim=-1), torch.ones(2)))
        True

        Keyed by the string, so it is stable across runs, processes and vocabularies:

        >>> b = _random_embeddings(["Condition: Asthma"], 8)
        >>> bool(torch.allclose(a[1], b[0]))
        True
    """
    rows = []
    for text in descriptions:
        seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
        vector = np.random.default_rng(seed).standard_normal(dim)
        rows.append(vector / np.linalg.norm(vector))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _encode_descriptions(descriptions: Sequence[str], text_encoder: str, d_text: int) -> Tensor:
    """Encode rendered descriptions with the frozen text encoder, once (§2.4.1).

    ``sentence_transformers`` is imported here rather than at module scope on purpose: ``pytest`` runs with
    ``--doctest-modules`` over ``src``, so importing this file must stay cheap and must not reach the
    network. It is also only ever reached by ``pretrain`` — see :meth:`Model.__init__`.
    """
    if text_encoder == RANDOM_TEXT_ENCODER:
        return _random_embeddings(descriptions, d_text)

    from sentence_transformers import SentenceTransformer

    encoded = SentenceTransformer(text_encoder).encode(list(descriptions))
    table = torch.as_tensor(np.asarray(encoded), dtype=torch.float32)
    if table.shape[-1] != d_text:
        raise ValueError(
            f"Text encoder {text_encoder!r} produces {table.shape[-1]}-dimensional embeddings, but the "
            f"model was built with d_text={d_text}. Set model.d_text={table.shape[-1]}."
        )
    return table


def build_text_table(
    code_metadata_fp: str,
    vocab_size: int,
    text_encoder: str = PAPER_TEXT_ENCODER,
    d_text: int = PAPER_TEXT_DIM,
) -> Tensor:
    """Build PORTER's cached event-description embedding table from a tensorized cohort's code metadata.

    "Each event description was encoded once using a frozen text encoder and cached as a static
    event-description embedding table" (§2.4.1). Row ``i`` of the result is the embedding of the
    description of the code whose ``code/vocab_index`` is ``i``; row :data:`PAD_INDEX` and any index the
    metadata does not mention stay zero.

    Args:
        code_metadata_fp: ``<data_dir>/patients/metadata/codes.parquet``, written by ``preprocess_data``.
        vocab_size: ``datamodule.config.vocab_size`` — PAD-inclusive, so the table is ``[vocab_size, d_text]``
            and is indexed directly by ``batch.code``.
        text_encoder: a ``sentence-transformers`` model id, or :data:`RANDOM_TEXT_ENCODER`.
        d_text: the encoder's output width.

    Examples:
        >>> import polars as pl, tempfile, torch
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({
        ...         "code": ["LAB//HGB", "COND//ASTHMA"],
        ...         "description": ["Hemoglobin", None],
        ...         "code/vocab_index": [1, 2],
        ...     }).write_parquet(fp)
        ...     table = build_text_table(str(fp), vocab_size=4, text_encoder="random", d_text=8)
        >>> table.shape
        torch.Size([4, 8])

        The padding row is zero, and so is any vocabulary index the metadata does not describe — a code
        with no embedding must contribute nothing rather than a random direction:

        >>> bool(table[0].abs().sum() == 0), bool(table[3].abs().sum() == 0)
        (True, True)
        >>> bool(table[1].abs().sum() > 0), bool(table[2].abs().sum() > 0)
        (True, True)
    """
    codes = pl.read_parquet(code_metadata_fp)
    if "code/vocab_index" not in codes.columns:
        raise ValueError(
            f"{code_metadata_fp} has no `code/vocab_index` column, so event descriptions cannot be aligned "
            "to batch.code. Point code_metadata_fp at a tensorized cohort's metadata/codes.parquet."
        )
    descriptions = codes.get_column("description") if "description" in codes.columns else None
    indices = codes.get_column("code/vocab_index").cast(pl.Int64).to_list()
    rendered = [
        render_event_description(code, desc)
        for code, desc in zip(
            codes.get_column("code").to_list(),
            descriptions.to_list() if descriptions is not None else [None] * len(codes),
            strict=True,
        )
    ]

    table = torch.zeros(vocab_size, d_text)
    encoded = _encode_descriptions(rendered, text_encoder, d_text)
    for index, row in zip(indices, encoded, strict=True):
        if 0 <= index < vocab_size and index != PAD_INDEX:
            table[index] = row
    return table


def build_value_statistics(code_metadata_fp: str, vocab_size: int) -> tuple[Tensor, Tensor]:
    """Recover the per-code mean and standard deviation meds-torch-data normalised ``numeric_value`` with.

    PORTER's numeric scalar is defined on the **raw** clinical value (§2.4.2), but meds-torch-data's
    tensorization runs MEDS-transforms' ``normalization`` stage, which replaces every ``numeric_value``
    with ``(value - mean) / std`` computed per code. Those statistics survive in ``codes.parquet``, so the
    model inverts the transform rather than applying a log-magnitude scalar to a z-score.

    Codes with no observed values (or a degenerate standard deviation) get ``mean=0, std=1``, i.e. the
    identity — they carry no numeric value for the inverse to matter to.

    Examples:
        >>> import polars as pl, tempfile, torch
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({
        ...         "code": ["LAB//HGB", "COND//ASTHMA"],
        ...         "code/vocab_index": [1, 2],
        ...         "values/n_occurrences": [4, 0],
        ...         "values/sum": [40.0, 0.0],
        ...         "values/sum_sqd": [404.0, 0.0],
        ...     }).write_parquet(fp)
        ...     mean, std = build_value_statistics(str(fp), vocab_size=3)
        >>> [round(v, 3) for v in mean.tolist()]
        [0.0, 10.0, 0.0]
        >>> [round(v, 3) for v in std.tolist()]
        [1.0, 1.0, 1.0]

        A z-score of 1 for code 1 therefore inverts to ``10 * 1 + ... `` — the round trip is exact:

        >>> raw = torch.tensor([13.0])
        >>> z = (raw - mean[1]) / std[1]
        >>> bool(torch.allclose(z * std[1] + mean[1], raw))
        True
    """
    mean = torch.zeros(vocab_size)
    std = torch.ones(vocab_size)
    codes = pl.read_parquet(code_metadata_fp)
    required = {"code/vocab_index", "values/n_occurrences", "values/sum", "values/sum_sqd"}
    if not required.issubset(codes.columns):
        return mean, std

    frame = codes.select(
        pl.col("code/vocab_index").cast(pl.Int64),
        pl.col("values/n_occurrences").cast(pl.Float64),
        pl.col("values/sum").cast(pl.Float64),
        pl.col("values/sum_sqd").cast(pl.Float64),
    )
    for index, n, total, total_sqd in frame.iter_rows():
        if not (0 <= index < vocab_size) or index == PAD_INDEX or not n:
            continue
        code_mean = total / n
        variance = total_sqd / n - code_mean**2
        if not math.isfinite(variance) or variance <= 0:
            continue
        mean[index] = code_mean
        std[index] = math.sqrt(variance)
    return mean, std


def build_composite_index(
    code_metadata_fp: str, vocab_size: int, value_bins: int
) -> tuple[Tensor, Tensor, int]:
    """Lay out PORTER's composite pretraining vocabulary (§2.4.4).

    "Each composite token combined the primary OMOP concept ID with discretized event attributes,
    including numeric values discretized into 10 quantile bins **where applicable**."

    "Where applicable" is what keeps this small: a concept that never carries a numeric value gets one
    slot, and one that does gets ``value_bins`` of them. A dense concept x bin cross product would be an
    order of magnitude larger than the paper's ~98,300-token vocabulary.

    Returns ``(offsets, boundaries, size)``: ``offsets[c]`` is where code ``c``'s block starts,
    ``boundaries[c]`` holds its ``value_bins - 1`` interior cut points (``nan`` when it has none), and
    ``size`` is the total number of composite classes — the width of the prediction head.

    The boundaries come from ``values/quantiles`` in the tensorized code metadata, which
    ``pipelines/quantiles.yaml`` fits before tensorization. Without that column every code gets one slot
    and the composite target degrades exactly to the bare code.

    Examples:
        >>> import polars as pl, tempfile, torch
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({
        ...         "code": ["LAB//HGB", "COND//ASTHMA"],
        ...         "code/vocab_index": [1, 2],
        ...         "values/quantiles": [
        ...             {"values/quantile/0.5": 10.0}, None,
        ...         ],
        ...     }).write_parquet(fp)
        ...     offsets, bounds, size = build_composite_index(str(fp), vocab_size=3, value_bins=2)

        Code 1 carries values, so it gets two slots; code 2 gets one. PAD keeps slot 0.

        >>> offsets.tolist()
        [0, 1, 3]
        >>> size
        4
        >>> bounds[1].tolist(), bool(bounds[2].isnan().all())
        ([10.0], True)

        With no quantile column at all, every code gets one slot and the layout is the identity:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({"code": ["A"], "code/vocab_index": [1]}).write_parquet(fp)
        ...     offsets, bounds, size = build_composite_index(str(fp), vocab_size=3, value_bins=10)
        >>> offsets.tolist(), size
        ([0, 1, 2], 3)
    """
    if value_bins < 1:
        raise ValueError(f"value_bins must be at least 1, got {value_bins}.")

    boundaries = torch.full((vocab_size, max(value_bins - 1, 1)), float("nan"))
    slots = torch.ones(vocab_size, dtype=torch.long)

    codes = pl.read_parquet(code_metadata_fp)
    if "values/quantiles" in codes.columns and value_bins > 1:
        frame = codes.select(pl.col("code/vocab_index").cast(pl.Int64), pl.col("values/quantiles"))
        for index, quantiles in frame.iter_rows():
            if not (0 <= index < vocab_size) or index == PAD_INDEX or not quantiles:
                continue
            cuts = [v for _, v in sorted(quantiles.items()) if v is not None]
            if len(cuts) != value_bins - 1:
                continue
            boundaries[index] = torch.tensor(cuts, dtype=torch.float32)
            slots[index] = value_bins

    offsets = torch.cat([torch.zeros(1, dtype=torch.long), slots.cumsum(0)[:-1]])
    return offsets, boundaries, int(slots.sum())


def build_reference_ranges(code_metadata_fp: str, vocab_size: int) -> tuple[Tensor, Tensor]:
    """Read a per-code reference range from code metadata, if the dataset carries one (§2.4.2).

    MEDS standardises no reference-range column, so this looks for any of :data:`REFERENCE_RANGE_COLUMNS`
    and returns all-``nan`` when none is present — which gates the reference-range branch off and sends
    every value down the log-magnitude fallback. That is the case for every dataset this repository is
    tested against; see the implementation report.

    Examples:
        >>> import polars as pl, tempfile, torch
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({"code": ["LAB//HGB"], "code/vocab_index": [1]}).write_parquet(fp)
        ...     low, high = build_reference_ranges(str(fp), vocab_size=2)
        >>> bool(low.isnan().all()), bool(high.isnan().all())
        (True, True)

        With a range present, only the described codes become finite:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     fp = Path(tmp) / "codes.parquet"
        ...     _ = pl.DataFrame({
        ...         "code": ["LAB//HGB"],
        ...         "code/vocab_index": [1],
        ...         "reference_low": [12.0],
        ...         "reference_high": [16.0],
        ...     }).write_parquet(fp)
        ...     low, high = build_reference_ranges(str(fp), vocab_size=2)
        >>> low.tolist(), high.tolist()
        ([nan, 12.0], [nan, 16.0])
    """
    low = torch.full((vocab_size,), float("nan"))
    high = torch.full((vocab_size,), float("nan"))
    codes = pl.read_parquet(code_metadata_fp)
    pair = next((p for p in REFERENCE_RANGE_COLUMNS if set(p).issubset(codes.columns)), None)
    if pair is None or "code/vocab_index" not in codes.columns:
        return low, high

    frame = codes.select(
        pl.col("code/vocab_index").cast(pl.Int64),
        pl.col(pair[0]).cast(pl.Float64).alias("low"),
        pl.col(pair[1]).cast(pl.Float64).alias("high"),
    )
    for index, lo, hi in frame.iter_rows():
        if not (0 <= index < vocab_size) or index == PAD_INDEX or lo is None or hi is None:
            continue
        low[index] = lo
        high[index] = hi
    return low, high


# ----------------------------------------------------------------------------------------------------
# The temporal backbone (§2.4.3): causal decoder, SwiGLU, RMSNorm, RoPE over patient age in days,
# alternating global and 128-wide local attention.
# ----------------------------------------------------------------------------------------------------


def backbone_parameter_count(cfg: dict[str, Any]) -> int:
    """Count the backbone parameters :class:`Model` builds for ``cfg`` — the paper's headline figure.

    Per layer: four ``hidden x hidden`` attention projections and three ``hidden x intermediate`` SwiGLU
    projections, all bias-free, plus two RMSNorm gains; then one final RMSNorm gain.

    Examples:
        The published geometry reproduces §2.4.3's "115.6 million backbone parameters":

        >>> backbone_parameter_count(PAPER_BACKBONE)
        115649280
        >>> round(backbone_parameter_count(PAPER_BACKBONE) / 1e6, 2)
        115.65

        The projection weights alone are ``28 * 7 * 768**2``; Supplementary Table S6's stated 115.63 M sits
        between that and the count above, so the paper appears to carry one RMSNorm gain per layer rather
        than the two a pre-norm block needs. The 21,504-parameter gap is 0.019%:

        >>> 28 * 7 * 768**2
        115605504

        The published geometry is what ships (``DEFAULT_BACKBONE is PAPER_BACKBONE``); the test-only
        geometry is the same shape two orders of magnitude smaller:

        >>> backbone_parameter_count(DEFAULT_BACKBONE) == backbone_parameter_count(PAPER_BACKBONE)
        True
        >>> backbone_parameter_count(SMALL_BACKBONE)
        459904
    """
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    layers = int(cfg["num_hidden_layers"])
    per_layer = 4 * hidden * hidden + 3 * hidden * inter + 2 * hidden
    return layers * per_layer + hidden


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``[B, heads, L, head_dim]`` queries/keys by the per-position angles in ``cos``/``sin``.

    The angles are computed in float32 (patient age in days spans orders of magnitude, and computing the
    rotation in half precision loses the small deltas), then cast down to ``x``. Without that cast the
    product promotes back to float32 and the next ``Linear`` fails under a true-half precision such as
    ``trainer.precision=bf16-true``.
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class RotaryTimeEmbedding(nn.Module):
    """RoPE angles from *elapsed time*, not token index (§2.4.3).

    "Temporal information was encoded using rotary position embeddings (RoPE) applied to patient age in
    days, so relative positional rotations reflected elapsed time between events." Patient age in days is
    the cumulative sum of ``batch.time_delta_days``. RoPE is a *relative* encoding, so the arbitrary offset
    introduced by a window that starts mid-timeline cancels in every query-key inner product.
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head dimension, got {head_dim}.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        angles = positions.unsqueeze(-1).float() * self.inv_freq
        return angles.cos(), angles.sin()


class SwiGLU(nn.Module):
    """The backbone's feed-forward block (Supplementary Table S4: "Activation SwiGLU").

    Three bias-free projections, which is what makes the published 28 x 768 geometry come to 115.6 M
    parameters — see :func:`backbone_parameter_count`.
    """

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PorterBlock(nn.Module):
    """One pre-norm decoder layer: RMSNorm → causal RoPE attention → RMSNorm → SwiGLU.

    The paper never names the normalisation type or its placement (see the implementation report);
    pre-norm RMSNorm is the choice its parameter arithmetic supports. Attention has no bias, no QK-norm and
    no grouped-query sharing — none of which the paper mentions.
    """

    def __init__(self, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden_size {hidden} is not divisible by num_attention_heads {heads}.")
        self.heads = heads
        self.head_dim = hidden // heads
        self.attn_norm = nn.RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = nn.RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, intermediate)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, window: int | None, impl: str) -> Tensor:
        b, length, _ = x.shape
        h = self.attn_norm(x)
        shape = (b, length, self.heads, self.head_dim)
        q = self.q_proj(h).view(shape).transpose(1, 2)
        k = self.k_proj(h).view(shape).transpose(1, 2)
        v = self.v_proj(h).view(shape).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = causal_attention(q, k, v, window=window, impl=impl)
        x = x + self.o_proj(attended.transpose(1, 2).reshape(b, length, -1))
        return x + self.ffn(self.ffn_norm(x))


#: How :func:`causal_attention` computes a sliding-window layer. ``auto`` picks ``flex`` on CUDA for long
#: sequences and ``dense`` otherwise; ``dense`` is the reference implementation and is what the equivalence
#: test compares against.
ATTENTION_IMPLS = ("auto", "dense", "flex")

#: Above this sequence length ``auto`` stops materialising an ``L x L`` mask. At the paper's 16,384-event
#: context that mask alone is 268 MB per attention call, which is what makes the dense path unusable at
#: full scale rather than merely slow.
FLEX_ATTENTION_MIN_LEN = 1024

#: Single-entry caches. Both masks depend only on ``(length, window, device)`` — never on the batch — so
#: they are worth reusing, but only *one* is worth keeping: meds-torch-data pads to the longest sequence
#: in each batch, so the length changes constantly, and an `L x L` bool mask is 268 MiB at the paper's
#: 16,384-event context. A 32-entry cache of those is 8 GiB of GPU memory, which is how this was first
#: written. Rebuilding is a handful of elementwise ops.
_BAND_MASK_CACHE: dict[tuple[int, int, str], Tensor] = {}
_BLOCK_MASK_CACHE: dict[tuple[int, int, str], Any] = {}
_FLEX_ATTENTION = None
_FLEX_UNAVAILABLE_REASON: str | None = None


def require_right_padding(keep: Tensor) -> None:
    """Assert padding only ever sits at the *end* of a sequence.

    Everything below depends on it. meds-torch-data right-pads (verified for both ``RANDOM`` and
    ``TO_END`` sampling), and given right padding a causal mask already prevents every real query from
    attending to a pad key — pad keys come strictly after real ones. That is what lets the global layers
    run as ``is_causal=True`` with **no mask tensor at all**, which is the only form
    ``scaled_dot_product_attention`` can serve from a FlashAttention kernel.

    If that ever stops holding, the model would silently attend across padding. Hence a check rather than
    a comment: it costs one elementwise comparison per batch.

    Examples:
        >>> import torch
        >>> require_right_padding(torch.tensor([[True, True, False]]))
        >>> require_right_padding(torch.tensor([[True, False, True]]))
        Traceback (most recent call last):
            ...
        ValueError: This model assumes right-padded sequences...
    """
    if keep.shape[1] > 1 and bool((keep[:, :-1].int() < keep[:, 1:].int()).any()):
        raise ValueError(
            "This model assumes right-padded sequences (padding only at the end), but a batch has a real "
            "measurement after a pad position. Causal attention alone no longer excludes padding, so the "
            "backbone would attend across it. Check the meds-torch-data sampling strategy."
        )


def _cached(cache: dict, key: tuple[int, int, str], build):
    """One-entry memo: hit, or rebuild and evict. See :data:`_BAND_MASK_CACHE` for why one."""
    hit = cache.get(key)
    if hit is None:
        cache.clear()
        hit = cache[key] = build()
    return hit


def _band_mask(length: int, window: int, device: torch.device) -> Tensor:
    """A ``[1, 1, L, L]`` causal sliding-window mask; ``True`` means *attend*.

    Independent of the batch — padding needs no term, because causality already excludes it (see
    :func:`require_right_padding`) — so one mask serves every sample and every layer of the same width.
    """

    def build() -> Tensor:
        rows = torch.arange(length, device=device).unsqueeze(1)
        cols = torch.arange(length, device=device).unsqueeze(0)
        return ((cols <= rows) & (rows - cols < window))[None, None]

    return _cached(_BAND_MASK_CACHE, (length, window, str(device)), build)


def _flex_attention_fn():
    """Return a compiled ``flex_attention``, or ``None`` if it is unavailable on this install.

    ``dynamic=True`` is load-bearing, not a preference. Sequence length changes on almost every batch
    (meds-torch-data pads to the batch maximum), and under ``dynamic=False`` each distinct length is a
    fresh compile. Dynamo gives up after ``recompile_limit`` (8) and silently runs ``flex_attention``
    *eagerly* — which materialises the full score matrix, 39 GiB for one layer at the paper's context.
    The fallback below would then swallow the resulting OOM into the dense path, leaving a run that is
    correct, quadratic, and quiet. With ``dynamic=True`` it compiles once and serves every length.
    """
    global _FLEX_ATTENTION, _FLEX_UNAVAILABLE_REASON
    if _FLEX_ATTENTION is not None or _FLEX_UNAVAILABLE_REASON is not None:
        return _FLEX_ATTENTION
    try:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    except Exception as exc:  # pragma: no cover - depends on the torch build
        _FLEX_UNAVAILABLE_REASON = str(exc)
        logger.warning("flex_attention is unavailable (%s); falling back to a dense mask.", exc)
    return _FLEX_ATTENTION


def _flex_block_mask(length: int, window: int, device: torch.device):
    """A cached block-sparse causal-window mask. ``create_block_mask`` is not cheap: 229 ms and 2.6 GiB
    transient at ``L = 16384``, which is per *layer* per *forward* if it is not memoised."""

    def build():
        from torch.nn.attention.flex_attention import create_block_mask

        def sliding_causal(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx - kv_idx < window)

        return create_block_mask(sliding_causal, B=None, H=None, Q_LEN=length, KV_LEN=length, device=device)

    return _cached(_BLOCK_MASK_CACHE, (length, window, str(device)), build)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, *, window: int | None, impl: str = "auto") -> Tensor:
    """Causal attention over ``[B, heads, L, head_dim]``, optionally limited to a sliding ``window``.

    Three paths, all computing the same function (``tests/test_attention.py`` asserts they agree):

    - **global** (``window is None``) — ``scaled_dot_product_attention(..., is_causal=True)``. No mask
      tensor, so PyTorch is free to dispatch to a FlashAttention kernel; memory is linear in ``L``.
    - **local, dense** — a cached ``[1, 1, L, L]`` band mask. Exact and simple, but quadratic in memory:
      268 MB at ``L = 16384``, which is why it is not the default at full scale.
    - **local, flex** — ``torch.compile``'d ``flex_attention`` with a block-sparse causal-window mask.
      Memory is linear in ``L``; this is the path a paper-scale run takes.

    ``auto`` chooses ``flex`` on CUDA once ``L`` exceeds :data:`FLEX_ATTENTION_MIN_LEN` and ``dense``
    otherwise, and falls back to ``dense`` if flex raises — a compile failure on an unfamiliar cluster
    should degrade throughput, not the run. ``flex`` never falls back: it is how a cluster smoke test
    finds out that the kernel does not work here.
    """
    global _FLEX_UNAVAILABLE_REASON

    if window is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    length = q.shape[-2]
    if window >= length:
        # The window covers the whole prefix, so this layer *is* the global one. Take the fast path.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    use_flex = impl == "flex" or (impl == "auto" and q.is_cuda and length >= FLEX_ATTENTION_MIN_LEN)
    flex = _flex_attention_fn() if use_flex else None
    if impl == "flex" and flex is None:
        raise RuntimeError(
            "attention_impl='flex' was requested but flex_attention is unavailable here "
            f"({_FLEX_UNAVAILABLE_REASON}). Use 'auto' to fall back to a dense mask, or fix the "
            "environment — Triton needs Python development headers, a C compiler and a writable cache."
        )
    if flex is not None:
        try:
            return flex(q, k, v, block_mask=_flex_block_mask(length, window, q.device))
        except Exception as exc:  # pragma: no cover - compile/runtime environment dependent
            if impl == "flex":
                raise
            # Latch the failure. `torch.compile` is lazy, so the import-time guard above almost never
            # fires; the realistic failures (no Triton shim, no compiler, unwritable cache) surface
            # here, on the first real call. Without latching, every layer of every batch retries a
            # kernel that cannot work and pays the compile attempt again.
            _FLEX_UNAVAILABLE_REASON = str(exc)
            logger.warning("flex_attention failed (%s); falling back to a dense mask.", exc)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=_band_mask(length, window, q.device))


def _check_ar_target(batch: MEDSTorchBatch) -> None:
    """Reject a batch the next-event objective cannot score, before it becomes a ``nan`` loss.

    ``cross_entropy`` returns ``nan`` rather than raising when every target position is the ignored pad
    index, and a ``nan`` loss only becomes visible several steps later as a model of ``nan`` weights.

    Examples:
        >>> import torch
        >>> from meds_torchdata import MEDSTorchBatch
        >>> one_each = MEDSTorchBatch(
        ...     code=torch.tensor([[3], [4]]),
        ...     numeric_value=torch.zeros(2, 1),
        ...     numeric_value_mask=torch.zeros(2, 1, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 1),
        ... )
        >>> _check_ar_target(one_each)
        Traceback (most recent call last):
            ...
        ValueError: Next-event prediction needs a measurement to predict, but every subject in this
        batch has only one...
    """
    if batch.code.shape[1] > 1 and bool((batch.code[:, 1:] != PAD_INDEX).any()):
        return
    raise ValueError(
        "Next-event prediction needs a measurement to predict, but every subject in this batch has "
        f"only one (sequence length {batch.code.shape[1]}). Filter out subjects with too little history "
        "before tensorizing, or raise the batch size so the batch is not made up entirely of them."
    )


def _plain(cfg: Any) -> dict[str, Any]:
    """Normalise a Hydra ``DictConfig`` (or ``None``) to a plain dict, so it is checkpointable."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    from omegaconf import OmegaConf

    return dict(OmegaConf.to_container(cfg, resolve=True))


# ----------------------------------------------------------------------------------------------------
# The model.
# ----------------------------------------------------------------------------------------------------


class Model(BaseLightningModule):
    """PORTER: a frozen language-grounded input pathway feeding a causal temporal Transformer.

    Built by ``pretrain`` from ``configs/model/default.yaml``; ``infer`` and ``predict`` rebuild it from
    the checkpoint's own hyperparameters instead (``_runtime.load_trained_module`` reads ``module_class``
    from the artifact manifest and calls ``load_from_checkpoint``), so overriding ``model.*`` on those
    commands has no effect.

    ``code_metadata_fp`` is deliberately **not** a saved hyperparameter. The encoded table is a persistent
    buffer, so it travels inside the checkpoint: the frozen text encoder is loaded exactly once in the
    whole DAG, by ``pretrain``, and ``infer``/``predict`` reload without importing
    ``sentence_transformers`` or touching the network. That also makes the artifact self-contained — it
    still scores correctly if the workspace it was trained from is deleted.

    Args:
        vocab_size: injected by the training commands from ``datamodule.config.vocab_size``. PAD-inclusive.
        code_metadata_fp: ``<data_dir>/patients/metadata/codes.parquet``. Read once, at construction.
        text_encoder: a ``sentence-transformers`` id (§2.4.1 uses ``FremyCompany/BioLORD-2023``), or
            ``"random"`` for §2.6.1's non-semantic control arm.
        d_text: the encoder's output width; must match what it produces.
        backbone: overrides merged over :data:`DEFAULT_BACKBONE`.
        numeric_hidden: width of the numeric MLP's two layers (§2.4.2: 128).
        denormalize_numeric_values: invert meds-torch-data's per-code z-scoring so the numeric scalar sees
            raw magnitudes, as §2.4.2 defines it. See :func:`build_value_statistics`.
        rope_theta: RoPE base. Not stated by the paper.
        rope_time_scale: multiplies patient age in days before rotation. Not stated by the paper.
        pooling: ``"last"`` (the state at the prediction time, §2.7.1) or ``"mean"``.
        attention_impl: how sliding-window layers are computed — see :func:`causal_attention`. ``auto``
            picks a block-sparse kernel on CUDA at long context and a dense mask otherwise.
        warmup_steps: Supplementary Table S4's 150, clamped to a tenth of the run — see
            :meth:`configure_optimizers`.
        min_lr_ratio: the floor of ``cosine_with_min_lr``, as a fraction of peak. Not stated by the paper.

    Examples:
        A tiny model, built without any code metadata so the doctest needs no encoder:

        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> model = Model(vocab_size=8, d_text=16, backbone={"hidden_size": 16, "num_hidden_layers": 2,
        ...                                                  "num_attention_heads": 2,
        ...                                                  "intermediate_size": 16})
        >>> model.text_table.shape
        torch.Size([8, 16])

        The vocabulary carries no learned embedding at all — that is what makes PORTER portable, and it is
        checkable: no parameter is ``[vocab_size, hidden]``-shaped except the discarded pretraining head.

        >>> sum(p.numel() for n, p in model.named_parameters() if n.startswith("head"))
        136
        >>> model.text_table.requires_grad
        False
    """

    def __init__(
        self,
        vocab_size: int,
        code_metadata_fp: str | None = None,
        text_encoder: str = PAPER_TEXT_ENCODER,
        d_text: int = PAPER_TEXT_DIM,
        backbone: Any = None,
        numeric_hidden: int = 128,
        denormalize_numeric_values: bool = True,
        value_bins: int = 10,
        composite_size: int | None = None,
        rope_theta: float = 10_000.0,
        rope_time_scale: float = 1.0,
        pooling: str = "last",
        attention_impl: str = "auto",
        warmup_steps: int = 150,
        min_lr_ratio: float = 0.1,
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}.")
        if attention_impl not in ATTENTION_IMPLS:
            raise ValueError(f"attention_impl must be one of {ATTENTION_IMPLS}, got {attention_impl!r}.")
        # Deliberately *not* a saved hyperparameter: it describes the host, not the model. `infer` and
        # `predict` rebuild through `load_from_checkpoint` with no kwargs, so a checkpoint pretrained on
        # a Triton-capable node would otherwise carry `flex` onto a node that cannot compile it — and
        # `flex` re-raises by contract. Reloads get `auto`, which degrades instead.
        self.attention_impl = attention_impl

        cfg = {**DEFAULT_BACKBONE, **_plain(backbone)}
        vocab_size = int(vocab_size)
        value_bins = int(value_bins)

        # §2.4.4's composite output vocabulary. Read before `save_hyperparameters` because the head's
        # width depends on it, and a reload (which gets no `code_metadata_fp`) must be able to rebuild
        # the same shape from the saved value alone.
        if code_metadata_fp is not None:
            offsets, boundaries, composite_size = build_composite_index(
                code_metadata_fp, vocab_size, value_bins
            )
        else:
            composite_size = int(composite_size or vocab_size)
            offsets = torch.arange(vocab_size)
            boundaries = torch.full((vocab_size, max(value_bins - 1, 1)), float("nan"))
        d_text = int(d_text)
        hidden = int(cfg["hidden_size"])

        # `code_metadata_fp` is absent on purpose: reload must not re-read a workspace or an encoder.
        self.save_hyperparameters(
            {
                "vocab_size": vocab_size,
                "text_encoder": text_encoder,
                "d_text": d_text,
                "backbone": cfg,
                "numeric_hidden": int(numeric_hidden),
                "denormalize_numeric_values": bool(denormalize_numeric_values),
                "value_bins": value_bins,
                "composite_size": composite_size,
                "rope_theta": float(rope_theta),
                "rope_time_scale": float(rope_time_scale),
                "pooling": pooling,
                "warmup_steps": int(warmup_steps),
                "min_lr_ratio": float(min_lr_ratio),
            }
        )

        # Frozen state, all persistent so it rides in the checkpoint rather than being rebuilt downstream.
        self.register_buffer("text_table", torch.zeros(vocab_size, d_text))
        self.register_buffer("value_mean", torch.zeros(vocab_size))
        self.register_buffer("value_std", torch.ones(vocab_size))
        self.register_buffer("ref_low", torch.full((vocab_size,), float("nan")))
        self.register_buffer("ref_high", torch.full((vocab_size,), float("nan")))
        self.register_buffer("composite_offset", offsets)
        self.register_buffer("value_boundaries", boundaries)
        if code_metadata_fp is not None:
            self.text_table.copy_(build_text_table(code_metadata_fp, vocab_size, text_encoder, d_text))
            mean, std = build_value_statistics(code_metadata_fp, vocab_size)
            self.value_mean.copy_(mean)
            self.value_std.copy_(std)
            low, high = build_reference_ranges(code_metadata_fp, vocab_size)
            self.ref_low.copy_(low)
            self.ref_high.copy_(high)

        # §2.4.1: "projected to the backbone hidden dimension using a two-layer MLP with GELU activation".
        self.text_proj = nn.Sequential(nn.Linear(d_text, hidden), nn.GELU(), nn.Linear(hidden, hidden))

        # §2.4.2: "two 128-dimensional fully connected layers and GELU activations", then two linear heads
        # producing gamma and beta, "initialized to zero so that the fused representation was initially
        # equal to the text representation".
        self.numeric_mlp = nn.Sequential(
            nn.Linear(4, int(numeric_hidden)),
            nn.GELU(),
            nn.Linear(int(numeric_hidden), int(numeric_hidden)),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(int(numeric_hidden), hidden)
        self.beta_head = nn.Linear(int(numeric_hidden), hidden)
        for head in (self.gamma_head, self.beta_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        heads = int(cfg["num_attention_heads"])
        self.window = int(cfg["local_attention_window"])
        if self.window < 1:
            raise ValueError(
                f"local_attention_window must be at least 1, got {self.window}. A window of 0 masks "
                "every key including the diagonal, which zeroes the local layers rather than erroring."
            )
        self.rope = RotaryTimeEmbedding(hidden // heads, theta=float(rope_theta))
        self.layers = nn.ModuleList(
            PorterBlock(hidden, heads, int(cfg["intermediate_size"]))
            for _ in range(int(cfg["num_hidden_layers"]))
        )
        self.norm_f = nn.RMSNorm(hidden)
        #: §2.4.4's next-event prediction head, over the *composite* vocabulary. Discarded downstream:
        #: `encode` never calls it.
        self.head = nn.Linear(hidden, composite_size)

    # ------------------------------------------------------------------------------------------------
    # The input pathway: batch -> fused event representations.
    # ------------------------------------------------------------------------------------------------

    def numeric_features(self, batch: MEDSTorchBatch) -> Tensor:
        """PORTER's four-dimensional numeric feature vector per event (§2.4.2).

        "The numeric feature vector contained this scalar, indicators for reference-range scaling and
        log-magnitude fallback, and an indicator for numeric value presence. Exactly one scaling indicator
        was active when a numeric value was present. For events without numeric content, all indicators and
        the scalar were set to zero."

        Note ``batch.numeric_value_mask`` is documented as *undefined* at padding, and meds-torch-data in
        fact leaves it ``True`` there (padding gets ``0.0``, and the mask is ``~isnan``). Presence is
        therefore the mask **and** the padding mask; using the mask alone fires the presence indicator on
        every pad position.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> model = Model(vocab_size=4, d_text=8, denormalize_numeric_values=False,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 1,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})
            >>> batch = MEDSTorchBatch(
            ...     code=torch.tensor([[1, 2, 0]]),
            ...     numeric_value=torch.tensor([[1000.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False, True]]),
            ...     time_delta_days=torch.zeros(1, 3),
            ... )
            >>> feats = model.numeric_features(batch)
            >>> feats.shape
            torch.Size([1, 3, 4])

            The first event has a value and no reference range, so it takes the log-magnitude fallback:
            ``log(1 + 1000) / 7 ≈ 0.987``, the fallback indicator and the presence indicator are on, and
            the reference-range indicator is off.

            >>> [round(v, 3) for v in feats[0, 0].tolist()]
            [0.987, 0.0, 1.0, 1.0]

            The second event carries no value, so the scalar and all three indicators are zero:

            >>> feats[0, 1].tolist()
            [0.0, 0.0, 0.0, 0.0]

            The third position is padding. ``numeric_value_mask`` is ``True`` there — and it must still
            produce zeros:

            >>> feats[0, 2].tolist()
            [0.0, 0.0, 0.0, 0.0]

            With a reference range installed, the same value takes the range branch instead, mapping the
            lower limit to 0 and the upper to 1:

            >>> model.ref_low[1], model.ref_high[1] = 0.0, 2000.0
            >>> round(model.numeric_features(batch)[0, 0, 0].item(), 3)
            0.5
            >>> model.numeric_features(batch)[0, 0, 1:].tolist()
            [1.0, 0.0, 1.0]

            And the clip bounds are the paper's:

            >>> batch.numeric_value[0, 0] = -10_000.0
            >>> round(model.numeric_features(batch)[0, 0, 0].item(), 3)
            -2.0
        """
        present = batch.numeric_value_mask & padding_mask(batch)
        value = (
            self.raw_numeric_values(batch)
            if self.hparams.denormalize_numeric_values
            else (torch.nan_to_num(batch.numeric_value))
        )

        low, high = self.ref_low[batch.code], self.ref_high[batch.code]
        has_range = torch.isfinite(low) & torch.isfinite(high) & (high > low)
        scaled = torch.nan_to_num((value - low) / (high - low)).clamp(*REF_RANGE_CLIP)
        fallback = (torch.log1p(value.abs()) / LOG_FALLBACK_DIVISOR).clamp(*LOG_FALLBACK_CLIP)

        use_range = present & has_range
        use_fallback = present & ~has_range
        scalar = torch.where(use_range, scaled, fallback) * present
        return torch.stack([scalar, use_range, use_fallback, present], dim=-1).float()

    def raw_numeric_values(self, batch: MEDSTorchBatch) -> Tensor:
        """Undo meds-torch-data's per-code z-scoring, recovering the clinical magnitude.

        Both consumers need the raw value: §2.4.2's scalar is defined on it, and §2.4.4's quantile bins
        were fitted on it. See :func:`build_value_statistics`.
        """
        return (
            torch.nan_to_num(batch.numeric_value) * self.value_std[batch.code] + self.value_mean[batch.code]
        )

    def composite_targets(self, batch: MEDSTorchBatch) -> Tensor:
        """The next-event labels of §2.4.4: concept ID x quantile bin, ``[B, L]``.

        A code with fitted quantiles contributes ``value_bins`` classes and the value picks one; a code
        without them contributes a single class, so the target degrades to the bare code. Padding maps to
        0, which is PAD's own block and is what ``compute_loss`` ignores.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> model = Model(vocab_size=4, d_text=8, value_bins=3, composite_size=10,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 1,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})

            Install a layout by hand: code 1 has two cut points (three bins) starting at slot 1, code 2
            has none and sits at slot 4.

            >>> model.composite_offset.copy_(torch.tensor([0, 1, 4, 5]))
            tensor([0, 1, 4, 5])
            >>> model.value_boundaries[1] = torch.tensor([10.0, 20.0])
            >>> batch = MEDSTorchBatch(
            ...     code=torch.tensor([[1, 1, 1, 2, 0]]),
            ...     numeric_value=torch.tensor([[5.0, 15.0, 25.0, 0.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, True, True, False, False]]),
            ...     time_delta_days=torch.zeros(1, 5),
            ... )

            Values below, between and above the cuts land in consecutive bins; the valueless code takes
            its single slot; padding is 0.

            >>> model.composite_targets(batch).tolist()
            [[1, 2, 3, 4, 0]]
        """
        codes = batch.code
        targets = self.composite_offset[codes]
        if int(self.hparams.value_bins) < 2:
            return torch.where(padding_mask(batch), targets, torch.zeros_like(targets))

        bounds = self.value_boundaries[codes]
        binned = torch.isfinite(bounds).all(dim=-1) & batch.numeric_value_mask & padding_mask(batch)
        values = self.raw_numeric_values(batch).unsqueeze(-1)
        offset = (values > torch.nan_to_num(bounds, nan=float("inf"))).sum(dim=-1)
        targets = targets + torch.where(binned, offset, torch.zeros_like(offset))
        return torch.where(padding_mask(batch), targets, torch.zeros_like(targets))

    def event_representations(self, batch: MEDSTorchBatch) -> Tensor:
        """Fuse the frozen text representation with the numeric pathway by FiLM (§2.4.2).

        ``h = (1 + gamma) * h_text + beta``, with the pathway gated off — ``gamma = beta = 0`` — for
        events with no numeric content, so a non-numeric event is exactly its projected text
        representation.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> _ = torch.manual_seed(0)
            >>> model = Model(vocab_size=4, d_text=8, denormalize_numeric_values=False,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 1,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})
            >>> model.text_table.normal_()  # doctest: +ELLIPSIS
            tensor(...)

            The gamma/beta heads start at zero, so the fusion is initially the identity — the paper's stated
            initialization:

            >>> batch = MEDSTorchBatch(
            ...     code=torch.tensor([[1, 2]]),
            ...     numeric_value=torch.tensor([[5.0, 0.0]]),
            ...     numeric_value_mask=torch.tensor([[True, False]]),
            ...     time_delta_days=torch.zeros(1, 2),
            ... )
            >>> text_only = model.text_proj(model.text_table[batch.code])
            >>> bool(torch.allclose(model.event_representations(batch), text_only))
            True

            Once the heads are non-zero, the event *with* a value is modulated and the one without is
            still untouched — which is the gating, not the zero-initialization:

            >>> _ = torch.nn.init.normal_(model.gamma_head.bias)
            >>> _ = torch.nn.init.normal_(model.beta_head.bias)
            >>> fused = model.event_representations(batch)
            >>> bool(torch.allclose(fused[:, 1], text_only[:, 1]))
            True
            >>> bool(torch.allclose(fused[:, 0], text_only[:, 0]))
            False
        """
        h_text = self.text_proj(self.text_table[batch.code])
        gate = (batch.numeric_value_mask & padding_mask(batch)).unsqueeze(-1)
        # `numeric_features` returns float32 by construction; cast to whatever the pathway's parameters
        # are, or a true-half precision (`bf16-true`, `16-true`) fails on the first Linear.
        features = self.numeric_features(batch).to(self.gamma_head.weight.dtype)
        z = self.numeric_mlp(features)
        gamma = self.gamma_head(z) * gate
        beta = self.beta_head(z) * gate
        return (1.0 + gamma) * h_text + beta

    def backbone_states(self, batch: MEDSTorchBatch) -> Tensor:
        """Run the temporal backbone over the fused event representations; return ``[B, L, hidden]``.

        Padding needs no attention mask term: sequences are right-padded and attention is causal, so a
        real query cannot reach a pad key. :func:`require_right_padding` is the check that keeps that
        true, and dropping the mask is what lets the global layers use a FlashAttention kernel at the
        paper's 16,384-event context.
        """
        require_right_padding(padding_mask(batch))
        h = self.event_representations(batch)
        age_days = torch.cumsum(torch.nan_to_num(batch.time_delta_days), dim=1)
        cos, sin = self.rope(age_days * self.hparams.rope_time_scale)
        impl = self.attention_impl
        for depth, layer in enumerate(self.layers):
            # Supplementary Table S4: "starting with a global attention layer, followed by a local
            # attention layer (attention width of 128), and repeating this pattern throughout."
            window = None if depth % 2 == 0 else self.window
            h = layer(h, cos, sin, window, impl)
        return self.norm_f(h)

    # ------------------------------------------------------------------------------------------------
    # The two hooks the DAG calls.
    # ------------------------------------------------------------------------------------------------

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        """Next-event prediction: cross-entropy over the next event's code (§2.4.4).

        The paper predicts a *composite* token — "the primary OMOP concept ID with discretized event
        attributes, including numeric values discretized into 10 quantile bins" — over a vocabulary of
        roughly 98,000 classes. Over a ``MEDSTorchBatch`` the available target is the tensorized code, so
        this predicts that; building the composite vocabulary is a preprocessing concern and is recorded as
        a deviation in the implementation report.

        This DAG never calls this with labels: ``supervised_train`` is ``ProbeTrainCommand``, which trains
        :class:`Probe` on frozen embeddings and never instantiates this class. A labelled batch means the
        command registry was rewired, so it raises rather than silently pretraining on task data.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> _ = torch.manual_seed(0)
            >>> model = Model(vocab_size=8, d_text=8,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 2,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})
            >>> batch = MEDSTorchBatch(
            ...     code=torch.tensor([[3, 4, 5, 2], [3, 6, 7, 0]]),
            ...     numeric_value=torch.zeros(2, 4),
            ...     numeric_value_mask=torch.zeros(2, 4, dtype=torch.bool),
            ...     time_delta_days=torch.tensor([[0.0, 1.0, 1.0, 3.0], [0.0, 2.0, 5.0, 0.0]]),
            ... )
            >>> loss, metrics = model.compute_loss(batch)
            >>> sorted(metrics)
            ['next_code_accuracy', 'perplexity']
            >>> bool(torch.isfinite(loss)), loss.shape
            (True, torch.Size([]))

            Padding is excluded from the objective, so the shorter subject's tail cannot be "predicted":

            >>> shorter = MEDSTorchBatch(
            ...     code=torch.tensor([[3, 4, 5, 2], [3, 6, 7, 0]]),
            ...     numeric_value=torch.zeros(2, 4),
            ...     numeric_value_mask=torch.zeros(2, 4, dtype=torch.bool),
            ...     time_delta_days=torch.tensor([[0.0, 1.0, 1.0, 3.0], [0.0, 2.0, 5.0, 0.0]]),
            ... )
            >>> bool(torch.allclose(model.compute_loss(shorter)[0], loss))
            True

            A labelled batch is an error, not a second training mode:

            >>> labelled = MEDSTorchBatch(
            ...     code=torch.tensor([[3, 4], [5, 6]]),
            ...     numeric_value=torch.zeros(2, 2),
            ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 2),
            ...     boolean_value=torch.tensor([True, False]),
            ... )
            >>> model.compute_loss(labelled)
            Traceback (most recent call last):
                ...
            ValueError: PORTER's pretraining objective was handed a labelled batch...
        """
        if batch.has_labels:
            raise ValueError(
                "PORTER's pretraining objective was handed a labelled batch, but in the probe DAG only "
                "`pretrain` trains this model and it never sees labels — `supervised_train` fits "
                "`porter.model.Probe` on frozen embeddings instead. A labelled batch here means "
                "`commands.py` registers a different supervised_train implementation."
            )
        _check_ar_target(batch)
        hidden = self.backbone_states(batch)
        logits = self.head(hidden[:, :-1])
        target = self.composite_targets(batch)[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD_INDEX
        )
        with torch.no_grad():
            scored = target != PAD_INDEX
            accuracy = (logits.argmax(-1)[scored] == target[scored]).float().mean()
        return loss, {"next_code_accuracy": accuracy, "perplexity": loss.detach().exp()}

    def encode(self, batch: MEDSTorchBatch) -> Tensor:
        """The frozen representation ``infer`` materialises: ``[B, hidden]`` per prediction timepoint.

        §2.7.1: "the backbone was frozen and hidden states were extracted at prediction time". The task
        datamodule samples ``TO_END``, so the last non-padding position *is* the prediction time. The paper
        does not say which layer or whether any pooling is applied; this takes the final layer's state at
        that position (``pooling: last``), with a masked mean available as ``pooling: mean``.

        The pretraining head is not involved — it is discarded before representation extraction (§2.4.4).

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> _ = torch.manual_seed(0)
            >>> model = Model(vocab_size=8, d_text=8,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 2,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})
            >>> batch = MEDSTorchBatch(
            ...     code=torch.tensor([[3, 4, 5, 2], [3, 6, 0, 0]]),
            ...     numeric_value=torch.zeros(2, 4),
            ...     numeric_value_mask=torch.zeros(2, 4, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 4),
            ... )
            >>> model.encode(batch).shape
            torch.Size([2, 8])

            The vector is the state at each subject's own last real event — position 3 for the first
            subject, position 1 for the second — never at padding:

            >>> states = model.backbone_states(batch)
            >>> pooled = model.encode(batch)
            >>> bool(torch.allclose(pooled[0], states[0, 3])), bool(torch.allclose(pooled[1], states[1, 1]))
            (True, True)
        """
        hidden = self.backbone_states(batch)
        keep = padding_mask(batch)
        if self.hparams.pooling == "mean":
            return masked_mean(hidden, keep)
        last = keep.sum(dim=1).sub(1).clamp(min=0)
        return hidden.gather(1, last[:, None, None].expand(-1, 1, hidden.shape[-1])).squeeze(1)

    def configure_optimizers(self):
        """AdamW with linear warmup then cosine decay to a floor (§2.4.4, Supplementary Table S4).

        The optimizer itself comes from ``configs/optimizer/porter.yaml`` (peak LR 5e-4, weight decay 0.05,
        betas 0.9/0.95). The schedule is built here rather than as a ``scheduler`` config because it needs
        ``trainer.estimated_stepping_batches``, which no config can express; supplying ``model.scheduler``
        overrides it. Warmup is clamped to a tenth of the run: the paper's 150 steps assumes a 894 M-event
        pretraining run, and would otherwise consume most of a short one.

        Examples:
            Outside a ``Trainer`` there is no step count to schedule against, so the optimizer is returned
            bare — which is what makes this safe to call from a doctest:

            >>> model = Model(vocab_size=4, d_text=8,
            ...               backbone={"hidden_size": 8, "num_hidden_layers": 1,
            ...                         "num_attention_heads": 2, "intermediate_size": 8})
            >>> opt = model.configure_optimizers()
            >>> type(opt).__name__
            'AdamW'

            With no optimizer factory supplied, the paper's own settings are used rather than torch's:

            >>> group = opt.param_groups[0]
            >>> group["lr"], group["weight_decay"], group["betas"]
            (0.0005, 0.05, (0.9, 0.95))
        """
        if self._optimizer_factory is not None:
            optimizer = self._optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=PAPER_LR, weight_decay=PAPER_WEIGHT_DECAY, betas=PAPER_BETAS
            )
        if self._scheduler_factory is not None:
            scheduler = self._scheduler_factory(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        try:
            total = max(1, int(self.trainer.estimated_stepping_batches))
        except RuntimeError:  # not attached to a Trainer (doctests, manual use)
            return optimizer

        warmup = min(int(self.hparams.warmup_steps), max(1, total // 10))
        floor = float(self.hparams.min_lr_ratio)

        def lr_scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class Probe(L.LightningModule):
    """L2-regularised logistic regression on frozen PORTER representations (§2.7.1).

    "Linear probes were trained using L2-regularized logistic regression implemented in Scikit-learn, with
    features standardized to zero mean and unit variance using statistics computed on the training subset
    of each task's split."

    ``supervised_train`` builds this through ``configs/model/probe.yaml``, injecting ``input_dim`` from the
    width of the materialized embeddings; batches are plain ``(features, labels)`` tensors. ``predict``
    reloads it from its checkpoint, so the standardisation statistics are buffers rather than state
    recomputed at scoring time.

    The paper fits this with scikit-learn's LBFGS and selects the inverse regularisation strength ``C`` from
    ``{1, 1e-1, 1e-2, 1e-3, 1e-4}`` on the validation set. This DAG's ``supervised_train`` is a Lightning
    ``fit`` loop, so the same convex objective is minimised by mini-batch AdamW instead, and ``C`` is a
    config key rather than an inner search. The correspondence is ``weight_decay ≈ 1 / (C · n_train)``; the
    paper's grid is reachable with
    ``meds-model supervised_train --multirun model.l2=1,0.1,0.01,0.001,0.0001``.

    Args:
        input_dim: injected by ``ProbeTrainCommand`` from the embedding width.
        lr: AdamW learning rate for the probe.
        l2: weight decay on the coefficient vector. Applied to the weight only — a logistic-regression
            penalty does not touch the intercept.
        standardize: fit and apply the training-split ``StandardScaler``.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> probe = Probe(input_dim=5)
        >>> probe(torch.randn(3, 5)).shape
        torch.Size([3])

        ``forward`` returns logits, not probabilities: ``predict`` applies the sigmoid itself.

        >>> probe.linear.bias.data.fill_(10.0)
        tensor([10.])
        >>> float(probe(torch.zeros(1, 5)).detach()[0])
        10.0

        The penalty reaches the coefficients and not the intercept:

        >>> groups = probe.configure_optimizers().param_groups
        >>> [(g["params"][0].shape[-1], g["weight_decay"]) for g in groups]
        [(5, 0.0001), (1, 0.0)]

        Standardisation starts as the identity and is frozen into buffers, so it survives the checkpoint
        round trip that ``predict`` depends on:

        >>> sorted(k for k in probe.state_dict() if not k.startswith("linear"))
        ['feat_mean', 'feat_std']
    """

    def __init__(self, input_dim: int, lr: float = 1e-2, l2: float = 1e-4, standardize: bool = True) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(int(input_dim), 1)
        self.register_buffer("feat_mean", torch.zeros(int(input_dim)))
        self.register_buffer("feat_std", torch.ones(int(input_dim)))

    def on_train_start(self) -> None:
        """Fit the scaler on the training split only, exactly as §2.7.1 specifies."""
        if not self.hparams.standardize:
            return
        loader = self.trainer.train_dataloader
        if loader is None:
            raise RuntimeError(
                "The probe's StandardScaler is fitted from the training dataloader, which is not "
                "available. Set model.standardize=false to train on unscaled embeddings."
            )
        features = torch.cat([x for x, _ in loader])
        self.feat_mean.copy_(features.mean(dim=0))
        self.feat_std.copy_(features.std(dim=0, unbiased=False).clamp(min=1e-6))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``[B]`` logits for a ``[B, input_dim]`` batch of frozen embeddings."""
        return self.linear((x - self.feat_mean) / self.feat_std).squeeze(-1)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        loss = F.binary_cross_entropy_with_logits(self(x), y.float())
        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        # `callbacks/default.yaml` monitors `val/loss`; without this log, checkpoint selection silently
        # degrades to the final weights rather than the best ones.
        x, y = batch
        loss = F.binary_cross_entropy_with_logits(self(x), y.float())
        self.log("val/loss", loss, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": [self.linear.weight], "weight_decay": float(self.hparams.l2)},
                {"params": [self.linear.bias], "weight_decay": 0.0},
            ],
            lr=float(self.hparams.lr),
        )
