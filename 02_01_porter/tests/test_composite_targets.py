"""PORTER's pretraining label is a composite token, and only the label (arXiv:2606.24102 §2.4.4).

"Each composite token combined the primary OMOP concept ID with discretized event attributes, including
numeric values discretized into 10 quantile bins where applicable. In PORTER, these composite tokens
served **only as output labels** for autoregressive pretraining, while model inputs were language-grounded
event representations."

That last sentence is the whole design, and it is what these tests hold in place. The obvious way to get
composite tokens — MEDS-transforms' `bin_numeric_values` — rewrites `HR` into `HR//value_[80,90)`, which
moves the magnitude into the *input* description and drops `numeric_value`. That is PORTER-NumText
(§2.6.2), which the paper reports as strictly worse. So the binning happens in `Model.composite_targets`,
against boundaries fitted by `pipelines/quantiles.yaml`, and the input pathway never sees it.
"""

import polars as pl
import pytest
import torch
from meds_model_base.testing import build_workspace

from porter.model import Model, build_composite_index

TINY = {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 16}


@pytest.fixture(scope="module")
def quantile_workspace(meds_root, tmp_path_factory):
    """A workspace built with the shipped default pipeline, so the quantiles are fitted."""
    return build_workspace(meds_root, tmp_path_factory.mktemp("quantiles") / "data")


@pytest.fixture(scope="module")
def code_metadata(quantile_workspace):
    return str(quantile_workspace / "patients" / "metadata" / "codes.parquet")


def test_the_pipeline_fits_quantiles(code_metadata):
    """Without this column everything below degrades silently, so assert it is really there."""
    codes = pl.read_parquet(code_metadata)
    assert "values/quantiles" in codes.columns, (
        "preprocess_data did not fit per-code quantiles. The default `pipeline:` in "
        "configs/preprocess_data.yaml is what produces them."
    )
    valued = codes.filter(pl.col("values/quantiles").is_not_null())
    assert len(valued), "the fixture has no code with numeric values, so this module proves nothing"


def test_quantiles_are_fitted_on_the_training_split_only(code_metadata, meds_root):
    """The label definition must not have seen tuning or held-out values.

    `aggregate_code_metadata` reduces over every shard unless told otherwise, so the decile cut points
    that *define the pretraining target* would otherwise be computed partly from data the model is
    scored on. `pipelines/quantiles.yaml` sets `train_only: true`; this checks it took effect, by
    recomputing the deciles by hand from the train shards and from all shards. Passing the first and
    failing the second is what makes this test non-vacuous — MEDS-transforms silently falls back to all
    shards when it finds no dedicated train split.
    """
    codes = pl.read_parquet(code_metadata)

    def deciles(pattern: str, code: str) -> list[float]:
        shards = sorted((meds_root / "data").glob(pattern))
        values = pl.concat([pl.read_parquet(shard) for shard in shards]).filter(
            (pl.col("code") == code) & pl.col("numeric_value").is_not_null()
        )["numeric_value"]
        return [round(float(values.quantile(q / 10)), 3) for q in range(1, 10)]

    valued = codes.filter(pl.col("values/quantiles").is_not_null())
    assert len(valued), "no code carries values, so this test would prove nothing"

    checked_a_difference = False
    for code, quantiles in zip(valued["code"].to_list(), valued["values/quantiles"].to_list(), strict=True):
        fitted = [round(v, 3) for _, v in sorted(quantiles.items())]
        from_train = deciles("train/*.parquet", code)
        from_all = deciles("*/*.parquet", code)
        assert fitted == from_train, (
            f"{code}: the shipped boundaries are not the training-split deciles.\n"
            f"  fitted     {fitted}\n  train-only {from_train}\n  all splits {from_all}"
        )
        checked_a_difference |= from_train != from_all

    assert checked_a_difference, (
        "the train-only and all-splits deciles are identical for every code, so this fixture cannot "
        "distinguish them and the assertion above is vacuous"
    )


def test_codes_with_values_get_a_block_and_others_get_one_slot(code_metadata):
    """ "Where applicable": only concepts that carry numbers pay for bins."""
    codes = pl.read_parquet(code_metadata)
    vocab_size = int(codes["code/vocab_index"].max()) + 1
    offsets, boundaries, size = build_composite_index(code_metadata, vocab_size, value_bins=10)

    n_valued = len(codes.filter(pl.col("values/quantiles").is_not_null()))
    # Each valued code costs 10 slots instead of 1; everything else costs 1.
    assert size == vocab_size + n_valued * 9
    assert size > vocab_size, "no code was given bins, so the composite target is just the bare code"
    assert torch.isfinite(boundaries).all(dim=-1).sum() == n_valued
    assert offsets[0] == 0, "PAD must own slot 0, which is the ignore_index"
    assert (offsets.diff() > 0).all(), "blocks must be disjoint and ordered"


def test_the_same_code_with_different_values_gets_different_labels(code_metadata):
    """The point of the whole exercise: magnitude reaches the objective."""
    codes = pl.read_parquet(code_metadata)
    vocab_size = int(codes["code/vocab_index"].max()) + 1
    valued = codes.filter(pl.col("values/quantiles").is_not_null())
    index = int(valued["code/vocab_index"][0])
    cuts = sorted(v for v in valued["values/quantiles"][0].values() if v is not None)

    model = Model(
        vocab_size=vocab_size, code_metadata_fp=code_metadata, text_encoder="random", d_text=8, backbone=TINY
    )
    # Values straddling the fitted cut points, expressed as the z-scores the batch actually carries.
    raw = torch.tensor([cuts[0] - 1e3, cuts[len(cuts) // 2], cuts[-1] + 1e3])
    z = (raw - model.value_mean[index]) / model.value_std[index]

    from meds_torchdata import MEDSTorchBatch

    batch = MEDSTorchBatch(
        code=torch.full((1, 3), index),
        numeric_value=z.unsqueeze(0),
        numeric_value_mask=torch.ones(1, 3, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 3),
    )
    targets = model.composite_targets(batch)[0]
    assert targets[0] < targets[1] < targets[2], f"bins are not ordered by magnitude: {targets.tolist()}"
    assert len(set(targets.tolist())) == 3, "three very different values collapsed to one label"


def test_inputs_are_untouched_by_quantization(code_metadata):
    """The input representation must not depend on the bin — that would be the NumText ablation."""
    codes = pl.read_parquet(code_metadata)
    vocab_size = int(codes["code/vocab_index"].max()) + 1
    binned = Model(
        vocab_size=vocab_size,
        code_metadata_fp=code_metadata,
        text_encoder="random",
        d_text=8,
        backbone=TINY,
        value_bins=10,
    )
    unbinned = Model(
        vocab_size=vocab_size,
        code_metadata_fp=code_metadata,
        text_encoder="random",
        d_text=8,
        backbone=TINY,
        value_bins=1,
    )
    # The text table is the entire input vocabulary, and it is keyed on the concept alone.
    assert torch.equal(binned.text_table, unbinned.text_table)
    assert binned.text_table.shape[0] == vocab_size, "the input vocabulary must stay concept-level"
    assert binned.head.out_features > unbinned.head.out_features, "only the output side should grow"


def test_composite_layout_survives_the_checkpoint(code_metadata, tmp_path):
    """`infer`/`predict` rebuild with no kwargs, so the layout has to ride in the checkpoint."""
    import lightning.pytorch as L

    codes = pl.read_parquet(code_metadata)
    vocab_size = int(codes["code/vocab_index"].max()) + 1
    model = Model(
        vocab_size=vocab_size, code_metadata_fp=code_metadata, text_encoder="random", d_text=8, backbone=TINY
    )

    trainer = L.Trainer(logger=False, accelerator="cpu", devices=1)
    trainer.strategy.connect(model)
    ckpt = tmp_path / "checkpoint"
    trainer.save_checkpoint(ckpt)
    reloaded = Model.load_from_checkpoint(ckpt, map_location="cpu")

    assert reloaded.head.out_features == model.head.out_features
    assert torch.equal(reloaded.composite_offset, model.composite_offset)
    assert torch.equal(torch.nan_to_num(reloaded.value_boundaries), torch.nan_to_num(model.value_boundaries))


def test_it_degrades_to_bare_codes_without_quantiles(tmp_path):
    """`pipeline: null` is a supported (and recorded) choice, not a crash."""
    fp = tmp_path / "codes.parquet"
    pl.DataFrame({"code": ["A", "B"], "code/vocab_index": [1, 2]}).write_parquet(fp)
    offsets, _, size = build_composite_index(str(fp), vocab_size=3, value_bins=10)
    assert size == 3
    assert offsets.tolist() == [0, 1, 2]
