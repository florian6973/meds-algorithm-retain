"""Static measurements must reach the model, because PORTER's demographics are timeline events.

arXiv:2606.24102 Table S3 renders sex, birth and death as ordinary events in the sequence. In MEDS those
split in two: birth and death are timestamped and dynamic, while sex, race and similar are *static*
measurements that meds-torch-data keeps out of the dynamic stream unless asked. The shipped datamodules
therefore default to `static_inclusion_mode: PREPEND`.

`conftest.py` overrides that to OMIT for the whole session, because the template's synthetic learnability
fixture has no static measurements at all and PREPEND raises on such a cohort (upstream
MEDS_model_template#16). That override would leave the *shipped* behaviour untested on real data, which
is the point of this module: it uses the real `simple_static_MEDS_dataset_with_task` fixture, which does
have statics, and drives the actual model over it.
"""

import pytest
import torch
from meds_torchdata import MEDSTorchDataConfig
from meds_torchdata.extensions.lightning_datamodule import Datamodule
from meds_torchdata.types import StaticInclusionMode, SubsequenceSamplingStrategy

from porter.model import Model, render_event_description, require_right_padding

TINY = {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 16}


def _batch(patients, mode):
    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(patients),
        max_seq_len=128,
        seq_sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
        static_inclusion_mode=mode,
    )
    datamodule = Datamodule(config=config, batch_size=4, num_workers=0)
    datamodule.setup("fit")
    return config, next(iter(datamodule.train_dataloader()))


def _codes_in(batch, vocabulary):
    return {vocabulary[c] for row in batch.code.tolist() for c in row if c in vocabulary}


@pytest.fixture
def vocabulary(data_dir):
    import polars as pl

    codes = pl.read_parquet(data_dir / "patients" / "metadata" / "codes.parquet")
    return dict(zip(codes["code/vocab_index"].cast(pl.Int64).to_list(), codes["code"].to_list(), strict=True))


def test_prepend_puts_statics_in_the_sequence_and_omit_does_not(data_dir, vocabulary):
    """The two modes must actually differ, or this whole module is vacuous."""
    _, prepended = _batch(data_dir / "patients", StaticInclusionMode.PREPEND)
    _, omitted = _batch(data_dir / "patients", StaticInclusionMode.OMIT)

    gained = _codes_in(prepended, vocabulary) - _codes_in(omitted, vocabulary)
    assert gained, (
        "PREPEND surfaced no code that OMIT did not — either the fixture has no static measurements, "
        "in which case this test proves nothing, or the mode is not being applied."
    )
    # The fixture's statics are eye colour and height; assert on the property, not the specific codes.
    assert all(code in vocabulary.values() for code in gained)


def test_the_model_runs_over_prepended_statics(data_dir):
    """Nothing in `model.py` needs to know about statics — they arrive as ordinary events."""
    config, batch = _batch(data_dir / "patients", StaticInclusionMode.PREPEND)
    require_right_padding(batch.code != 0)  # PREPEND must not break the mask-free attention path

    model = Model(vocab_size=config.vocab_size, d_text=8, backbone=TINY).eval()
    with torch.no_grad():
        pooled = model.encode(batch)
        loss, _ = model.compute_loss(batch)
    assert pooled.shape == (batch.code.shape[0], TINY["hidden_size"])
    assert torch.isfinite(loss)


def test_static_codes_render_as_descriptions(data_dir, vocabulary):
    """A static that reaches the model must also reach the text pathway with a usable description."""
    import polars as pl

    codes = pl.read_parquet(data_dir / "patients" / "metadata" / "codes.parquet")
    rendered = [
        render_event_description(code, description)
        for code, description in zip(codes["code"].to_list(), codes["description"].to_list(), strict=True)
    ]
    assert all(text and text.strip() for text in rendered), "every vocabulary entry needs some text"
    # Nothing should render as the bare `DOMAIN//VALUE` form the encoder was never meant to see.
    assert not any("//" in text for text in rendered), rendered
