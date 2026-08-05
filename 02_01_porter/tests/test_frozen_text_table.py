"""The cached event-description table is built once, by `pretrain`, and then rides in the checkpoint.

PORTER encodes each event description once with a frozen text encoder and caches the result
(arXiv:2606.24102 §2.4.1). Here that cache is a persistent buffer and `code_metadata_fp` is deliberately
**not** a saved hyperparameter, so `infer` and `predict` — which rebuild the model with
`load_from_checkpoint` and no arguments — restore the table from the checkpoint instead of re-encoding.

That is a load-bearing property, not an optimisation: it is what makes a published artifact scorable on a
host with no network and no HuggingFace cache, and what keeps the frozen encoder from being loaded three
times per run. These tests fail if either half regresses.
"""

import sys

import polars as pl
import pytest
import torch

from porter.model import Model, build_text_table

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TINY_BACKBONE = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 16,
}


class _BlockImport:
    """A `sys.meta_path` finder that makes importing `name` raise, to prove it is never needed."""

    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(f"{self.name}."):
            raise ImportError(f"{fullname} must not be imported here")
        return None


@pytest.fixture
def code_metadata(tmp_path):
    """A minimal tensorized-cohort `codes.parquet`: codes, descriptions, vocabulary indices."""
    fp = tmp_path / "codes.parquet"
    pl.DataFrame(
        {
            "code": ["LAB//HGB", "COND//ASTHMA", "PROC//XRAY"],
            "description": ["Hemoglobin", "Asthma", None],
            "code/vocab_index": [1, 2, 3],
        }
    ).write_parquet(fp)
    return fp


def test_table_is_frozen_and_not_a_parameter(code_metadata):
    """The encoder is frozen: its output is a buffer, so no optimizer can ever update it."""
    model = Model(
        vocab_size=4,
        code_metadata_fp=str(code_metadata),
        text_encoder="random",
        d_text=8,
        backbone=TINY_BACKBONE,
    )
    assert "text_table" in dict(model.named_buffers())
    assert "text_table" not in dict(model.named_parameters())
    assert not model.text_table.requires_grad
    # Row 0 is PAD and must contribute nothing; the three described codes must all be populated.
    assert model.text_table[0].abs().sum() == 0
    assert (model.text_table[1:].abs().sum(dim=-1) > 0).all()


def test_checkpoint_carries_the_table_and_reload_needs_no_encoder(code_metadata, tmp_path):
    """`load_from_checkpoint` restores the cache without importing the text encoder at all."""
    import lightning.pytorch as L

    model = Model(
        vocab_size=4,
        code_metadata_fp=str(code_metadata),
        text_encoder="random",
        d_text=8,
        backbone=TINY_BACKBONE,
    )
    original = model.text_table.clone()
    assert original.abs().sum() > 0

    # The path is not a hyperparameter, so a reload cannot go looking for the workspace or the encoder.
    assert "code_metadata_fp" not in model.hparams

    trainer = L.Trainer(logger=False, accelerator="cpu", devices=1)
    trainer.strategy.connect(model)
    ckpt = tmp_path / "checkpoint"
    trainer.save_checkpoint(ckpt)

    blocker = _BlockImport("sentence_transformers")
    sys.meta_path.insert(0, blocker)
    sys.modules.pop("sentence_transformers", None)
    try:
        reloaded = Model.load_from_checkpoint(ckpt, map_location="cpu")
    finally:
        sys.meta_path.remove(blocker)

    assert torch.equal(reloaded.text_table, original), "the cached table did not survive the checkpoint"
    assert torch.equal(reloaded.value_mean, model.value_mean)
    assert torch.equal(reloaded.value_std, model.value_std)


def test_missing_vocabulary_rows_stay_zero(tmp_path):
    """A vocabulary index the metadata does not describe contributes nothing, rather than noise."""
    fp = tmp_path / "codes.parquet"
    pl.DataFrame(
        {"code": ["LAB//HGB"], "description": ["Hemoglobin"], "code/vocab_index": [1]}
    ).write_parquet(fp)
    table = build_text_table(str(fp), vocab_size=5, text_encoder="random", d_text=8)
    assert table[1].abs().sum() > 0
    assert table[[0, 2, 3, 4]].abs().sum() == 0


def test_encoder_width_mismatch_is_a_named_error(code_metadata, monkeypatch):
    """A `d_text` that disagrees with the encoder must fail loudly, not reshape or truncate silently.

    A stub encoder stands in for a real checkpoint: the failure is about the declared width, and reaching
    HuggingFace to demonstrate that would make this a network test.
    """
    import types

    import numpy as np

    class _WrongWidthEncoder:
        def __init__(self, name):
            self.name = name

        def encode(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _WrongWidthEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(ValueError, match=r"384-dimensional.*d_text=768"):
        build_text_table(str(code_metadata), vocab_size=4, text_encoder="stub/encoder", d_text=768)
