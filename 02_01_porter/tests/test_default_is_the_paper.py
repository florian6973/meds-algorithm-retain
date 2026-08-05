"""`configs/model/default.yaml` must be PORTER at the published scale.

`tests/conftest.py` sets the `PORTER_*` size variables for the whole session so the suite can train on
CPU, which means every other test composes the *small* branch of `default.yaml` and none of them would
notice if its published literals were wrong. This module clears those variables and checks the real
default — otherwise the one config that decides what a cluster run trains is the one nothing tests.
"""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from meds_model_base.lightning import register_structured_configs

import porter
from porter.model import PAPER_BACKBONE, SMALL_BACKBONE, backbone_parameter_count

CONFIG_DIR = str(Path(porter.__file__).parent / "configs")

#: arXiv:2606.24102 §2.4.3 / Supplementary Table S6: "The transformer backbone (115.63 M) ...".
PUBLISHED_BACKBONE_PARAMETERS = 115_649_280

#: §2.4.3: "The maximum effective context window was therefore 16,384 clinical events per patient."
PUBLISHED_CONTEXT = 16_384


@pytest.fixture
def unscaled(monkeypatch):
    """Compose configs as a real run would, with the test suite's size overrides removed."""
    for key in list(os.environ):
        if key.startswith("PORTER_"):
            monkeypatch.delenv(key, raising=False)
    register_structured_configs()

    def _compose(config_name: str, *overrides: str):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            return compose(config_name=config_name, overrides=["input_data_dir=/tmp/x", *overrides])

    return _compose


@pytest.mark.parametrize("command", ["pretrain", "infer", "predict", "supervised_train"])
def test_every_command_composes_the_published_context(unscaled, command):
    """All four command configs carry the paper's context window, not a convenient smaller one."""
    assert unscaled(command).max_seq_len == PUBLISHED_CONTEXT


def test_default_model_is_the_published_geometry(unscaled):
    """The shipped default must be the 115.6 M backbone, to the parameter."""
    backbone = dict(unscaled("pretrain").model.backbone)
    assert backbone_parameter_count(backbone) == PUBLISHED_BACKBONE_PARAMETERS
    assert backbone == dict(PAPER_BACKBONE)


def test_small_is_selectable_and_is_not_the_default(unscaled):
    """`model=small` exists for laptops and is emphatically not what a bare run gets."""
    small = dict(unscaled("pretrain", "model=small").model.backbone)
    assert small == dict(SMALL_BACKBONE)
    assert backbone_parameter_count(small) < PUBLISHED_BACKBONE_PARAMETERS / 100


def test_the_test_suite_really_does_scale_the_model_down():
    """The other direction: conftest's variables must actually take effect, or the tiers train 115.6 M."""
    register_structured_configs()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="pretrain", overrides=["input_data_dir=/tmp/x"])
    assert dict(cfg.model.backbone) == dict(SMALL_BACKBONE)


@pytest.mark.parametrize("command", ["pretrain", "infer", "predict", "supervised_train"])
def test_demographics_reach_the_model_by_default(unscaled, command):
    """Table S3 renders demographics as timeline events, so statics must not be silently dropped."""
    assert str(unscaled(command).datamodule.config.static_inclusion_mode).lower().endswith("prepend")


def test_optimizer_is_the_papers(unscaled):
    """§2.4.4: AdamW, peak LR 5e-4, weight decay 0.05, betas (0.9, 0.95)."""
    optimizer = unscaled("pretrain").optimizer
    assert optimizer.lr == pytest.approx(5e-4)
    assert optimizer.weight_decay == pytest.approx(0.05)
    assert list(optimizer.betas) == [0.9, 0.95]
