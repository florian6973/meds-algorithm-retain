"""Device portability of the real DAG.

`device` defaults to `auto`, so on this host the ordinary local gate already runs the DAG on CUDA and the
manifests carry real GPU byte counts. That makes the *untested* direction the opposite one: a CPU-only
node, and a checkpoint trained on GPU but scored on CPU. Both are ordinary cluster shapes.

The failure mode these tests exist for is a device-conditional code path that a single-device run cannot
see -- a tensor left on the host, a dtype promoted differently on CUDA, a `.to(device)` that drops a type,
or a parameter that is pinned at construction and quietly revived by a device move.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from retain.commands import COMMANDS  # noqa: E402
from retain.config import RetainConfig  # noqa: E402
from retain.model import Retain  # noqa: E402

from .fixtures import SMOKE_HYPERPARAMETERS, write_meds_dataset, write_task_labels  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this host")

PROBABILITY = "predicted_boolean_probability"


def _run_dag(root, *, train_device, predict_device, **overrides):
    """Run the real three commands, training and scoring on the named devices."""
    meds_dir = write_meds_dataset(root / "meds")
    labels_dir = write_task_labels(meds_dir, root / "labels")
    data_dir, model_dir, predictions_dir = root / "data", root / "model", root / "predictions"
    COMMANDS["preprocess_data"]()(
        {"external_meds_dir": meds_dir, "output_data_dir": data_dir, "do_overwrite": True}
    )
    COMMANDS["supervised_train"]()(
        {
            "input_data_dir": data_dir,
            "external_labels_dir": labels_dir,
            "output_supervised_model_dir": model_dir,
            "do_overwrite": True,
            **SMOKE_HYPERPARAMETERS,
            **overrides,
            "device": train_device,
        }
    )
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": model_dir,
            "external_labels_dir": labels_dir,
            "output_predictions_dir": predictions_dir,
            "do_overwrite": True,
            "device": predict_device,
        }
    )
    return model_dir, pl.read_parquet(predictions_dir / "predictions.parquet").sort(
        "subject_id", "prediction_time"
    )


def test_the_whole_dag_runs_on_cpu(tmp_path):
    """A CPU-only node must be able to run every command, not just the model's forward pass."""
    _, predictions = _run_dag(tmp_path, train_device="cpu", predict_device="cpu")
    assert predictions.height > 0
    assert predictions.get_column(PROBABILITY).is_between(0.0, 1.0).all()
    assert predictions.get_column(PROBABILITY).null_count() == 0


@requires_cuda
def test_the_whole_dag_runs_on_cuda(tmp_path):
    _, predictions = _run_dag(tmp_path, train_device="cuda", predict_device="cuda")
    assert predictions.height > 0
    assert predictions.get_column(PROBABILITY).is_between(0.0, 1.0).all()
    assert predictions.get_column(PROBABILITY).null_count() == 0


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=requires_cuda)],
)
def test_two_runs_with_the_same_seed_are_bit_identical(tmp_path, device):
    """The control that every cross-device claim rests on: fix the device, fix the seed, run twice.

    This is the regression test for a seed-ordering bug -- seeding *after* the model is constructed, so
    weight initialisation draws from an unseeded global generator. That failure is invisible in a
    CPU-versus-CUDA comparison, where it would simply look like device noise, and it is only visible here.
    It also covers a stray generator entering through the length-sorted batching, the per-epoch batch
    reshuffle, or the fallback validation split.

    Compared exactly, not within a tolerance: on a fixed device and a fixed seed there is no source of
    legitimate variation to allow for.
    """
    _, first = _run_dag(tmp_path / "first", train_device=device, predict_device=device)
    _, second = _run_dag(tmp_path / "second", train_device=device, predict_device=device)
    assert first.equals(second)


def test_different_seeds_produce_different_models(tmp_path):
    """The other half of the control: prove the seed is actually wired in.

    Without this, a pipeline that ignored `seed` entirely would still pass the test above.
    """
    _, first = _run_dag(tmp_path / "seed-0", train_device="cpu", predict_device="cpu", seed=0)
    _, second = _run_dag(tmp_path / "seed-1", train_device="cpu", predict_device="cpu", seed=1)
    assert not np.allclose(
        first.get_column(PROBABILITY).to_numpy(), second.get_column(PROBABILITY).to_numpy()
    )


def test_weight_initialisation_does_not_depend_on_the_device():
    """`reset_parameters` runs before the move, so both devices start from the same weights."""
    config = RetainConfig(embed_size=8, alpha_hidden_dim_size=6, beta_hidden_dim_size=6)
    torch.manual_seed(0)
    on_cpu = Retain(12, config)
    torch.manual_seed(0)
    elsewhere = Retain(12, config).to("cuda" if torch.cuda.is_available() else "cpu")
    for left, right in zip(on_cpu.parameters(), elsewhere.parameters()):
        assert torch.equal(left.detach(), right.detach().cpu())


@requires_cuda
def test_cpu_and_cuda_agree_on_the_same_seed_without_dropout(tmp_path):
    """Same seed, same data, different device: the fit and the scores must match numerically.

    Dropout is disabled here on purpose. CUDA draws its masks from its own generator, so a seeded run
    on CPU and a seeded run on CUDA see *different* dropout masks and therefore train to different
    parameters -- documented PyTorch behaviour, not a property of this port, and confirmed separately
    (weight initialisation above is identical; only the masks differ). With dropout off, the two devices
    compute the same arithmetic, so any remaining disagreement would be a real device-conditional bug:
    a tensor left on the host, a dtype promoted differently, a lost tensor type.

    The consequence for the benchmark is recorded in IMPLEMENTATION_REPORT.md: a RETAIN run is
    reproducible on a fixed device, and model comparisons must hold the device fixed.
    """
    no_dropout = {"keep_prob_emb": 1.0, "keep_prob_context": 1.0}
    _, on_cpu = _run_dag(tmp_path / "cpu", train_device="cpu", predict_device="cpu", **no_dropout)
    _, on_cuda = _run_dag(tmp_path / "cuda", train_device="cuda", predict_device="cuda", **no_dropout)
    assert on_cpu.select("subject_id", "prediction_time").equals(
        on_cuda.select("subject_id", "prediction_time")
    )
    np.testing.assert_allclose(
        on_cpu.get_column(PROBABILITY).to_numpy(),
        on_cuda.get_column(PROBABILITY).to_numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


@requires_cuda
def test_a_gpu_checkpoint_scores_on_cpu(tmp_path):
    """Train on the GPU node, score on a CPU node: the checkpoint must survive the move."""
    _, mixed = _run_dag(tmp_path / "mixed", train_device="cuda", predict_device="cpu")
    _, pure = _run_dag(tmp_path / "pure", train_device="cuda", predict_device="cuda")
    np.testing.assert_allclose(
        mixed.get_column(PROBABILITY).to_numpy(),
        pure.get_column(PROBABILITY).to_numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


@requires_cuda
def test_the_pinned_recurrent_bias_survives_a_device_move(tmp_path):
    """`bias_hh` is zeroed at construction and excluded from the optimiser.

    Neither property is enforced by anything after `__init__`, so a `.cuda()` round trip or a state-dict
    reload is exactly where it could quietly come back. Checked on a freshly moved model and on a
    checkpoint that has actually been trained on the GPU.
    """
    model = Retain(12, RetainConfig(embed_size=8, alpha_hidden_dim_size=6, beta_hidden_dim_size=6)).cuda()
    for gru in (model.gru_alpha, model.gru_beta):
        assert not gru.bias_hh_l0.requires_grad
        assert torch.count_nonzero(gru.bias_hh_l0) == 0
    assert all(parameter.is_cuda for parameter in model.trainable_parameters())

    model_dir, _ = _run_dag(tmp_path, train_device="cuda", predict_device="cuda")
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)["state_dict"]
    for name in ("gru_alpha.bias_hh_l0", "gru_beta.bias_hh_l0"):
        assert torch.count_nonzero(state[name]) == 0, f"{name} became non-zero during GPU training"
