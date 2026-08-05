"""The backbone's three attention paths must compute the same function.

`porter.model.causal_attention` has a global path (`is_causal=True`, no mask tensor, FlashAttention
eligible), a dense sliding-window path (an `L x L` band mask) and a block-sparse `flex_attention` path.
The last two exist only because the first cannot express a window; if they ever disagree, a paper-scale
run on a cluster silently computes something different from what the test tiers checked.

The other invariant here is the one that lets the global path drop its mask entirely: sequences are
right-padded and attention is causal, so padding cannot reach a real query. That is asserted directly —
a real position's hidden state must not move when padding is appended.
"""

import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from porter.model import Model, causal_attention, require_right_padding

TINY = {"hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2, "intermediate_size": 16}


def _reference(q, k, v, window):
    """Naive causal (optionally windowed) attention, written for clarity rather than speed."""
    length = q.shape[-2]
    rows = torch.arange(length).unsqueeze(1)
    cols = torch.arange(length).unsqueeze(0)
    allowed = cols <= rows
    if window is not None:
        allowed = allowed & (rows - cols < window)
    scores = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v


@pytest.mark.parametrize("window", [None, 1, 3, 8, 64])
def test_matches_the_naive_reference(window):
    """Every dispatch path equals textbook masked softmax attention."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 32, 8, dtype=torch.float64) for _ in range(3))
    got = causal_attention(q, k, v, window=window, impl="dense")
    assert torch.allclose(got, _reference(q, k, v, window), atol=1e-10)


def test_window_at_least_the_sequence_is_the_global_path():
    """A window wider than the sequence is plain causal attention, and must take the mask-free path."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 12, 8, dtype=torch.float64) for _ in range(3))
    wide = causal_attention(q, k, v, window=64, impl="dense")
    glob = causal_attention(q, k, v, window=None, impl="dense")
    assert torch.allclose(wide, glob, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention's kernel path needs CUDA")
def test_flex_matches_dense():
    """The block-sparse path a cluster run uses must equal the dense path the tiers check.

    Triton JIT-compiles a small C shim against `Python.h`, so a machine without Python development
    headers (or without a C compiler) cannot build the kernel at all. That is an environment fact, not a
    defect in this model, and it is the *only* condition allowed to skip here — a wrong answer or any
    other failure is a real finding. `porter.model.causal_attention` falls back to the dense mask in
    exactly this case, so the model still runs; see the README's cluster notes.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 256, 32, device="cuda") for _ in range(3))
    dense = causal_attention(q, k, v, window=64, impl="dense")
    try:
        flex = causal_attention(q, k, v, window=64, impl="flex")
    except Exception as exc:
        message = str(exc)
        if "cuda_utils.c" in message or "Python.h" in message:
            pytest.skip(f"Triton cannot build its runtime shim on this host: {message[:160]}")
        pytest.fail(f"attention_impl=flex raised on CUDA: {exc}")
    assert torch.allclose(dense, flex, atol=1e-4, rtol=1e-4)


def _batch(codes: list[list[int]]) -> MEDSTorchBatch:
    code = torch.tensor(codes)
    return MEDSTorchBatch(
        code=code,
        numeric_value=torch.zeros_like(code, dtype=torch.float32),
        numeric_value_mask=torch.zeros_like(code, dtype=torch.bool),
        time_delta_days=torch.ones_like(code, dtype=torch.float32),
    )


def test_padding_cannot_reach_a_real_position():
    """The invariant that lets the global layers drop their mask: padding must be inert.

    Same three real events, padded to three different lengths. If padding leaked into attention, the
    hidden states at the real positions would move.
    """
    torch.manual_seed(0)
    model = Model(vocab_size=8, d_text=8, backbone=TINY).eval()
    with torch.no_grad():
        unpadded = model.backbone_states(_batch([[3, 4, 5]]))
        padded = model.backbone_states(_batch([[3, 4, 5, 0, 0, 0, 0]]))
    assert torch.allclose(unpadded, padded[:, :3], atol=1e-6)


def test_a_short_row_does_not_disturb_a_long_one():
    """Batching a short subject with a long one must not change the long one's states."""
    torch.manual_seed(0)
    model = Model(vocab_size=8, d_text=8, backbone=TINY).eval()
    with torch.no_grad():
        alone = model.backbone_states(_batch([[3, 4, 5, 6, 7]]))
        together = model.backbone_states(_batch([[3, 4, 5, 6, 7], [3, 4, 0, 0, 0]]))
    assert torch.allclose(alone[0], together[0], atol=1e-6)


def test_interior_padding_is_refused():
    """meds-torch-data right-pads; if that ever changes, fail loudly instead of attending across it."""
    require_right_padding(torch.tensor([[True, True, False, False]]))
    with pytest.raises(ValueError, match="right-padded"):
        require_right_padding(torch.tensor([[True, False, True, False]]))


def test_the_model_refuses_interior_padding_end_to_end():
    """The guard is wired into the forward pass, not just available as a helper."""
    model = Model(vocab_size=8, d_text=8, backbone=TINY).eval()
    with pytest.raises(ValueError, match="right-padded"), torch.no_grad():
        model.backbone_states(_batch([[3, 0, 5]]))
