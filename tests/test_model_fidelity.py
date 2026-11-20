"""Check the port against a transcription of the official Theano equations.

`_reference_scores` is `build_model` from `mp2893/retain` @ 9fd39c46 rewritten in NumPy, one sequence at a
time, with no batching, no packing and no masking. If the batched PyTorch implementation agrees with it,
then padding, packing, the two reversals and the attention mask are all bookkeeping rather than model
changes -- which is the only claim the rest of the report rests on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from retain_meds.config import RETAINConfig  # noqa: E402
from retain_meds._model import Retain  # noqa: E402
from retain_meds._training import collate, roc_auc, score  # noqa: E402
from retain_meds._windows import SequenceBatch  # noqa: E402


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _reference_gru(inputs, weight_ih, weight_hh, bias, hidden_size):
    """`gru_layer`'s `stepFn`, verbatim.

    `W_gru` is `weight_ih.T`, `U_gru` is `weight_hh.T` and `b_gru` is the single Theano bias, added to
    `W x` only. That is why `bias_hh` is pinned to zero in the model: with it zero, this is exactly what
    `nn.GRU` computes.
    """
    state = np.zeros(hidden_size, dtype=np.float64)
    states = []
    for step in inputs:
        wx = step @ weight_ih.T + bias
        uh = state @ weight_hh.T
        reset = _sigmoid(wx[:hidden_size] + uh[:hidden_size])
        update = _sigmoid(wx[hidden_size : 2 * hidden_size] + uh[hidden_size : 2 * hidden_size])
        candidate = np.tanh(wx[2 * hidden_size :] + reset * uh[2 * hidden_size :])
        state = update * state + (1.0 - update) * candidate
        states.append(state)
    return np.array(states)


def _reference_scores(model, sequences, gaps=None):
    """`build_model` for a list of visit sequences, each a list of code-index lists."""
    parameters = {name: value.detach().numpy().astype(np.float64) for name, value in model.named_parameters()}
    config = model.config
    alpha_hidden = config.alpha_hidden_dim_size
    beta_hidden = config.beta_hidden_dim_size

    scores = []
    for index, visits in enumerate(sequences):
        # Step 1: v_j = W_emb x_j, with x_j binary, so the embedding of a visit is the sum of the rows
        # of W_emb named by its codes.
        embeddings = np.array(
            [parameters["W_emb"][list(codes)].sum(axis=0) for codes in visits], dtype=np.float64
        ).reshape(len(visits), config.embed_size)
        recurrent = embeddings
        if config.use_time:
            durations = np.asarray(gaps[index], dtype=np.float64).reshape(-1, 1)
            if config.use_log_time:
                durations = np.log(durations + 1.0)
            recurrent = np.concatenate([embeddings, durations], axis=1)

        # Steps 2 and 3: run both GRUs over the reversed sequence, then put the states back in time
        # order, and scale by the 0.5 factor the official code applies.
        reversed_input = recurrent[::-1]
        alpha_states = (
            _reference_gru(
                reversed_input,
                parameters["gru_alpha.weight_ih_l0"],
                parameters["gru_alpha.weight_hh_l0"],
                parameters["gru_alpha.bias_ih_l0"],
                alpha_hidden,
            )[::-1]
            * 0.5
        )
        beta_states = (
            _reference_gru(
                reversed_input,
                parameters["gru_beta.weight_ih_l0"],
                parameters["gru_beta.weight_hh_l0"],
                parameters["gru_beta.bias_ih_l0"],
                beta_hidden,
            )[::-1]
            * 0.5
        )
        logits = alpha_states @ parameters["w_alpha.weight"].T + parameters["w_alpha.bias"]
        logits = logits.reshape(-1)
        alpha = np.exp(logits - logits.max())
        alpha = alpha / alpha.sum()
        beta = np.tanh(beta_states @ parameters["W_beta.weight"].T + parameters["W_beta.bias"])

        # Steps 4 and 5.
        context = (alpha[:, None] * beta * embeddings).sum(axis=0)
        logit = context @ parameters["w_output.weight"].reshape(-1) + parameters["w_output.bias"][0]
        scores.append(_sigmoid(float(logit)))
    return np.array(scores)


def _batch_from(sequences, times=None):
    """Pack Python visit lists into the CSR layout `build_sequences` produces."""
    row_ptr = [0]
    visit_ptr = [0]
    codes: list[int] = []
    visit_time: list[int] = []
    for index, visits in enumerate(sequences):
        row_ptr.append(row_ptr[-1] + len(visits))
        for position, visit in enumerate(visits):
            codes.extend(visit)
            visit_ptr.append(visit_ptr[-1] + len(visit))
            visit_time.append(
                int(times[index][position]) if times is not None else position * 86_400_000_000
            )
    return SequenceBatch(
        keys=np.column_stack(
            (np.arange(len(sequences), dtype=np.int64), np.zeros(len(sequences), dtype=np.int64))
        ),
        row_ptr=np.array(row_ptr, dtype=np.int64),
        visit_ptr=np.array(visit_ptr, dtype=np.int64),
        visit_time=np.array(visit_time, dtype=np.int64),
        codes=np.array(codes, dtype=np.int32),
    )


def _model(**overrides):
    config = RETAINConfig(
        embed_size=6, alpha_hidden_dim_size=5, beta_hidden_dim_size=4, batch_size=4, **overrides
    )
    torch.manual_seed(0)
    model = Retain(12, config).double()
    # Give every bias a non-zero value so the test would notice a bias dropped or misplaced.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "bias" in name and "bias_hh" not in name:
                parameter.copy_(torch.linspace(-0.3, 0.3, parameter.numel()).reshape(parameter.shape))
    model.eval()
    return model


SEQUENCES = [
    [[1, 4], [2], [0, 3, 7]],
    [[5]],
    [[1], [1, 2], [3, 9], [0], [8, 11]],
    [[6, 7]],
]


def test_batched_forward_matches_the_theano_reference():
    model = _model()
    batch = _batch_from(SEQUENCES)
    actual = score(model, batch, model.config, torch.device("cpu"))
    expected = _reference_scores(model, SEQUENCES)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_forward_matches_the_reference_with_visit_durations():
    model = _model(use_time=True, use_log_time=True)
    day = 86_400_000_000
    times = [
        [0, 3 * day, 30 * day],
        [0],
        [0, day, 2 * day, 40 * day, 41 * day],
        [0],
    ]
    batch = _batch_from(SEQUENCES, times)
    gaps = [np.diff(np.array(row, dtype=np.float64) / day, prepend=row[0] / day) for row in times]
    actual = score(model, batch, model.config, torch.device("cpu"))
    expected = _reference_scores(model, SEQUENCES, gaps)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_a_sequence_scores_the_same_alone_as_inside_a_padded_batch():
    """Padding, packing and the attention mask must not touch a shorter row's score."""
    model = _model()
    together = score(model, _batch_from(SEQUENCES), model.config, torch.device("cpu"))
    for index, sequence in enumerate(SEQUENCES):
        alone = score(model, _batch_from([sequence]), model.config, torch.device("cpu"))
        assert alone[0] == pytest.approx(together[index], rel=1e-9, abs=1e-11)


def test_an_empty_history_scores_the_learned_intercept():
    """The documented policy for a key whose subject has no prior events: no exclusion, base rate."""
    model = _model()
    batch = _batch_from([[]])
    actual = score(model, batch, model.config, torch.device("cpu"))
    expected = _sigmoid(float(model.w_output.bias.detach()))
    assert actual[0] == pytest.approx(expected, rel=1e-9)


def test_attention_weights_sum_to_one_over_real_visits_only():
    model = _model()
    batch = _batch_from(SEQUENCES)
    rows = np.arange(len(SEQUENCES))
    tensors = collate(batch, rows, model.config, torch.device("cpu"))
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
    alpha = output.alpha.numpy()
    np.testing.assert_allclose(alpha.sum(axis=1), np.ones(len(SEQUENCES)), rtol=1e-12)
    for index, sequence in enumerate(SEQUENCES):
        assert alpha[index, len(sequence) :].sum() == pytest.approx(0.0, abs=1e-12)


def test_the_recurrent_bias_that_theano_does_not_have_is_pinned_to_zero():
    model = _model()
    for gru in (model.gru_alpha, model.gru_beta):
        assert not gru.bias_hh_l0.requires_grad
        assert torch.count_nonzero(gru.bias_hh_l0) == 0
    assert all(parameter.requires_grad for parameter in model.trainable_parameters())


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([1, 0, 1, 0, 1], [0.9, 0.9, 0.2, 0.2, 0.5]),
        ([0, 1], [0.5, 0.5]),
    ],
)
def test_roc_auc_matches_the_pairwise_definition(labels, scores):
    labels = np.array(labels, dtype=bool)
    scores = np.array(scores, dtype=float)
    positive = scores[labels]
    negative = scores[~labels]
    pairs = (positive[:, None] > negative[None, :]).sum() + 0.5 * (
        positive[:, None] == negative[None, :]
    ).sum()
    assert roc_auc(labels, scores) == pytest.approx(pairs / (positive.size * negative.size))


def test_roc_auc_is_undefined_for_a_single_class():
    assert roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_sequence_construction_refuses_to_see_ground_truth():
    from retain_meds._windows import build_sequences

    frame = pl.DataFrame(
        {"subject_id": [1], "prediction_time": [None], "boolean_value": [True]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us"), "boolean_value": pl.Boolean},
    )
    with pytest.raises(ValueError, match="never see label columns"):
        build_sequences(None, frame, RETAINConfig())
