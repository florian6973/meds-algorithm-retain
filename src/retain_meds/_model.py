"""RETAIN, transcribed from `mp2893/retain` @ 9fd39c46 (`retain.py`, `build_model`).

Steps 1-5 of the paper, with the official code as the tie-breaker wherever the paper leaves a choice open:

    v_j = W_emb x_j                                 Step 1, linear, no bias, no activation
    g_i..g_1 = GRU_alpha(v_i..v_1)                  Step 2, reverse time
    e_j = w_alpha . g_j + b_alpha ; alpha = softmax(e) over visits
    h_i..h_1 = GRU_beta(v_i..v_1)                   Step 3, reverse time
    beta_j = tanh(W_beta h_j + b_beta)
    c_i = sum_j alpha_j beta_j * v_j                Step 4
    y = sigmoid(w_output . c_i + b_output)          Step 5, binary case

Three transcription notes:

* `torch.nn.GRU` is gate-for-gate identical to the official `gru_layer`: both compute
  `n = tanh(W_in x + r * (W_hn h + b_hn))` with the reset gate applied to the whole recurrent term, both
  use `h' = (1 - z) n + z h`, and both order the packed gates `(r, z, n)`. No custom cell is needed.
  One difference is real and is removed here rather than tolerated: Theano's `gru_layer` has a single
  bias, added to `W x` only, while `nn.GRU` has a second recurrent bias whose `n` component sits *inside*
  the reset-gate product as `r * b_hn`, a term the official cell does not have. `bias_hh` is therefore
  pinned to zero and excluded from optimisation, which makes the two cells algebraically identical.
* The official code multiplies both GRU outputs by `0.5` before the attention heads. That factor is in
  `retain.py` and again in `test_retain.py`, so it is deliberate upstream rather than a typo, and it is
  kept here. It is not in the paper: on alpha it acts as a softmax temperature.
* Dropout is applied to the embedding *in place*, so the same dropped `v` feeds the attention GRUs and the
  context sum of Step 4. `build_model` rebinds `emb` before using it in `attentionStep`.

Only the final timestep is computed. The official training graph builds `c_t` for every prefix but reads
`y_hat` at `lengths - 1`, because this code performs sequence classification; computing `c_T` alone is the
same value, and is what the official `test_retain.py` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils

from .config import RetainConfig


@dataclass
class RetainOutput:
    """Logits plus the quantities Eq. (5) needs to attribute a score to individual codes."""

    logits: torch.Tensor
    alpha: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None


def reverse_padded(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each row's first `length` positions and leave its padding untouched."""
    batch, total = sequence.shape[0], sequence.shape[1]
    positions = torch.arange(total, device=sequence.device).unsqueeze(0)
    limits = lengths.to(sequence.device).unsqueeze(1)
    flipped = (limits - 1 - positions).clamp(min=0)
    index = torch.where(positions < limits, flipped, positions)
    return sequence.gather(1, index.unsqueeze(-1).expand(-1, -1, sequence.shape[2]))


class Retain(nn.Module):
    def __init__(self, vocabulary_size: int, config: RetainConfig) -> None:
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.config = config
        embed = config.embed_size
        self.W_emb = nn.Parameter(torch.empty(self.vocabulary_size, embed))
        self.gru_alpha = nn.GRU(config.gru_input_size, config.alpha_hidden_dim_size, batch_first=True)
        self.gru_beta = nn.GRU(config.gru_input_size, config.beta_hidden_dim_size, batch_first=True)
        self.w_alpha = nn.Linear(config.alpha_hidden_dim_size, 1)
        self.W_beta = nn.Linear(config.beta_hidden_dim_size, embed)
        self.w_output = nn.Linear(embed, 1)
        self.embedding_dropout = nn.Dropout(p=1.0 - config.keep_prob_emb)
        self.context_dropout = nn.Dropout(p=1.0 - config.keep_prob_context)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """`get_random_weight` is uniform(-0.1, 0.1); every bias in `init_params` starts at zero."""
        for name, parameter in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)
            else:
                nn.init.uniform_(parameter, -0.1, 0.1)
        for gru in (self.gru_alpha, self.gru_beta):
            gru.bias_hh_l0.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything the official `tparams` optimises, which excludes the pinned `bias_hh`."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        codes: torch.Tensor,
        offsets: torch.Tensor,
        slots: torch.Tensor,
        lengths: torch.Tensor,
        gaps: torch.Tensor | None = None,
        *,
        n_rows: int,
        n_visits: int,
        return_attention: bool = False,
    ) -> RetainOutput:
        """Score `n_rows` sequences padded to `n_visits` visits.

        `codes`/`offsets` are an `EmbeddingBag` view of the batch: one bag per real visit, holding that
        visit's vocabulary indices. `x_j` is binary, so summing the rows of `W_emb` named by the bag *is*
        `W_emb x_j`, without ever materialising the `|C|`-wide one-hot matrix. `slots` places each bag at
        its `row * n_visits + position` cell of the padded grid.
        """
        device = self.W_emb.device
        embed = self.config.embed_size
        flat = torch.zeros(n_rows * n_visits, embed, device=device, dtype=self.W_emb.dtype)
        if codes.numel():
            bags = F.embedding_bag(
                codes, self.W_emb, offsets=offsets, mode="sum", include_last_offset=True
            )
            flat = flat.index_copy(0, slots, bags)
        visits = self.embedding_dropout(flat.view(n_rows, n_visits, embed))

        recurrent_input = visits
        if self.config.use_time:
            if gaps is None:
                raise ValueError("use_time is set but no visit gaps were supplied")
            durations = torch.zeros(n_rows * n_visits, 1, device=device, dtype=visits.dtype)
            if codes.numel():
                durations = durations.index_copy(0, slots, gaps.to(visits.dtype).unsqueeze(1))
            durations = durations.view(n_rows, n_visits, 1)
            if self.config.use_log_time:
                durations = torch.log1p(durations)
            recurrent_input = torch.cat([visits, durations], dim=2)

        reversed_input = reverse_padded(recurrent_input, lengths)
        packed = rnn_utils.pack_padded_sequence(
            reversed_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        alpha_states, _ = self.gru_alpha(packed)
        beta_states, _ = self.gru_beta(packed)
        alpha_states, _ = rnn_utils.pad_packed_sequence(
            alpha_states, batch_first=True, total_length=n_visits
        )
        beta_states, _ = rnn_utils.pad_packed_sequence(
            beta_states, batch_first=True, total_length=n_visits
        )
        alpha_states = reverse_padded(alpha_states, lengths) * 0.5
        beta_states = reverse_padded(beta_states, lengths) * 0.5

        mask = torch.arange(n_visits, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)
        logits_alpha = self.w_alpha(alpha_states).squeeze(-1)
        alpha = torch.softmax(logits_alpha.masked_fill(~mask, float("-inf")), dim=1)
        beta = torch.tanh(self.W_beta(beta_states))

        context = (alpha.unsqueeze(-1) * beta * visits).sum(dim=1)
        logits = self.w_output(self.context_dropout(context)).squeeze(-1)
        if not return_attention:
            return RetainOutput(logits=logits)
        return RetainOutput(logits=logits, alpha=alpha, beta=beta, embeddings=visits)

    def regularization(self) -> torch.Tensor:
        """The four L2 penalties of `build_model`, on weights only, never on biases."""
        config = self.config
        penalty = config.L2_output * (self.w_output.weight**2).sum()
        penalty = penalty + config.L2_alpha * (self.w_alpha.weight**2).sum()
        penalty = penalty + config.L2_beta * (self.W_beta.weight**2).sum()
        return penalty + config.L2_emb * (self.W_emb**2).sum()
