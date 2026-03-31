"""Shared IQN path: implicit τ sampling, cosine embedding, ``iqn_fc``, state multiply, dueling readout.

Used by classic :class:`trackmania_rl.agents.iqn.IQN_Network` and
:class:`trackmania_rl.nn_build.iqn_multimodal.IQNSharedBackboneNetwork` (same math, different state features).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def forward_iqn_q_values_from_state_features(
    state_features: Tensor,
    num_quantiles: int,
    tau: Optional[Tensor],
    *,
    iqn_embedding_dimension: int,
    iqn_fc: nn.Module,
    V_head: nn.Module,
    A_head: nn.Module,
    A_head_multi: Optional[nn.Module],
    n_actions: int,
    n_actions_per_block: int,
) -> Tuple[Tensor, Tensor]:
    """Map per-batch state vectors (B, D) to Q samples (B * num_quantiles, …) and τ tensor."""
    batch_size = state_features.shape[0]
    dev = state_features.device
    if tau is None:
        tau = (
            torch.arange(num_quantiles // 2, device=dev, dtype=torch.float32)
            .repeat_interleave(batch_size)
            .unsqueeze(1)
            + torch.rand(size=(batch_size * num_quantiles // 2, 1), device=dev, dtype=torch.float32)
        ) / num_quantiles
        tau = torch.cat((tau, 1 - tau), dim=0)
    quantile_net = torch.cos(
        torch.arange(1, iqn_embedding_dimension + 1, 1, device=tau.device, dtype=tau.dtype) * math.pi * tau
    )
    quantile_net = quantile_net.expand([-1, iqn_embedding_dimension])
    quantile_net = iqn_fc(quantile_net)
    h = state_features.repeat(num_quantiles, 1)
    h = h * quantile_net

    V = V_head(h)
    if A_head_multi is None:
        A = A_head(h)
        Q = V + A - A.mean(dim=-1).unsqueeze(-1)
        return Q, tau
    a_hidden = A_head(h)
    A = A_head_multi(a_hidden).view(-1, n_actions_per_block, n_actions)
    Q = V.unsqueeze(1) + A - A.mean(dim=-1, keepdim=True)
    return Q, tau
