"""Shared policy forward bundle for PPO and future preference-based methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PolicyOutput:
    """One forward pass at fixed observation(s).

    Discrete: ``logits`` (B, n_actions). Continuous (future): ``action_mean`` / ``action_log_std``.
    ``value`` is V(s) with shape (B, 1).
    """

    logits: Optional[torch.Tensor]
    value: torch.Tensor
    action_mean: Optional[torch.Tensor] = None
    action_log_std: Optional[torch.Tensor] = None
