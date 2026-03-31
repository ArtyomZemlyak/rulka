"""On-policy RL utilities (PPO GAE, losses). DPO/GRPO can add sibling modules later."""

from trackmania_rl.agents.policy_optimization.ppo import compute_gae, ppo_loss_components
from trackmania_rl.agents.policy_optimization.rollout_rewards import ppo_rewards_and_dones_from_rollout
from trackmania_rl.agents.policy_optimization.types import PolicyOutput

__all__ = [
    "PolicyOutput",
    "compute_gae",
    "ppo_loss_components",
    "ppo_rewards_and_dones_from_rollout",
]
