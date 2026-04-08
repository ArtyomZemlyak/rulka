"""On-policy RL utilities (PPO GAE, losses) and preference / group-relative objectives."""

from trackmania_rl.agents.policy_optimization.dpo import dpo_preference_loss, sum_log_probs_evaluate
from trackmania_rl.agents.policy_optimization.grpo import grpo_policy_objective, group_relative_advantages
from trackmania_rl.agents.policy_optimization.ppo import compute_gae, ppo_loss_components
from trackmania_rl.agents.policy_optimization.rollout_rewards import ppo_rewards_and_dones_from_rollout
from trackmania_rl.agents.policy_optimization.types import PolicyOutput

__all__ = [
    "PolicyOutput",
    "compute_gae",
    "dpo_preference_loss",
    "grpo_policy_objective",
    "group_relative_advantages",
    "ppo_loss_components",
    "ppo_rewards_and_dones_from_rollout",
    "sum_log_probs_evaluate",
]
