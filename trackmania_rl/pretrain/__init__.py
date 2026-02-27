"""
Visual pretraining package for IQN backbone (Level 0).

Public API:
  PretrainConfig         — pydantic-settings model (loads from config_files/pretrain/vis/pretrain_config.yaml)
  load_pretrain_config   — load from a custom YAML path
  train_pretrain         — unified training entry point (native or Lightning)
  save_encoder_artifact  — save encoder.pt + pretrain_meta.json + metrics.csv
  load_encoder_artifact  — load and validate saved artifact

Configuration:
  config_files/pretrain/vis/pretrain_config.yaml  — default values (edit to change project-wide defaults)
  PRETRAIN_<FIELD> env vars          — runtime overrides
  Constructor kwargs                 — programmatic overrides (highest priority)

Quick start:
  from trackmania_rl.pretrain import PretrainConfig, train_pretrain
  cfg = PretrainConfig(task="simclr", epochs=100)
  train_pretrain(cfg)
"""

from config_files.pretrain_schema import PretrainConfig, load_pretrain_config
from trackmania_rl.pretrain.train import train_pretrain
from trackmania_rl.pretrain.export import save_encoder_artifact, load_encoder_artifact

__all__ = [
    "PretrainConfig",
    "load_pretrain_config",
    "train_pretrain",
    "save_encoder_artifact",
    "load_encoder_artifact",
]
