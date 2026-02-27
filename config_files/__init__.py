"""
Configuration package for TrackMania RL.

Configuration is loaded from YAML files via Pydantic Settings.
Use: python scripts/train.py --config config_files/rl/config_default.yaml

- config_loader: load_config(), get_config(), set_config()
- config_schema: Pydantic models for validation
"""

from config_files.config_loader import get_config, load_config, set_config

__all__ = [
    "get_config",
    "load_config",
    "set_config",
]
