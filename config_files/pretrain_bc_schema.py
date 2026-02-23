"""
Pydantic-settings schema for Level 1 BC (behavioral cloning) pretraining.

Configuration is loaded from (highest priority first):

  1. Constructor kwargs  — programmatic overrides (e.g. from CLI)
  2. Env vars            — PRETRAIN_BC_<FIELD>
  3. pretrain_config_bc.yaml — persisted defaults in config_files/

Example:
  from config_files.pretrain_bc_schema import BCPretrainConfig, load_pretrain_bc_config
  cfg = BCPretrainConfig()
  cfg = load_pretrain_bc_config("my_bc.yaml", epochs=20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from config_files.pretrain_schema import LightningConfig

_DEFAULT_BC_CONFIG_PATH = Path(__file__).parent / "pretrain_config_bc.yaml"


class _BCPretrainYamlSource(PydanticBaseSettingsSource):
    """Reads BCPretrainConfig fields from a YAML file."""

    def __init__(self, settings_cls: type, yaml_path: Path = _DEFAULT_BC_CONFIG_PATH) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        self._yaml_path = yaml_path
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as fh:
                self._data = yaml.safe_load(fh) or {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self._data.items()
            if k in self.settings_cls.model_fields
        }


class BCPretrainConfig(BaseSettings):
    """Configuration for Level 1 BC (behavioral cloning) pretraining.

    Priority: constructor kwargs → env (PRETRAIN_BC_*) → pretrain_config_bc.yaml → defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRETRAIN_BC_",
        env_nested_delimiter="__",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- Data ----------
    data_dir: Path = Field(
        default=Path("maps/img"),
        description="Root of the frame tree (track_id/replay_name/ with manifest.json + frames).",
    )
    output_dir: Path = Field(
        default=Path("output/ptretrain/bc"),
        description="Base output directory. Run stored in output_dir/run_NNN/ or output_dir/<run_name>/.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Subdirectory name for this run. null = auto-increment run_001, run_002, ...",
    )

    # ---------- Preprocessed cache ----------
    preprocess_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for BC cache (train.npy, train_actions.npy, etc.). null = disabled.",
    )
    cache_load_in_ram: bool = Field(default=False, description="Load cache arrays into RAM.")
    cache_build_workers: int = Field(default=0, ge=0, description="Threads for cache construction.")

    # ---------- Image ----------
    image_size: int = Field(default=64, ge=16, le=512, description="Square input resolution.")
    n_stack: int = Field(default=1, ge=1, le=32, description="Consecutive frames per sample.")

    # ---------- BC-specific ----------
    bc_mode: Literal["backbone", "full_policy", "auxiliary_head"] = Field(
        default="backbone",
        description="backbone = save encoder only; full_policy = encoder + action head; auxiliary_head = separate head as signal.",
    )
    encoder_init_path: Optional[Path] = Field(
        default=None,
        description="Path to Level 0 encoder.pt to initialize BC CNN. null = train from scratch.",
    )
    use_floats: bool = Field(default=False, description="Use float features in BC (requires float data in replays).")
    n_actions: int = Field(default=12, ge=2, description="Number of discrete actions (must match RL config.inputs).")

    # ---------- Training ----------
    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    workers: int = Field(default=4, ge=0, description="DataLoader worker processes.")
    pin_memory: bool = Field(
        default=True,
        description="Pin CPU tensors to page-locked memory for faster host→GPU transfer.",
    )
    prefetch_factor: int = Field(
        default=4,
        ge=1,
        description="DataLoader prefetch factor per worker (only when workers > 0).",
    )
    val_fraction: float = Field(default=0.1, ge=0.0, le=1.0, description="Track-level val split fraction.")
    seed: int = Field(default=42, description="RNG seed for split and reproducibility.")
    grad_clip: float = Field(default=1.0, ge=0.0, description="Gradient clipping by norm (0 = off).")
    use_tqdm: bool = Field(
        default=True,
        description="Show tqdm progress bar during native training.",
    )
    lightning: LightningConfig = Field(
        default_factory=LightningConfig,
        description="PyTorch Lightning settings (used by BC training). Nested in YAML under lightning:.",
    )

    # ---------- Validators ----------
    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("preprocess_cache_dir", "encoder_init_path", mode="before")
    @classmethod
    def _coerce_optional_path(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

    # ---------- Settings sources ----------
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _BCPretrainYamlSource(settings_cls),
        )


def load_pretrain_bc_config(
    yaml_path: Optional[Path | str] = None,
    **overrides: Any,
) -> BCPretrainConfig:
    """Load BCPretrainConfig from a YAML file, then apply overrides. Missing file uses empty dict."""
    path = Path(yaml_path) if yaml_path else _DEFAULT_BC_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data.update(overrides)
    return BCPretrainConfig.model_validate(data)
