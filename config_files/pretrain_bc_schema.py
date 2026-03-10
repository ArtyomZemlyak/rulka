"""
Pydantic-settings schema for Level 1 BC (behavioral cloning) pretraining.

Configuration is loaded from (highest priority first):

  1. Constructor kwargs  — programmatic overrides (e.g. from CLI)
  2. Env vars            — PRETRAIN_BC_<FIELD>
  3. pretrain_config_bc.yaml — persisted defaults in config_files/pretrain/bc/

Example:
  from config_files.pretrain_bc_schema import BCPretrainConfig, load_pretrain_bc_config
  cfg = BCPretrainConfig()
  cfg = load_pretrain_bc_config("my_bc.yaml", epochs=20)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from config_files.pretrain_schema import LightningConfig

_DEFAULT_BC_CONFIG_PATH = Path(__file__).parent / "pretrain" / "bc" / "pretrain_config_bc.yaml"


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
    # Restrict training/val to these track IDs only (e.g. ['A01-Race']). null = use all tracks in data_dir.
    track_ids: Optional[list[str]] = Field(
        default=None,
        description="If set, only these track IDs are used for train/val. Useful for fine-tuning on a single track.",
    )

    # ---------- Preprocessed cache ----------
    preprocess_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for BC cache (train.npy, train_actions.npy, etc.). null = disabled.",
    )
    # Path to RL config. Required: image_size, n_actions, dense_hidden_dimension,
    # iqn_embedding_dimension, float params (when use_floats) all come from here.
    rl_config_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "rl" / "config_default.yaml",
        description="Path to RL config; all RL-matching params are loaded from it.",
    )
    cache_load_in_ram: bool = Field(default=False, description="Load cache arrays into RAM.")
    cache_build_workers: int = Field(default=0, ge=0, description="Threads for cache construction.")

    # ---------- Image ----------
    # image_size loaded from rl_config_path (neural_network.w_downsized).
    n_stack: int = Field(default=1, ge=1, le=32, description="Consecutive frames per sample.")

    # Image normalization: "01" = [0, 1] (current pretrain default); "iqn" = (x-128)/128
    # to match IQN_Network forward (better transfer when loading encoder into img_head).
    image_normalization: Literal["01", "iqn"] = Field(
        default="01",
        description="01 = [0,1]; iqn = (x-128)/128 to align with IQN at transfer.",
    )

    # ---------- BC-specific ----------
    bc_mode: Literal["backbone", "full_policy", "auxiliary_head"] = Field(
        default="backbone",
        description="backbone = save encoder only; full_policy = encoder + action head; auxiliary_head = separate head as signal.",
    )
    # current_tick = action at last frame of window (MDP-aligned: observe s_t → output a_t).
    # next_tick = action at next timestep (observe s_t → output a_{t+1}; one-step-ahead).
    bc_target: Literal["current_tick", "next_tick"] = Field(
        default="current_tick",
        description="current_tick = action at last frame (π(a_t|s_t)); next_tick = action at next timestep (π(a_{t+1}|s_t)). Ignored when bc_time_offsets_ms is set.",
    )
    # Multi-offset BC: predict action at several time offsets (ms) from the last frame. Engine physics step ≥10ms.
    # Example: [-10, 0, 10, 100] = past, current (MDP), near future, farther future. Default [0] = single head (same as current_tick).
    bc_time_offsets_ms: list[int] = Field(
        default_factory=lambda: [0],
        description="Time offsets in ms from last frame; one linear head per offset. [0] = single head (backward compat).",
    )
    # Weights for loss per offset (same length as bc_time_offsets_ms). Default all 1.0. E.g. weight 0 more.
    bc_offset_weights: Optional[list[float]] = Field(
        default=None,
        description="Loss weight per offset. None = all 1.0. Length must match bc_time_offsets_ms.",
    )
    encoder_init_path: Optional[Path] = Field(
        default=None,
        description="Path to Level 0 encoder.pt to initialize BC CNN. null = train from scratch.",
    )
    # Path to a previous BC run directory (e.g. output/ptretrain/bc/v3_multi_offset) to load full model from iqn_bc.pt for fine-tuning. Overrides encoder_init_path when use_full_iqn.
    bc_resume_run_dir: Optional[Path] = Field(
        default=None,
        description="Path to BC run dir containing iqn_bc.pt (and pretrain_meta.json) to resume/fine-tune from. Use with use_full_iqn.",
    )
    use_floats: bool = Field(default=False, description="Use float features in BC.")
    save_float_head: bool = Field(
        default=True,
        description="When use_floats True, save float head weights for IQN float_feature_extractor injection (future).",
    )

    # ---------- Actions head (same layout as IQN A_head for transfer) ----------
    use_actions_head: bool = Field(
        default=False,
        description="If True, action head is two-layer MLP matching IQN A_head: Linear(dense, dense_hidden//2) -> LeakyReLU -> Linear(..., n_actions). Enables save_actions_head for RL A_head injection.",
    )
    save_actions_head: bool = Field(
        default=True,
        description="When use_actions_head True, save action head weights as actions_head.pt for IQN A_head injection in RL. With multi-offset, saves the first head (offset 0) only so it can be merged into IQN A_head.",
    )
    merge_actions_head: bool = Field(
        default=False,
        description="When True and multi-offset (a_head or full IQN), save all offset heads into the same file: actions_head.pt gets {offset_0: ..., offset_1: ...}; iqn_bc.pt gets A_head.* (first) plus A_head_offset_1.*, A_head_offset_2.*, .... If False, only the first head is saved (default for RL merge).",
    )
    # ---------- Full IQN in BC (two variants) ----------
    use_full_iqn: bool = Field(
        default=False,
        description="If True, train full IQN network in BC (img_head + float_feature_extractor + iqn_fc + A_head + V_head) for 1:1 transfer. Requires use_floats and rl_config_path.",
    )
    full_iqn_random_tau: bool = Field(
        default=False,
        description="When use_full_iqn True: if True sample tau ~ U(0,1) per batch; if False use fixed tau=0.5. Only used when use_full_iqn is True.",
    )
    # dense_hidden_dimension, iqn_embedding_dimension loaded from rl_config_path.
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability on features before action head (training only). Can reduce overfitting.",
    )
    action_head_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout between the two Linear layers of A_head MLP (use_actions_head only). Saved state_dict is remapped to IQN layout (no dropout in file).",
    )

    # ---------- Training ----------
    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 penalty for optimizer (AdamW when > 0, else Adam). Can reduce overfitting for larger heads.",
    )
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

    @field_validator("preprocess_cache_dir", "encoder_init_path", "bc_resume_run_dir", mode="before")
    @classmethod
    def _coerce_optional_path(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

    @field_validator("rl_config_path", mode="before")
    @classmethod
    def _coerce_rl_config_path(cls, v: Any) -> Path:
        if v is None:
            return Path(__file__).resolve().parent / "rl" / "config_default.yaml"
        return Path(v)

    @model_validator(mode="after")
    def _require_rl_config_exists(self) -> "BCPretrainConfig":
        if not Path(self.rl_config_path).exists():
            raise ValueError(f"rl_config_path {self.rl_config_path} does not exist")
        return self

    @model_validator(mode="after")
    def _require_floats_when_full_iqn(self) -> "BCPretrainConfig":
        if self.use_full_iqn and not self.use_floats:
            raise ValueError("use_full_iqn is True but use_floats is False; full IQN requires float inputs.")
        return self

    @model_validator(mode="after")
    def _bc_offset_weights_length(self) -> "BCPretrainConfig":
        if self.bc_offset_weights is not None and len(self.bc_offset_weights) != len(self.bc_time_offsets_ms):
            raise ValueError(
                f"bc_offset_weights length ({len(self.bc_offset_weights)}) must match bc_time_offsets_ms ({len(self.bc_time_offsets_ms)})"
            )
        return self
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


def _load_pretrain_bc_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file. If it contains _base_, load that file first and merge (current file wins)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base_name = data.pop("_base_", None)
    if base_name is None:
        return data
    base_path = path.parent / base_name
    base_data = _load_pretrain_bc_yaml(base_path)
    base_data.update(data)
    return base_data


def load_pretrain_bc_config(
    yaml_path: Optional[Path | str] = None,
    **overrides: Any,
) -> BCPretrainConfig:
    """Load BCPretrainConfig from a YAML file, then apply overrides. Missing file uses empty dict.
    If the YAML contains _base_: <filename>, that file is loaded first and the current file overrides it."""
    path = Path(yaml_path) if yaml_path else _DEFAULT_BC_CONFIG_PATH
    data = _load_pretrain_bc_yaml(path)
    data.update(overrides)
    return BCPretrainConfig.model_validate(data)
