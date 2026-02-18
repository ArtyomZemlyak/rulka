"""
Pydantic-settings schema for Level 0 visual pretraining.

Configuration is loaded from (highest priority first):

  1. Constructor kwargs  — programmatic overrides (e.g. from CLI)
  2. Env vars            — PRETRAIN_<FIELD> (flat fields) or PRETRAIN_LIGHTNING__<FIELD>
                           (nested lightning config).  Examples:
                             PRETRAIN_TASK=simclr
                             PRETRAIN_LIGHTNING__PATIENCE=5
  3. pretrain_config.yaml — persisted defaults in config_files/

Examples
--------
# Load defaults from pretrain_config.yaml + env vars:
from config_files.pretrain_schema import PretrainConfig
cfg = PretrainConfig()

# Override task and epochs (YAML is still the base):
cfg = PretrainConfig(task="simclr", epochs=100)

# Override nested lightning settings:
from config_files.pretrain_schema import LightningConfig
cfg = PretrainConfig(lightning=LightningConfig(tensorboard=False, patience=5))

# Override via environment (PowerShell):
#   $env:PRETRAIN_TASK = "simclr"
#   $env:PRETRAIN_LIGHTNING__PATIENCE = "5"

# Load from a custom YAML file:
from config_files.pretrain_schema import load_pretrain_config
cfg = load_pretrain_config("my_experiment.yaml")

# Load from custom YAML, then apply additional overrides:
cfg = load_pretrain_config("my_experiment.yaml", epochs=200)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "pretrain_config.yaml"


# ---------------------------------------------------------------------------
# Custom YAML settings source (pydantic-settings >= 2.0 compatible)
# ---------------------------------------------------------------------------

class _PretrainYamlSource(PydanticBaseSettingsSource):
    """Reads PretrainConfig fields from a YAML file.

    Works with pydantic-settings 2.0+.  The path is resolved at instantiation
    time so the default config is always found regardless of CWD.
    """

    def __init__(self, settings_cls: type, yaml_path: Path = _DEFAULT_CONFIG_PATH) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
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


# ---------------------------------------------------------------------------
# LightningConfig — all PyTorch Lightning-specific settings
# ---------------------------------------------------------------------------

class LightningConfig(BaseModel):
    """PyTorch Lightning training loop settings.

    These are used only when ``framework='lightning'``.
    Nested under ``lightning:`` in pretrain_config.yaml.
    Env var override: ``PRETRAIN_LIGHTNING__<FIELD>`` (double-underscore delimiter).
    """

    # --- Loggers ---
    tensorboard: bool = Field(
        default=True,
        description="Enable TensorBoard logger.",
    )
    tensorboard_dir: str = Field(
        default="tb",
        description=(
            "TensorBoard log subdirectory relative to output_dir. "
            "Full path: output_dir/tensorboard_dir/"
        ),
    )
    csv_logger: bool = Field(
        default=True,
        description="Enable CSV logger (per-epoch metrics → output_dir/csv_dir/metrics.csv).",
    )
    csv_dir: str = Field(
        default="csv",
        description="CSV log subdirectory relative to output_dir.",
    )

    # --- Checkpoints ---
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Checkpoint subdirectory relative to output_dir.",
    )
    save_top_k: int = Field(
        default=1,
        ge=-1,
        description=(
            "Number of best checkpoints to keep.  "
            "-1 = keep all;  0 = no checkpoints;  1 = best only."
        ),
    )
    checkpoint_monitor: str = Field(
        default="auto",
        description=(
            "Metric to monitor for ModelCheckpoint and EarlyStopping. "
            "'auto' uses 'val_loss' when val_fraction > 0, else 'train_loss'."
        ),
    )

    # --- Early stopping ---
    early_stopping: bool = Field(
        default=True,
        description="Enable EarlyStopping callback.",
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Number of epochs with no improvement before stopping.",
    )
    min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum change in monitored metric to qualify as an improvement.",
    )

    # --- Trainer hardware ---
    accelerator: str = Field(
        default="auto",
        description=(
            "PyTorch Lightning accelerator: 'auto' | 'cpu' | 'gpu' | 'tpu' | 'mps'. "
            "'auto' lets Lightning pick the best available device."
        ),
    )
    precision: Literal["auto", "32-true", "16-mixed", "bf16-mixed"] = Field(
        default="auto",
        description=(
            "Training precision. "
            "'auto': 16-mixed if CUDA is available, else 32-true. "
            "'16-mixed': mixed FP16 (fast, less VRAM). "
            "'bf16-mixed': BF16 (RTX 30xx+; more stable than FP16). "
            "'32-true': full FP32 (safe default for CPU / debugging)."
        ),
    )
    devices: Optional[int] = Field(
        default=None,
        description="Number of devices (GPUs/CPUs) to use.  None = let Lightning decide.",
    )

    # --- Logging frequency ---
    log_every_n_steps: int = Field(
        default=50,
        ge=1,
        description=(
            "Log metrics every N training steps. "
            "Capped automatically to batch count // 4 to avoid empty-step warnings."
        ),
    )

    # --- Misc trainer flags ---
    deterministic: bool = Field(
        default=False,
        description=(
            "Make training deterministic (reproducible but slower). "
            "Also set seed in PretrainConfig for full reproducibility."
        ),
    )
    enable_model_summary: bool = Field(
        default=True,
        description="Print model parameter summary at the start of training.",
    )
    profiler: Optional[str] = Field(
        default=None,
        description=(
            "PyTorch Lightning profiler: None | 'simple' | 'advanced' | 'pytorch'. "
            "Use 'simple' to see per-step timing without much overhead."
        ),
    )
    float32_matmul_precision: Optional[Literal["highest", "high", "medium"]] = Field(
        default="medium",
        description=(
            "Sets torch.set_float32_matmul_precision() to unlock Tensor Core usage "
            "on Ampere / Ada / Blackwell GPUs (RTX 30xx / 40xx / 50xx). "
            "'medium' = recommended (best speed, negligible FP32 precision loss). "
            "'high' = slightly more precise. "
            "'highest' = full FP32, same as not setting (disables Tensor Cores). "
            "null = do not call (PyTorch default behaviour)."
        ),
    )


# ---------------------------------------------------------------------------
# PretrainConfig schema
# ---------------------------------------------------------------------------

class PretrainConfig(BaseSettings):
    """All hyperparameters for one Level 0 visual pretraining run.

    Configuration priority (highest → lowest):

      Constructor kwargs  → env vars (PRETRAIN_*)  → pretrain_config.yaml  → field defaults

    The ``pretrain_config.yaml`` file lives in ``config_files/`` and is always
    resolved relative to that directory (independent of working directory).
    """

    model_config = SettingsConfigDict(
        env_prefix="PRETRAIN_",
        env_nested_delimiter="__",   # enables PRETRAIN_LIGHTNING__TENSORBOARD=false
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- Data ----------
    data_dir: Path = Field(
        default=Path("maps/img"),
        description="Root of the frame tree (maps/img/<track_id>/<replay_name>/).",
    )
    output_dir: Path = Field(
        default=Path("pretrain_visual_out"),
        description=(
            "Base output directory.  Each run is stored in a versioned subdirectory "
            "output_dir/run_NNN/ (when run_name is null) or output_dir/<run_name>/."
        ),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Subdirectory name for this run, created inside output_dir. "
            "null = auto-increment: run_001, run_002, ... "
            "Explicit name (e.g. 'ae_baseline', 'simclr_v2') lets you label experiments."
        ),
    )

    # ---------- Preprocessed data cache ----------
    preprocess_cache_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory for preprocessed frame cache (train.npy / val.npy / cache_meta.json). "
            "null = disabled; raw images are decoded on-the-fly from data_dir (original behaviour). "
            "When set, train_pretrain() checks whether the cache is valid and, if not, "
            "builds it automatically from data_dir before training. "
            "The cache encodes image_size, n_stack, val_fraction and seed; any change to these "
            "fields invalidates the cache and triggers a rebuild. "
            "Adding / removing replays in data_dir also triggers a rebuild (source_signature check). "
            "Example: cache/pretrain_64"
        ),
    )
    cache_load_in_ram: bool = Field(
        default=False,
        description=(
            "Load the preprocessed cache arrays fully into RAM before training starts. "
            "Speeds up random-access I/O on small datasets (< ~8 GB). "
            "On large datasets use the default False (memory-mapped reads, OS page cache). "
            "Only has effect when preprocess_cache_dir is set."
        ),
    )
    cache_build_workers: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of threads used for parallel frame loading during cache construction. "
            "0 = single-threaded (safe default; avoid GIL contention on Windows). "
            "Increase (e.g. 4–8) when disk I/O is the preprocessing bottleneck. "
            "Only has effect when the cache needs to be (re)built."
        ),
    )

    # ---------- Task ----------
    task: Literal["ae", "vae", "simclr"] = Field(
        default="ae",
        description=(
            "'ae': autoencoder (MSE). "
            "'vae': variational AE (ELBO). "
            "'simclr': contrastive (NT-Xent)."
        ),
    )
    framework: Literal["native", "lightly", "lightning"] = Field(
        default="native",
        description=(
            "'native': pure PyTorch. "
            "'lightly': Lightly SimCLR (pip install lightly). "
            "'lightning': PyTorch Lightning (pip install lightning)."
        ),
    )

    # ---------- Image / stacking ----------
    image_size: int = Field(
        default=64, ge=16, le=512,
        description="Square input resolution. Must match IQN w_downsized / h_downsized.",
    )
    n_stack: int = Field(default=1, ge=1, le=32, description="Consecutive frames per sample.")
    stack_mode: Literal["channel", "concat"] = Field(
        default="channel",
        description=(
            "'channel': N-ch encoder (needs kernel-averaging for IQN). "
            "'concat': 1-ch encoder + fusion — IQN-compatible (recommended for n_stack > 1)."
        ),
    )

    # ---------- Training ----------
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    workers: int = Field(
        default=4,
        ge=0,
        description=(
            "DataLoader worker processes for parallel image loading. "
            "0 = single process (CPU bottleneck, GPU starvation). "
            "Recommended: CPU_cores // 2. On Windows uses spawn (not fork)."
        ),
    )
    pin_memory: bool = Field(
        default=True,
        description=(
            "Pin DataLoader tensors to page-locked memory for faster host→GPU transfers. "
            "Set False when training on CPU only."
        ),
    )
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description=(
            "Number of batches each DataLoader worker pre-fetches. "
            "Only effective when workers > 0. Higher = more RAM, better GPU utilisation."
        ),
    )
    grad_clip: float = Field(default=1.0, ge=0.0, description="Gradient clipping (0 = off).")

    # ---------- Task-specific ----------
    vae_latent: int = Field(default=64, ge=2, description="VAE latent dimension.")
    proj_dim: int = Field(default=128, ge=2, description="SimCLR projection head dimension.")
    temperature: float = Field(default=0.5, gt=0.0, description="SimCLR NT-Xent temperature.")
    kl_weight: float = Field(default=1e-3, ge=0.0, description="VAE KL coefficient β.")

    # ---------- Data split ----------
    val_fraction: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description=(
            "Track-level val split fraction. "
            "0.0 = no split (no val_loss in metrics, early stopping on train_loss). "
            "0.1 = 10 % of tracks held out (recommended; enables val_loss and "
            "proper early stopping)."
        ),
    )
    seed: int = Field(default=42, description="RNG seed for the track-level split.")

    # ---------- Lightning settings ----------
    lightning: LightningConfig = Field(
        default_factory=LightningConfig,
        description="PyTorch Lightning-specific settings (used only when framework='lightning').",
    )

    # ---------- Misc ----------
    use_tqdm: bool = Field(default=True, description="Show tqdm progress bars (native path).")

    # ---------- Validators ----------

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("preprocess_cache_dir", mode="before")
    @classmethod
    def _coerce_cache_dir(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

    @model_validator(mode="after")
    def _check_cross_constraints(self) -> "PretrainConfig":
        if self.n_stack > 1 and self.framework == "lightly":
            raise ValueError(
                "framework='lightly' does not support n_stack > 1. "
                "Use framework='native' or framework='lightning'."
            )
        if self.framework == "lightly" and self.task != "simclr":
            raise ValueError(
                f"framework='lightly' is only supported for task='simclr'. Got task='{self.task}'."
            )
        return self

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
            init_settings,                       # 1st: constructor kwargs (highest)
            env_settings,                        # 2nd: PRETRAIN_* env vars
            _PretrainYamlSource(settings_cls),   # 3rd: pretrain_config.yaml
        )


# ---------------------------------------------------------------------------
# Utility loaders
# ---------------------------------------------------------------------------

def load_pretrain_config(
    yaml_path: Optional[Path | str] = None,
    **overrides: Any,
) -> PretrainConfig:
    """Load a PretrainConfig from a specific YAML file.

    Unlike ``PretrainConfig()`` (which always merges ``pretrain_config.yaml`` with env vars),
    this function loads *exclusively* from the given file, then applies ``overrides``.
    Missing fields use their Python defaults (not values from ``pretrain_config.yaml``).

    Parameters
    ----------
    yaml_path:
        Path to a YAML file.  If ``None``, uses the default ``pretrain_config.yaml``.
    **overrides:
        Additional field values that take priority over the file.
    """
    path = Path(yaml_path) if yaml_path else _DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    data.update(overrides)
    return PretrainConfig.model_validate(data)
