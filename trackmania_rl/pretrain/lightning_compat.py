"""Import PyTorch Lightning from either package name.

``pip install lightning`` exposes ``import lightning as L`` (recommended).
Some environments only have ``pip install pytorch-lightning`` → ``import pytorch_lightning``.
"""

from __future__ import annotations

import sys
from typing import Any, Type

_LIGHTNING_AVAILABLE = False
LightningModule: Type[Any] = object  # type: ignore[assignment,misc]
LightningDataModule: Type[Any] = object  # type: ignore[assignment,misc]
Trainer: Any = None
Callback: Type[Any] = object  # type: ignore[assignment,misc]
ModelCheckpoint: Type[Any] = object  # type: ignore[assignment,misc]
EarlyStopping: Type[Any] = object  # type: ignore[assignment,misc]
TensorBoardLogger: Type[Any] = object  # type: ignore[assignment,misc]
CSVLogger: Type[Any] = object  # type: ignore[assignment,misc]

_IMPORT_ATTEMPTS: list[str] = []


def _record(msg: str) -> None:
    _IMPORT_ATTEMPTS.append(msg)


try:
    import lightning as _L  # type: ignore[no-redef]
    from lightning.pytorch.callbacks import Callback as _Callback
    from lightning.pytorch.callbacks import EarlyStopping as _EarlyStopping
    from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger as _CSVLogger
    from lightning.pytorch.loggers import TensorBoardLogger as _TensorBoardLogger

    LightningModule = _L.LightningModule
    LightningDataModule = _L.LightningDataModule
    Trainer = _L.Trainer
    Callback = _Callback
    ModelCheckpoint = _ModelCheckpoint
    EarlyStopping = _EarlyStopping
    TensorBoardLogger = _TensorBoardLogger
    CSVLogger = _CSVLogger
    _LIGHTNING_AVAILABLE = True
except ImportError as e:
    _record(f"import lightning / lightning.pytorch: {e!r}")
    try:
        import pytorch_lightning as _L  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import Callback as _Callback
        from pytorch_lightning.callbacks import EarlyStopping as _EarlyStopping
        from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger as _CSVLogger
        from pytorch_lightning.loggers import TensorBoardLogger as _TensorBoardLogger

        LightningModule = _L.LightningModule
        LightningDataModule = _L.LightningDataModule
        Trainer = _L.Trainer
        Callback = _Callback
        ModelCheckpoint = _ModelCheckpoint
        EarlyStopping = _EarlyStopping
        TensorBoardLogger = _TensorBoardLogger
        CSVLogger = _CSVLogger
        _LIGHTNING_AVAILABLE = True
    except ImportError as e2:
        _record(f"import pytorch_lightning: {e2!r}")

LIGHTNING_AVAILABLE = _LIGHTNING_AVAILABLE

INSTALL_HINT = (
    "Install into the same interpreter you use for scripts (see python below): "
    "  python -m pip install lightning"
    "   or   python -m pip install pytorch-lightning"
)


def lightning_import_debug_message() -> str:
    """Human-readable lines for ImportError when Lightning is missing."""
    lines = [INSTALL_HINT, f"  sys.executable = {sys.executable!r}"]
    if _IMPORT_ATTEMPTS:
        lines.append("Import attempts:")
        lines.extend(f"  - {s}" for s in _IMPORT_ATTEMPTS)
    return "\n".join(lines)
