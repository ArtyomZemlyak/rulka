"""
Datasets and data utilities for Level 0 visual pretraining.

Key improvements over the original ImageFolderFrames:

  ReplayFrameDataset  — temporal stacks are built *within a single replay directory*,
                        never crossing replay/track boundaries.  This prevents the
                        data-leakage that would occur if consecutive frames come from
                        different replays that happen to sort adjacently.

  split_track_ids     — train/val split is done at the *track level*, not at random
                        frame indices, so the same track never appears in both splits.

  ReplayFrameDataModule — Lightning DataModule wrapper (requires pip install lightning).

  CachedPretrainDataset — backed by a preprocessed .npy cache file produced by
                          ``build_cache`` (see ``preprocess.py``).  Supports
                          memory-mapped reads (low RAM) or full RAM loading.

  CachedPretrainDataModule — Lightning DataModule for the preprocessed cache;
                             train/val split is already baked into train.npy / val.npy.

Legacy-compatible FlatFrameDataset preserves the original rglob behaviour for
cases where the directory structure is not the standard <track_id>/<replay_name>/.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".npy"))


# ---------------------------------------------------------------------------
# Low-level frame loader (shared by all dataset classes)
# ---------------------------------------------------------------------------

def _load_one_frame(p: Path, size: int) -> torch.Tensor:
    """Load one image file as a grayscale tensor of shape (1, size, size) ∈ [0, 1]."""
    import cv2
    import numpy as np

    if p.suffix.lower() == ".npy":
        img = np.load(p).squeeze()
        if img.ndim == 3:
            img = img.mean(axis=-1)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((size, size), dtype=np.float32)
        else:
            img = img.astype(np.float32) / 255.0

    if img.ndim != 2:
        img = img.mean(axis=-1)

    t = torch.from_numpy(img.astype("float32")).unsqueeze(0)  # (1, H, W)
    if t.shape[1] != size or t.shape[2] != size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return t


# ---------------------------------------------------------------------------
# Train/val split helpers
# ---------------------------------------------------------------------------

def split_track_ids(
    root: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split track-ID directories into deterministic train / val lists.

    Splitting at the *track* level ensures that the same track never appears
    in both splits, which is critical for measuring generalisation.

    Returns
    -------
    (train_ids, val_ids)
    """
    root = Path(root)
    track_ids = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not track_ids:
        return [], []

    rng = random.Random(seed)
    shuffled = track_ids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction)) if val_fraction > 0 else 0
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Primary dataset: hierarchy-aware, replay-grouped temporal stacks
# ---------------------------------------------------------------------------

class ReplayFrameDataset(Dataset):
    """Dataset for maps/img/<track_id>/<replay_name>/ frame hierarchy.

    When ``n_stack > 1``, sliding windows are built **within each replay
    directory** so no temporal window crosses a replay boundary.

    Parameters
    ----------
    root:
        Root of the frame tree (e.g. ``maps/img/``).
    track_ids:
        If given, only load frames from these track sub-directories.
        Pass ``None`` to use the full tree (no train/val split).
    size:
        Target image resolution (square).
    n_stack:
        Number of consecutive frames per sample.  ``1`` = single frame.
    """

    def __init__(
        self,
        root: Path,
        track_ids: Optional[list[str]] = None,
        size: int = 64,
        n_stack: int = 1,
        image_normalization: str = "01",
    ) -> None:
        self.size = size
        self.n_stack = n_stack
        self._image_normalization = image_normalization
        root = Path(root)

        if track_ids is not None:
            track_dirs = sorted(
                root / tid for tid in track_ids if (root / tid).is_dir()
            )
        else:
            track_dirs = sorted(d for d in root.iterdir() if d.is_dir())

        # Per-replay sorted frame lists
        self._replay_frames: list[list[Path]] = []
        for track_dir in track_dirs:
            if not track_dir.is_dir():
                continue
            for replay_dir in sorted(track_dir.iterdir()):
                if not replay_dir.is_dir():
                    continue
                frames = sorted(
                    p for p in replay_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
                if len(frames) >= max(1, n_stack):
                    self._replay_frames.append(frames)

        # Flat index: (replay_group_index, start_frame_index_within_group)
        self._index: list[tuple[int, int]] = []
        for g_idx, frames in enumerate(self._replay_frames):
            n_items = len(frames) if n_stack <= 1 else len(frames) - n_stack + 1
            for start in range(n_items):
                self._index.append((g_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        g_idx, start = self._index[idx]
        frames = self._replay_frames[g_idx]
        if self.n_stack <= 1:
            t = _load_one_frame(frames[start], self.size)
        else:
            stack = [_load_one_frame(frames[start + i], self.size) for i in range(self.n_stack)]
            t = torch.stack(stack, dim=0)  # (N, 1, H, W)
        if self._image_normalization == "iqn":
            t = (t - 0.5) / 0.5
        return t

    @property
    def n_replays(self) -> int:
        return len(self._replay_frames)


# ---------------------------------------------------------------------------
# Legacy-compatible flat dataset (same behaviour as original ImageFolderFrames)
# ---------------------------------------------------------------------------

class FlatFrameDataset(Dataset):
    """Flat recursive dataset — same behaviour as the original ImageFolderFrames.

    Use this when the directory structure is not the standard hierarchy or
    when backward-compatible behaviour is required.

    Note: temporal stacks (n_stack > 1) may cross replay/track boundaries
    because the dataset is unaware of the directory structure.
    """

    def __init__(
        self,
        root: Path,
        size: int = 64,
        n_stack: int = 1,
    ) -> None:
        self.size = size
        self.n_stack = n_stack
        self.paths = sorted(
            p for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self) -> int:
        if self.n_stack <= 1:
            return len(self.paths)
        return max(0, len(self.paths) - self.n_stack + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.n_stack <= 1:
            return _load_one_frame(self.paths[idx], self.size)
        frames = [_load_one_frame(self.paths[idx + i], self.size) for i in range(self.n_stack)]
        return torch.stack(frames, dim=0)  # (N, 1, H, W)


# ---------------------------------------------------------------------------
# Cached dataset: backed by preprocessed .npy files from build_cache()
# ---------------------------------------------------------------------------

class CachedPretrainDataset(Dataset):
    """PyTorch Dataset backed by a preprocessed ``.npy`` cache file.

    The cache file is produced by ``build_cache`` (``preprocess.py``) and has
    shape ``(N, n_stack, 1, H, W)``, dtype float32. Values are in [0, 1].
    When image_normalization is "iqn", (x - 0.5) / 0.5 is applied in __getitem__.

    ``__getitem__`` returns:
      - shape ``(1, H, W)`` when ``n_stack == 1``   (matches ReplayFrameDataset)
      - shape ``(n_stack, 1, H, W)`` when ``n_stack > 1``

    Parameters
    ----------
    npy_path:
        Path to a ``.npy`` file created by ``build_cache``.
    load_in_ram:
        If ``True``, load the entire array into RAM for fast random access.
        If ``False`` (default), use memory-mapping — the OS page cache keeps
        frequently accessed pages in RAM, with minimal upfront memory cost.
    expected_image_size:
        Optional sanity-check; raises ``ValueError`` when the file's spatial
        dimensions do not match.
    expected_n_stack:
        Optional sanity-check; raises ``ValueError`` when the file's n_stack
        axis does not match.
    image_normalization:
        "01" = return as-is [0,1]; "iqn" = (x - 0.5) / 0.5 for IQN/BC transfer.
    """

    def __init__(
        self,
        npy_path: Path,
        load_in_ram: bool = False,
        expected_image_size: Optional[int] = None,
        expected_n_stack: Optional[int] = None,
        image_normalization: str = "01",
    ) -> None:
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(
                f"CachedPretrainDataset: cache file not found: {npy_path}"
            )

        mmap_mode = None if load_in_ram else "r"
        self._data: np.ndarray = np.load(str(npy_path), mmap_mode=mmap_mode)
        self._image_normalization = image_normalization

        if self._data.ndim != 5:
            raise ValueError(
                f"Expected 5-D array (N, n_stack, 1, H, W), "
                f"got shape {self._data.shape} in {npy_path}"
            )

        _, n_stack_file, _ch, h, w = self._data.shape

        if expected_n_stack is not None and n_stack_file != expected_n_stack:
            raise ValueError(
                f"n_stack mismatch: cache file has {n_stack_file}, "
                f"expected {expected_n_stack}  (file: {npy_path})"
            )
        if expected_image_size is not None and (
            h != expected_image_size or w != expected_image_size
        ):
            raise ValueError(
                f"image_size mismatch: cache file has {h}×{w}, "
                f"expected {expected_image_size}  (file: {npy_path})"
            )

        self._n_stack = n_stack_file
        self._image_size = h

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # np.array() copies the slice — necessary for mmap safety with multi-
        # process DataLoader workers (each process opens the file independently).
        sample = np.array(self._data[idx], dtype=np.float32)  # (n_stack, 1, H, W)
        t = torch.from_numpy(sample)
        if self._image_normalization == "iqn":
            t = (t - 0.5) / 0.5
        if self._n_stack == 1:
            return t.squeeze(0)  # (1, H, W) — same shape as ReplayFrameDataset
        return t  # (n_stack, 1, H, W)

    @property
    def n_stack(self) -> int:
        return self._n_stack

    @property
    def image_size(self) -> int:
        return self._image_size


# ---------------------------------------------------------------------------
# BC datasets: manifest-based (frame, action_idx)
# ---------------------------------------------------------------------------


class BCReplayDataset(Dataset):
    """Dataset for BC: (frame stack, action_idx or action_indices) from manifest.json on the fly.

    Uses _collect_bc_index from preprocess. When bc_time_offsets_ms has one offset, returns
    (tensor, int); when multiple, returns (tensor, tensor of shape (n_offsets,)).
    """

    def __init__(
        self,
        root: Path,
        track_ids: Optional[list[str]],
        size: int = 64,
        n_stack: int = 1,
        bc_target: str = "current_tick",
        bc_time_offsets_ms: Optional[list[int]] = None,
        image_normalization: str = "01",
    ) -> None:
        from trackmania_rl.pretrain.preprocess import _collect_bc_index

        root = Path(root)
        if track_ids is None:
            track_ids = sorted(d.name for d in root.iterdir() if d.is_dir())
        if bc_time_offsets_ms is None:
            bc_time_offsets_ms = [0]
        self._items = _collect_bc_index(root, track_ids, n_stack, bc_target, bc_time_offsets_ms)
        self._size = size
        self._n_stack = max(1, n_stack)
        self._image_normalization = image_normalization
        self._n_offsets = len(bc_time_offsets_ms)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor]:
        paths, action_indices = self._items[idx]
        stack = [_load_one_frame(paths[i], self._size) for i in range(self._n_stack)]
        t = torch.stack(stack, dim=0)  # (n_stack, 1, H, W)
        if self._image_normalization == "iqn":
            t = (t - 0.5) / 0.5
        if self._n_stack == 1:
            t = t.squeeze(0)
        if self._n_offsets == 1:
            return t, action_indices[0]
        return t, torch.tensor(action_indices, dtype=torch.long)


class CachedBCDataset(Dataset):
    """Dataset backed by BC cache: train.npy + train_actions.npy (or val).

    Frames in cache are stored in [0, 1]. When image_normalization is "iqn",
    we apply (x - 0.5) / 0.5 in __getitem__. Actions shape: (N,) for single offset
    or (N, n_offsets) for multi-offset. When floats_path exists, returns
    (img, float_inputs, action); else (img, action) for backward compat.
    """

    def __init__(
        self,
        npy_path: Path,
        actions_path: Path,
        floats_path: Optional[Path] = None,
        load_in_ram: bool = False,
        expected_image_size: Optional[int] = None,
        expected_n_stack: Optional[int] = None,
        image_normalization: str = "01",
    ) -> None:
        npy_path = Path(npy_path)
        actions_path = Path(actions_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"CachedBCDataset: cache not found: {npy_path}")
        if not actions_path.exists():
            raise FileNotFoundError(f"CachedBCDataset: actions not found: {actions_path}")

        mmap_mode = None if load_in_ram else "r"
        self._frames = np.load(str(npy_path), mmap_mode=mmap_mode)
        self._actions = np.load(str(actions_path), mmap_mode=mmap_mode)
        self._floats: Optional[np.ndarray] = None
        if floats_path is not None and floats_path.exists():
            self._floats = np.load(str(floats_path), mmap_mode=mmap_mode)
        self._image_normalization = image_normalization

        if self._frames.ndim != 5:
            raise ValueError(f"Expected 5-D frames (N, n_stack, 1, H, W), got {self._frames.shape}")
        if self._actions.ndim == 1:
            if len(self._actions) != len(self._frames):
                raise ValueError(f"Actions shape {self._actions.shape} does not match frames {len(self._frames)}")
            self._n_offsets = 1
        elif self._actions.ndim == 2:
            if self._actions.shape[0] != len(self._frames):
                raise ValueError(f"Actions shape {self._actions.shape} does not match frames {len(self._frames)}")
            self._n_offsets = self._actions.shape[1]
        else:
            raise ValueError(f"Expected actions 1-D or 2-D, got {self._actions.shape}")

        _, self._n_stack, _, h, w = self._frames.shape
        if expected_n_stack is not None and self._n_stack != expected_n_stack:
            raise ValueError(f"n_stack mismatch: cache={self._n_stack}, expected={expected_n_stack}")
        if expected_image_size is not None and (h != expected_image_size or w != expected_image_size):
            raise ValueError(f"image_size mismatch: cache {h}x{w}, expected {expected_image_size}")

        self._image_size = h

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int | torch.Tensor]:
        frame = np.array(self._frames[idx], dtype=np.float32)
        t = torch.from_numpy(frame)
        if self._image_normalization == "iqn":
            t = (t - 0.5) / 0.5
        if self._n_stack == 1:
            t = t.squeeze(0)
        act = int(self._actions[idx]) if self._n_offsets == 1 else torch.from_numpy(self._actions[idx].astype(np.int64))
        if self._floats is not None:
            f = torch.from_numpy(np.array(self._floats[idx], dtype=np.float32))
            return t, f, act
        return t, act

    @property
    def n_stack(self) -> int:
        return self._n_stack

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def n_offsets(self) -> int:
        return self._n_offsets


class _EmptyBCDataset(Dataset):
    """Empty dataset for BC val_dataloader when there is no validation split."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise IndexError("EmptyBCDataset has no elements")


# ---------------------------------------------------------------------------
# Lightning DataModule (optional — requires pip install lightning)
# ---------------------------------------------------------------------------

try:
    import lightning as L  # noqa: F401  (import tested here; used below)
    _LIGHTNING_AVAILABLE = True
except ImportError:
    _LIGHTNING_AVAILABLE = False

if _LIGHTNING_AVAILABLE:
    import lightning as L  # type: ignore[no-redef]

    class ReplayFrameDataModule(L.LightningDataModule):
        """Lightning DataModule wrapping ReplayFrameDataset with train/val split.

        The split is performed at the track level (see ``split_track_ids``).

        Parameters
        ----------
        data_dir:
            Root of the frame tree.
        image_size, n_stack, batch_size, workers:
            Forwarded to ReplayFrameDataset / DataLoader.
        val_fraction:
            Fraction of track IDs reserved for validation.  ``0`` disables
            validation entirely (val_dataloader returns an empty loader).
        seed:
            RNG seed for the track split.
        task:
            Used to set ``drop_last=True`` for SimCLR (required for NT-Xent).
        """

        def __init__(
            self,
            data_dir: Path,
            image_size: int = 64,
            n_stack: int = 1,
            batch_size: int = 128,
            workers: int = 4,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            val_fraction: float = 0.1,
            seed: int = 42,
            task: str = "ae",
            image_normalization: str = "01",
        ) -> None:
            super().__init__()
            self.data_dir = Path(data_dir)
            self.image_size = image_size
            self.n_stack = n_stack
            self.batch_size = batch_size
            self.workers = workers
            self.pin_memory = pin_memory
            self.prefetch_factor = prefetch_factor
            self.val_fraction = val_fraction
            self.seed = seed
            self.task = task
            self.image_normalization = image_normalization
            self._train_ids: Optional[list[str]] = None
            self._val_ids: Optional[list[str]] = None
            self.n_train_samples: int = 0
            self.n_val_samples: int = 0

        def setup(self, stage: Optional[str] = None) -> None:
            if self.val_fraction > 0:
                self._train_ids, self._val_ids = split_track_ids(
                    self.data_dir, self.val_fraction, self.seed
                )
            else:
                self._train_ids = None
                self._val_ids = None

            # Pre-compute sample counts for metadata
            train_ds = ReplayFrameDataset(self.data_dir, self._train_ids, self.image_size, self.n_stack, self.image_normalization)
            self.n_train_samples = len(train_ds)
            if self._val_ids:
                val_ds = ReplayFrameDataset(self.data_dir, self._val_ids, self.image_size, self.n_stack, self.image_normalization)
                self.n_val_samples = len(val_ds)
            else:
                self.n_val_samples = 0

        def train_dataloader(self) -> DataLoader:
            ds = ReplayFrameDataset(
                self.data_dir, self._train_ids, self.image_size, self.n_stack, self.image_normalization
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                drop_last=(self.task == "simclr"),
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        def val_dataloader(self) -> DataLoader:
            if not self._val_ids:
                # Return an empty dataset-backed loader so Lightning doesn't complain
                empty = ReplayFrameDataset(self.data_dir, [], self.image_size, self.n_stack, self.image_normalization)
                return DataLoader(empty, batch_size=self.batch_size, num_workers=0)
            ds = ReplayFrameDataset(self.data_dir, self._val_ids, self.image_size, self.n_stack, self.image_normalization)
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        @property
        def has_val(self) -> bool:
            return bool(self._val_ids)

else:
    # Stub so imports don't fail when Lightning is absent
    class ReplayFrameDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ReplayFrameDataModule requires lightning. "
                "Install it with: pip install lightning"
            )


if _LIGHTNING_AVAILABLE:

    class CachedPretrainDataModule(L.LightningDataModule):  # type: ignore[misc]
        """Lightning DataModule for preprocessed ``.npy`` cache directories.

        The train/val split is already materialised in ``train.npy`` and
        ``val.npy`` by ``build_cache``; this DataModule just wraps them.

        Parameters
        ----------
        cache_dir:
            Directory containing ``train.npy``, optionally ``val.npy``, and
            ``cache_meta.json`` as produced by ``build_cache``.
        batch_size, workers, pin_memory, prefetch_factor:
            Forwarded to DataLoader.
        task:
            Used to set ``drop_last=True`` for SimCLR (required for NT-Xent).
        load_in_ram:
            If ``True``, load cache arrays fully into RAM at DataLoader init
            time.  Speeds up random access on small datasets; avoid on
            datasets larger than available RAM.
        expected_image_size, expected_n_stack:
            Optional sanity-checks forwarded to ``CachedPretrainDataset``.
        """

        def __init__(
            self,
            cache_dir: Path,
            batch_size: int = 128,
            workers: int = 4,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            task: str = "ae",
            load_in_ram: bool = False,
            expected_image_size: Optional[int] = None,
            expected_n_stack: Optional[int] = None,
            image_normalization: str = "01",
        ) -> None:
            super().__init__()
            self.cache_dir = Path(cache_dir)
            self.batch_size = batch_size
            self.workers = workers
            self.pin_memory = pin_memory
            self.prefetch_factor = prefetch_factor
            self.task = task
            self.load_in_ram = load_in_ram
            self.expected_image_size = expected_image_size
            self.expected_n_stack = expected_n_stack
            self.image_normalization = image_normalization
            self.n_train_samples: int = 0
            self.n_val_samples: int = 0
            self._has_val: bool = False

        def setup(self, stage: Optional[str] = None) -> None:
            from trackmania_rl.pretrain.preprocess import (
                CACHE_TRAIN_FILE, CACHE_VAL_FILE,
            )
            train_ds = CachedPretrainDataset(
                self.cache_dir / CACHE_TRAIN_FILE,
                load_in_ram=self.load_in_ram,
                expected_image_size=self.expected_image_size,
                expected_n_stack=self.expected_n_stack,
                image_normalization=self.image_normalization,
            )
            self.n_train_samples = len(train_ds)

            val_path = self.cache_dir / CACHE_VAL_FILE
            self._has_val = val_path.exists() and len(
                CachedPretrainDataset(val_path, load_in_ram=False, image_normalization=self.image_normalization)
            ) > 0
            if self._has_val:
                val_ds = CachedPretrainDataset(
                    val_path,
                    load_in_ram=self.load_in_ram,
                    expected_image_size=self.expected_image_size,
                    expected_n_stack=self.expected_n_stack,
                    image_normalization=self.image_normalization,
                )
                self.n_val_samples = len(val_ds)

        def _make_ds(self, split: str) -> CachedPretrainDataset:
            from trackmania_rl.pretrain.preprocess import (
                CACHE_TRAIN_FILE, CACHE_VAL_FILE,
            )
            fname = CACHE_TRAIN_FILE if split == "train" else CACHE_VAL_FILE
            return CachedPretrainDataset(
                self.cache_dir / fname,
                load_in_ram=self.load_in_ram,
                expected_image_size=self.expected_image_size,
                expected_n_stack=self.expected_n_stack,
                image_normalization=self.image_normalization,
            )

        def train_dataloader(self) -> DataLoader:
            ds = self._make_ds("train")
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                drop_last=(self.task == "simclr"),
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        def val_dataloader(self) -> DataLoader:
            from trackmania_rl.pretrain.preprocess import CACHE_VAL_FILE
            val_path = self.cache_dir / CACHE_VAL_FILE
            if not self._has_val or not val_path.exists():
                # Empty loader so Lightning doesn't complain
                empty_ds = CachedPretrainDataset.__new__(CachedPretrainDataset)
                empty_ds._data = np.zeros(
                    (0, max(1, self.expected_n_stack or 1), 1,
                     self.expected_image_size or 64,
                     self.expected_image_size or 64),
                    dtype=np.float32,
                )
                empty_ds._n_stack = self.expected_n_stack or 1
                empty_ds._image_size = self.expected_image_size or 64
                return DataLoader(empty_ds, batch_size=self.batch_size, num_workers=0)
            ds = self._make_ds("val")
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        @property
        def has_val(self) -> bool:
            return self._has_val

    # -----------------------------------------------------------------------
    # BC Lightning DataModules (Level 1)
    # -----------------------------------------------------------------------

    class CachedBCDataModule(L.LightningDataModule):  # type: ignore[misc]
        """Lightning DataModule for BC preprocessed cache (train.npy + train_actions.npy, etc.)."""

        def __init__(
            self,
            cache_dir: Path,
            batch_size: int = 128,
            workers: int = 4,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            load_in_ram: bool = False,
            expected_image_size: Optional[int] = None,
            expected_n_stack: Optional[int] = None,
            image_normalization: str = "01",
        ) -> None:
            super().__init__()
            from trackmania_rl.pretrain.contract import (
                CACHE_TRAIN_FILE,
                CACHE_TRAIN_ACTIONS_FILE,
                CACHE_TRAIN_FLOATS_FILE,
                CACHE_VAL_FILE,
                CACHE_VAL_ACTIONS_FILE,
                CACHE_VAL_FLOATS_FILE,
            )
            self.cache_dir = Path(cache_dir)
            self.batch_size = batch_size
            self.workers = workers
            self.pin_memory = pin_memory
            self.prefetch_factor = prefetch_factor
            self.load_in_ram = load_in_ram
            self.expected_image_size = expected_image_size
            self.expected_n_stack = expected_n_stack
            self.image_normalization = image_normalization
            self._train_file = CACHE_TRAIN_FILE
            self._train_actions_file = CACHE_TRAIN_ACTIONS_FILE
            self._val_file = CACHE_VAL_FILE
            self._val_actions_file = CACHE_VAL_ACTIONS_FILE
            self.n_train_samples: int = 0
            self.n_val_samples: int = 0
            self._has_val: bool = False
            self._floats_path_train: Optional[Path] = None
            self._floats_path_val: Optional[Path] = None
            self.bc_time_offsets_ms: list[int] = [0]
            self.bc_offset_weights: Optional[list[float]] = None
            self.n_offsets: int = 1

        def setup(self, stage: Optional[str] = None) -> None:
            import json
            meta_path = self.cache_dir / "cache_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        meta = json.load(fh)
                    self.bc_time_offsets_ms = meta.get("bc_time_offsets_ms", [0])
                    self.bc_offset_weights = meta.get("bc_offset_weights")
                    self.n_offsets = len(self.bc_time_offsets_ms)
                    if meta.get("has_floats"):
                        tp = self.cache_dir / CACHE_TRAIN_FLOATS_FILE
                        vp = self.cache_dir / CACHE_VAL_FLOATS_FILE
                        if tp.exists():
                            self._floats_path_train = tp
                        if vp.exists():
                            self._floats_path_val = vp
                except (OSError, json.JSONDecodeError):
                    pass
            train_ds = CachedBCDataset(
                self.cache_dir / self._train_file,
                self.cache_dir / self._train_actions_file,
                floats_path=self._floats_path_train,
                load_in_ram=self.load_in_ram,
                expected_image_size=self.expected_image_size,
                expected_n_stack=self.expected_n_stack,
                image_normalization=self.image_normalization,
            )
            self.n_train_samples = len(train_ds)
            val_frames = self.cache_dir / self._val_file
            val_actions = self.cache_dir / self._val_actions_file
            self._has_val = val_frames.exists() and val_actions.exists()
            if self._has_val:
                val_ds = CachedBCDataset(
                    val_frames,
                    val_actions,
                    floats_path=self._floats_path_val,
                    load_in_ram=self.load_in_ram,
                    expected_image_size=self.expected_image_size,
                    expected_n_stack=self.expected_n_stack,
                    image_normalization=self.image_normalization,
                )
                self.n_val_samples = len(val_ds)

        def train_dataloader(self) -> DataLoader:
            ds = CachedBCDataset(
                self.cache_dir / self._train_file,
                self.cache_dir / self._train_actions_file,
                floats_path=self._floats_path_train,
                load_in_ram=self.load_in_ram,
                expected_image_size=self.expected_image_size,
                expected_n_stack=self.expected_n_stack,
                image_normalization=self.image_normalization,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                drop_last=True,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        def val_dataloader(self) -> DataLoader:
            if not self._has_val:
                empty_ds = _EmptyBCDataset()
                return DataLoader(empty_ds, batch_size=self.batch_size, num_workers=0)
            ds = CachedBCDataset(
                self.cache_dir / self._val_file,
                self.cache_dir / self._val_actions_file,
                floats_path=self._floats_path_val,
                load_in_ram=self.load_in_ram,
                expected_image_size=self.expected_image_size,
                expected_n_stack=self.expected_n_stack,
                image_normalization=self.image_normalization,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        @property
        def has_val(self) -> bool:
            return self._has_val

    class BCReplayDataModule(L.LightningDataModule):  # type: ignore[misc]
        """Lightning DataModule for BC from replay dirs (split_track_ids + BCReplayDataset)."""

        def __init__(
            self,
            data_dir: Path,
            image_size: int = 64,
            n_stack: int = 1,
            batch_size: int = 128,
            workers: int = 4,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            val_fraction: float = 0.1,
            seed: int = 42,
            bc_target: str = "current_tick",
            bc_time_offsets_ms: Optional[list[int]] = None,
            bc_offset_weights: Optional[list[float]] = None,
            image_normalization: str = "01",
        ) -> None:
            super().__init__()
            self.data_dir = Path(data_dir)
            self.image_size = image_size
            self.n_stack = n_stack
            self.batch_size = batch_size
            self.workers = workers
            self.pin_memory = pin_memory
            self.prefetch_factor = prefetch_factor
            self.val_fraction = val_fraction
            self.seed = seed
            self.bc_target = bc_target
            self.bc_time_offsets_ms = bc_time_offsets_ms if bc_time_offsets_ms is not None else [0]
            self.bc_offset_weights = bc_offset_weights
            self.n_offsets = len(self.bc_time_offsets_ms)
            self.image_normalization = image_normalization
            self._train_ids: Optional[list[str]] = None
            self._val_ids: Optional[list[str]] = None
            self.n_train_samples: int = 0
            self.n_val_samples: int = 0

        def setup(self, stage: Optional[str] = None) -> None:
            if self.val_fraction > 0:
                self._train_ids, self._val_ids = split_track_ids(
                    self.data_dir, self.val_fraction, self.seed
                )
            else:
                self._train_ids = sorted(d.name for d in self.data_dir.iterdir() if d.is_dir())
                self._val_ids = []
            train_ds = BCReplayDataset(
                self.data_dir, self._train_ids, size=self.image_size, n_stack=self.n_stack,
                bc_target=self.bc_target, bc_time_offsets_ms=self.bc_time_offsets_ms,
                image_normalization=self.image_normalization,
            )
            self.n_train_samples = len(train_ds)
            if self._val_ids:
                val_ds = BCReplayDataset(
                    self.data_dir, self._val_ids, size=self.image_size, n_stack=self.n_stack,
                    bc_target=self.bc_target, bc_time_offsets_ms=self.bc_time_offsets_ms,
                    image_normalization=self.image_normalization,
                )
                self.n_val_samples = len(val_ds)
            else:
                self.n_val_samples = 0

        def train_dataloader(self) -> DataLoader:
            ds = BCReplayDataset(
                self.data_dir, self._train_ids, size=self.image_size, n_stack=self.n_stack,
                bc_target=self.bc_target, bc_time_offsets_ms=self.bc_time_offsets_ms,
                image_normalization=self.image_normalization,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                drop_last=True,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        def val_dataloader(self) -> DataLoader:
            if not self._val_ids:
                empty = BCReplayDataset(
                    self.data_dir, [], size=self.image_size, n_stack=self.n_stack,
                    bc_target=self.bc_target, bc_time_offsets_ms=self.bc_time_offsets_ms,
                    image_normalization=self.image_normalization,
                )
                return DataLoader(empty, batch_size=self.batch_size, num_workers=0)
            ds = BCReplayDataset(
                self.data_dir, self._val_ids, size=self.image_size, n_stack=self.n_stack,
                bc_target=self.bc_target, bc_time_offsets_ms=self.bc_time_offsets_ms,
                image_normalization=self.image_normalization,
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.pin_memory,
                persistent_workers=(self.workers > 0),
                prefetch_factor=self.prefetch_factor if self.workers > 0 else None,
            )

        @property
        def has_val(self) -> bool:
            return bool(self._val_ids)

else:
    class CachedPretrainDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CachedPretrainDataModule requires lightning. "
                "Install it with: pip install lightning"
            )

    class CachedBCDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CachedBCDataModule requires lightning. Install it with: pip install lightning"
            )

    class BCReplayDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BCReplayDataModule requires lightning. Install it with: pip install lightning"
            )
