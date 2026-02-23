"""
Level 0 artifact contract.

An encoder artifact directory contains exactly:

  encoder.pt         — torch state dict compatible with IQN_Network.img_head.
                       For n_stack=1 or stack_mode=concat: 1-channel input, direct load.
                       For stack_mode=channel (N-ch): requires kernel-averaging to 1-ch before IQN load.

  pretrain_meta.json — metadata and reproducibility record (see META_SCHEMA below).

  metrics.csv        — per-epoch loss history (optional, written when available).

pretrain_meta.json schema (version "1"):

  version            (str):   "1"
  task               (str):   "ae" | "vae" | "simclr"
  framework          (str):   "native" | "lightly" | "lightning"
  image_size         (int):   pixel size (square input assumed)
  n_stack            (int):   number of stacked consecutive frames
  stack_mode         (str):   "channel" (N-ch encoder) | "concat" (1-ch encoder + fusion)
  in_channels        (int):   input channels of saved encoder (1 for IQN-compatible)
  enc_dim            (int):   output dimension of encoder (flattened)
  normalization      (str):   "none"  (grayscale frames normalised to [0, 1])
  epochs_trained     (int):   number of completed epochs
  train_loss_final   (float): last epoch train loss
  val_loss_final     (float | null): last epoch val loss; null if no validation split
  timestamp          (str):   ISO 8601 training finish timestamp
  arch_hash          (str):   sha256[:16] of "layer_name:shape" pairs (catches architecture drift)
  dataset_root       (str):   absolute path of data_dir used for training
  n_train_samples    (int):   number of training frames seen
  n_val_samples      (int):   number of validation frames (0 if no split)
  seed               (int):   random seed used for train/val split

IQN img_head architecture reference (must match for direct weight loading):
  Conv2d(1,  16, 4x4, stride=2) → LeakyReLU
  Conv2d(16, 32, 4x4, stride=2) → LeakyReLU
  Conv2d(32, 64, 3x3, stride=2) → LeakyReLU
  Conv2d(64, 32, 3x3, stride=1) → LeakyReLU
  Flatten()

---------------------------------------------------------------------------
Preprocessed data cache contract  (preprocess.py / CachedPretrainDataset)
---------------------------------------------------------------------------

A cache directory (``preprocess_cache_dir`` in PretrainConfig) contains:

  train.npy          — numpy array, shape (N_train, n_stack, 1, H, W), dtype float32,
                       C-contiguous.  Readable via np.load(..., mmap_mode='r').
  val.npy            — same layout for validation samples.
                       Absent when val_fraction == 0.
  cache_meta.json    — JSON with the following fields used for cache validation:

    source_data_dir   (str):   resolved absolute path of the raw data_dir
    image_size        (int):   square resolution in pixels
    n_stack           (int):   temporal stack depth
    val_fraction      (float): fraction of tracks in validation split
    seed              (int):   RNG seed for the track-level split
    source_signature  (dict):  {n_tracks, n_replays, n_frame_files} counted from data_dir
    n_train           (int):   number of samples in train.npy
    n_val             (int):   number of samples in val.npy (0 when no val split)

Cache is considered valid when all six key fields match (source_data_dir,
image_size, n_stack, val_fraction, seed, source_signature).  Any change
(including adding / removing replays) causes an automatic rebuild.

The train/val split is baked into the cache at build time and does NOT need
to be recomputed during training.  track_ids are NOT stored separately.
"""

ENCODER_FILE = "encoder.pt"
META_FILE = "pretrain_meta.json"
METRICS_FILE = "metrics.csv"
META_VERSION = "1"

# IQN img_head channel progression [in, l1, l2, l3, l4]
IQN_CHANNELS = [1, 16, 32, 64, 32]
IQN_KERNELS = [(4, 4), (4, 4), (3, 3), (3, 3)]
IQN_STRIDES = [2, 2, 2, 1]

# Preprocessed cache file names (canonical; shared with preprocess.py / datasets.py)
CACHE_TRAIN_FILE = "train.npy"
CACHE_VAL_FILE = "val.npy"
CACHE_META_FILE = "cache_meta.json"

# Fields in cache_meta.json that determine cache validity
CACHE_VALIDATION_FIELDS = (
    "source_data_dir",
    "image_size",
    "n_stack",
    "val_fraction",
    "seed",
    "source_signature",
)

# ---------------------------------------------------------------------------
# BC preprocessed cache contract  (preprocess.py build_bc_cache / CachedBCDataset)
# ---------------------------------------------------------------------------
#
# A BC cache directory (``preprocess_cache_dir`` in BCPretrainConfig) contains:
#
#   train.npy           — (N_train, n_stack, 1, H, W) float32, same as Level 0.
#   train_actions.npy   — (N_train,) int64, action index per sample.
#   val.npy             — same as train.npy for validation. Absent when val_fraction == 0.
#   val_actions.npy     — (N_val,) int64. Absent when val_fraction == 0.
#   cache_meta.json     — same fields as Level 0 plus:
#     n_actions         (int): number of action classes (must match BCPretrainConfig.n_actions)
#
# Validation: is_bc_cache_valid() checks source_data_dir, image_size, n_stack,
# val_fraction, seed, source_signature, and n_actions.
#
# Reusing Level 0 cache: You can use the SAME directory for both Level 0 and
# BC. Build Level 0 first (build_cache → train.npy, val.npy, cache_meta.json).
# When you run BC with that directory as preprocess_cache_dir, build_bc_cache
# will add only train_actions.npy and val_actions.npy (same row order as
# train.npy/val.npy) and update cache_meta.json with n_actions. If any sample
# lacks action_idx in manifest.json, a full BC cache is built instead.

CACHE_TRAIN_ACTIONS_FILE = "train_actions.npy"
CACHE_VAL_ACTIONS_FILE = "val_actions.npy"
BC_CACHE_VALIDATION_FIELDS = CACHE_VALIDATION_FIELDS + ("n_actions",)
