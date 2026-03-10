.. _hf_dataset:

Publishing dataset to Hugging Face Hub
======================================

Convert rulka frame data (``maps/img``, replays, VCP) into a Hugging Face dataset and push it to the Hub.

Prerequisites
-------------

Install the ``hf`` optional dependency::

   pip install -e ".[hf]"

This installs ``datasets`` and ``huggingface_hub``.

Pipeline
--------

**Step 1. Convert** — Build Parquet shards + replays + VCP + README + LICENSE::

   python scripts/dataset/convert_to_hf_dataset.py \
     --data-dir maps/img \
     --replays-dir maps/replays \
     --vcp-dir maps/vcp \
     --output-dir hf_dataset \
     --repo-id username/rulka-tmnf-raw-v1 \
     --val-fraction 0.1

**Step 2. Push to Hub** — Upload the dataset::

   hf auth login   # if not already logged in
   python scripts/dataset/push_to_hf.py \
     --local-path hf_dataset \
     --repo-id username/trackmania-tmnf-frames

Output structure
----------------

After conversion, ``--output-dir`` contains::

   output_dir/
   ├── data/
   │   ├── train-00000-of-00128.parquet
   │   ├── train-00001-of-00128.parquet
   │   ├── ...
   │   ├── val-00000-of-00014.parquet
   │   └── ...
   ├── replays/
   │   ├── <track_id>/
   │   │   └── <replay_name>.gbx
   │   └── ...
   ├── vcp/
   │   ├── <track_id>_0.5m_cl.npy
   │   └── ...
   ├── track_index.json
   ├── README.md
   └── LICENSE

- **data/** — Parquet shards with frames (JPEG bytes) and metadata (track_id, replay_name, step, time_ms, action_idx, inputs, etc.)
- **replays/** — Source ``.replay.gbx`` files, one per captured replay
- **vcp/** — Waypoint trajectories (one per track)
- **track_index.json** — Mapping ``track_id → {replays, has_vcp}``
- **README.md** — Dataset card with YAML frontmatter, usage examples, citation
- **LICENSE** — CC-BY-4.0

Options
-------

**convert_to_hf_dataset:**

- ``--repo-id`` — HF repo id for README examples (default: username/rulka-tmnf-raw-v1)
- ``--no-vcp`` — Do not include VCP files
- ``--symlink`` — Use symlinks instead of copying replays/VCP (saves disk space)
- ``--require-action-idx`` — Skip frames without ``action_idx`` in manifest
- ``--max-shard-size-mb 450`` — Target Parquet shard size in MB
- ``--workers N`` — Parallel workers for scan, Dataset build (num_proc), Parquet write, and copy. Default: cpu_count - 1

**push_to_hf:**

- ``--private`` — Create a private repository
- ``--num-workers N`` — Parallel upload workers (huggingface_hub). Default: cpu_count - 1

Data source
-----------

Frames and replays come from the pipeline described in :ref:`tmnf_replays`. Replays are obtained from TMNF-X (ManiaExchange). Game content © Ubisoft/Nadeo.
