# Dataset scripts for Hugging Face Hub

Scripts to convert rulka frame data to HF format and push to the Hub.

## Quick start

```bash
# 1. Convert (from project root)
python scripts/dataset/convert_to_hf_dataset.py \
  --data-dir maps/img \
  --replays-dir maps/replays \
  --vcp-dir maps/vcp \
  --output-dir data/raw/v1 \
  --repo-id your-username/rulka-tmnf-raw-v1

# 2. Push to HF
hf auth login   # if not logged in
python scripts/dataset/push_to_hf.py \
  --local-path data/raw/v1 \
  --repo-id your-username/rulka-tmnf-raw-v1
```

## Checklist before push

- [ ] Set `--repo-id` when converting (README examples will use correct URLs)
- [ ] Run `hf auth login` if not authenticated
- [ ] Create repo on HF first (optional; `push_to_hf.py` creates it with `exist_ok=True`)
- [ ] Output contains: `data/`, `replays/`, `vcp/`, `track_index.json`, `README.md`, `LICENSE`

## Output structure

```
output_dir/
├── data/
│   ├── train-*.parquet
│   └── val-*.parquet
├── replays/<track_id>/*.gbx
├── vcp/<track_id>_0.5m_cl.npy
├── track_index.json
├── README.md      # Dataset card (YAML frontmatter + usage)
└── LICENSE        # CC-BY-4.0
```
