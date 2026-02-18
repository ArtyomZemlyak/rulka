.. _pretrain_replay_roadmap:

Pretrain on replays: roadmap and integration with IQN
=====================================================

This page systematizes **pretraining and imitation-learning options** for training on TrackMania replays (frames + actions), from the simplest to the most advanced. Each option is described with **how it plugs into the current IQN training pipeline**.

**Current pipeline (no pretrain):** Collectors produce transitions (frame stacks, float features, actions) → replay buffer → learner trains IQN (CNN ``img_head`` + float MLP + quantile heads) with TD loss. Checkpoints: ``weights1.torch`` / ``weights2.torch``; no built-in “load only backbone” — integration is via loading pretrained weights into the full network before saving the first checkpoint or at startup.

**Data source:** Frames and inputs come from :ref:`tmnf_replays` (TMInterface capture at e.g. 64 FPS; ``manifest.json`` has ``inputs`` and state per frame). Use ``maps/img/<track_id>/<replay_name>/`` as the dataset root.

---

Track representation for RL (context for pretrain)
--------------------------------------------------

How the **track** (and optionally the **trajectory**) is represented determines what the agent can learn (e.g. shortcuts, jumps). Below is a compact summary of options; pretrain and BC can use one or several of these as inputs or auxiliary targets.

**1. Checkpoint system (common in tmrl and similar)**  
Demo trajectory is split into equally spaced points; reward = number of checkpoints passed. Simple, no explicit 3D; but depends on demo quality and does not explicitly encode geometry.

**2. LIDAR-style**  
Rays from the car to track boundaries (e.g. 19 rays), often with a short history (e.g. shape (4, 19)). Can be derived from screenshots via raycasting. Good for local geometry; needs temporal model (LSTM/stack) for dynamics.

**3. Images (CNN-based)**  
Screenshots 84×84 or 128×128, stack of 4 frames; optional preprocessing (Canny, blur). What we use now in IQN. No explicit 3D; 3D and jumps are implicit in the image.

**4. Coordinate / API state**  
TMInterface (Nations Forever): position, orientation (yaw/pitch/roll), speed. OpenPlanet (Trackmania 2020): full physics. Useful for reward and for auxiliary inputs; not a replacement for pixels if we want visual policy.

**5. 3D representation (for jumps and shortcuts)**  
Tracks are 3D; top replays often “cut” by jumping between segments. To let the model learn such maneuvers:

- **Track mesh / 3D geometry** — explicit 3D surface of the track.
- **Volumetric checkpoints** — 3D gates or spheres instead of 2D points; multiple valid trajectories (including air).
- **Segment graph with jump edges** — nodes = segments (straight/turn/jump), edges = connectivity; special edges for “jump from A to B” with attributes (takeoff/landing, required_speed, time_saved). Enables reward and planning for shortcuts.
- **Air corridors** — 3D regions for flight; reward inside corridor, penalty outside. Turns implicit cut from replays into an explicit learning signal.

**6. Track / trajectory embeddings (compact representation)**  
Instead of feeding millions of 3D points, encode the track or trajectory into a **fixed-size vector** (e.g. 128–512 dim) and feed that to the policy or use it for reward/planning:

- **SeCTAR (trajectory VAE)** — full trajectory encoded to latent z (e.g. 32–128 dim); decoder reconstructs state sequence. Enables planning in latent space.
- **Track-agnostic embedding** — one CNN/GNN encoder over track representation (e.g. graph or mini-map); same model works across tracks; vector = “which track + where”.
- **Graph-based track embedding** — track as graph (nodes = key points/segments, edges = segments + jump edges); GNN encodes it to a vector. Fits 3D and jumps (same graph as in 3D representation above).
- **Point-ROPE** — positional embeddings for 3D coordinates (e.g. rotary embeddings); preserves relative geometry; can be combined with graph embedding for 3D jumps.

**When to use what (from KB):**  
BC: images + action history. RL from scratch: checkpoint reward + API coordinates. For **jumps and cuts**: 3D representation (segment graph, air corridors, volumetric checkpoints). For **compact global context** and transfer across tracks: track/trajectory embedding (pretrain encoder on many tracks, then freeze or finetune for RL).

---

Key insights (from knowledge base)
----------------------------------

1. **Compounding errors in racing**
   BC (behavioral cloning) trained on expert (replay) distribution suffers from **compounding errors**: small mistakes lead to states the policy never saw, so errors grow roughly as **O(T²ε)** over trajectory length T. In TrackMania, a mistake at 50 s can invalidate the rest of the lap. **DAGGER** reduces this to O(Tε) by iteratively querying an expert on *rollouts of the current policy*, but requires an “expert” (e.g. human or strong bot) to label new states. **Hybrid BC + RL** avoids needing a live expert: BC gives a safe initial policy; RL (e.g. PPO or continued IQN) improves it via environment interaction.

2. **Latent Action Model (Genie-style)**
   Genie (arXiv:2402.15391) uses a **VQ-VAE–based latent action model** to infer discrete latent actions from **video only** (no action labels). For TrackMania this allows learning from **YouTube or other videos** without key logs: train LAM on unlabeled video, then map latent actions to real controls with a small labeled set (e.g. ~200 expert replays with ``inputs`` in ``manifest.json``).

3. **BC + PPO (or BC + IQN)**
   Recommended pipeline: **(1)** BC pretraining on 50+ hours of replays to get a good initial policy; **(2)** RL fine-tuning (PPO or continued IQN) with reward = progress + penalty for resets; **(3)** optional **combined loss** L = L_RL + λ·L_BC to keep behavior close to expert while improving. BC solves “where to start”; RL solves exploration and long-horizon credit assignment.

4. **Architecture**
   Minimal temporal setup: **stack of 4 frames (e.g. 84×84×4)** plus **LSTM(256)** over last 10 action steps. For 64 FPS and full laps (30–60 s = 1920–3840 frames), **Transformer with causal attention** can model long dependencies better than a short LSTM.

5. **Multimodal actions**
   Same state (e.g. entering a turn) can have multiple good actions (early vs late apex). **BCE / MSE on a single head** averages modes and can degrade. Alternatives: **Mixture of Experts**, **Conditional VAE**, or **quantile regression** for continuous actions (steer, brake).

---

Simplest experiments to run first (minimal setup)
-------------------------------------------------

These are the **smallest, highest-signal experiments** you can do without relying on the full roadmap. Each has a single variable and a clear metric. Order is from “no code” to “one new script”.

1. **Baseline: pretrain vs no pretrain**
   - **What:** Two identical IQN runs (same config, same maps, same duration). Run A: start from scratch. Run B: load ``encoder.pt`` from ``pretrain_visual_backbone.py`` into ``img_head``, then start learner from that checkpoint.
   - **Metric:** Same wall-clock or same number of env steps → compare eval race time / finish rate. If B is better, pretrain is worth it.
   - **Why first:** Zero new code; only need to produce one ``encoder.pt`` and load it once into a full IQN checkpoint.

2. **Amount of pretrain data**
   - **What:** Same pretrain task (e.g. AE on frames). Three runs: pretrain on **1 h**, **5 h**, **20 h** of replay frames (subsample dirs or limit by track count). Same epochs and batch size. Then same IQN from each encoder.
   - **Metric:** IQN performance (e.g. best A01 at 30 min) vs “hours of pretrain data”. Expect a saturating curve; tells you how much replay data is enough.
   - **Why simple:** Only variable is dataset size; no new methods.

3. **Pretrain task: AE vs SimCLR**
   - **What:** Same frames, same encoder architecture. Run 1: autoencoder (reconstruction). Run 2: SimCLR (contrastive, already in ``pretrain_visual_backbone.py`` with ``--task simclr --framework lightly``). Load each encoder into IQN (same way), same RL config.
   - **Metric:** Which encoder gives faster/better IQN? Literature often favors contrastive for control; one experiment checks it for TrackMania.
   - **Why simple:** Both paths exist in the repo; only compare two checkpoints.

4. **Same-domain vs out-of-domain pretrain**
   - **What:** Pretrain A: only TrackMania frames (from ``maps/img``). Pretrain B: generic images (e.g. ImageNet grayscale, or another game). Same architecture and training recipe. Then IQN on TrackMania from A and from B.
   - **Metric:** IQN performance. Expect A ≥ B; size of the gap shows how much domain matters.
   - **Why simple:** No new algorithm; only data source changes.

5. **Pretrain epochs: underfitting vs overfitting**
   - **What:** Fix data and task (e.g. AE, 5 h of frames). Pretrain with **10**, **50**, **200** epochs. Same IQN from each encoder.
   - **Metric:** IQN performance vs pretrain epochs. Often there is a sweet spot; too many epochs can overfit to pretrain distribution.
   - **Why simple:** Single knob; easy to plot.

6. **Frozen vs fine-tuned backbone**
   - **What:** Load ``encoder.pt`` into IQN. Run A: **freeze** ``img_head`` for the first N steps (e.g. 50k–100k), then unfreeze. Run B: **never freeze**. Same config otherwise.
   - **Metric:** IQN sample efficiency and final performance. Common in transfer: short freeze can help stability; long freeze can cap performance.
   - **Why simple:** One flag or short branch in the learner; no new data or scripts.

7. **BC with minimal data (sanity check)**
   - **What:** Implement the simplest BC: frames → one discrete action (e.g. 9 classes: steer ∈ {−1, 0, +1} × accel ∈ {0, 1}). Train on **one track**, **5 replays** only. Measure **validation accuracy** (or accuracy on a held-out replay).
   - **Metric:** If accuracy >> random (e.g. >> 11%), the signal is there; then you can scale to more data. If not, check labels or architecture.
   - **Why simple:** Proves that “frame → action” is learnable from very little data before investing in full BC pipeline.

8. **Single-track pretrain vs multi-track**
   - **What:** Pretrain (AE or BC) on frames from **one track only** vs from **many tracks**. Then run IQN on (a) the same track, (b) a different track.
   - **Metric:** Same-track vs different-track performance. Single-track pretrain often helps same track and may hurt others (overfitting to one track); multi-track should generalize better.
   - **Why simple:** Only varies “which tracks in pretrain”; no new methods.

9. **Frame stack at pretrain: 1 vs 4**
   - **What:** Pretrain with **1 frame** (single image) vs **4 stacked frames** (temporal). Use ``--n-stack 4 --stack-mode concat`` for 4-frame; save 1-ch encoder. Load into IQN (which uses 4-frame stack: copy same encoder for each channel or use one channel and replicate). Same IQN config.
   - **Metric:** Does 4-frame pretrain give better IQN than 1-frame? Tests whether temporal pretrain helps.
   - **Why simple:** Same script, different ``--n-stack``; one extra run for “1 frame”.

10. **BC action space: discrete first**
    - **What:** For the first BC version, use **discrete** actions (e.g. 9 or 27 bins: steer × accel × brake binned) and cross-entropy loss instead of continuous regression. Easier to train and debug.
    - **Metric:** Validation accuracy and later IQN transfer. If discrete BC works, add continuous later.
    - **Why simple:** Fewer moving parts than continuous + MSE; same data and encoder.

**Suggested order:** 1 → 3 → 5 → 6 (all with existing pretrain script and IQN). Then 2 and 4 (data scale and domain). Then 7 and 8 when you add BC; 9 and 10 when you touch frame stack or action space.

**Experiments involving track representation (after basics):**

11. **Observation: image-only vs image + progress**
    - **What:** Same IQN, but in one run add a **scalar progress** (e.g. checkpoint index or normalized distance along demo trajectory from API) to the float inputs. In the other run use only images + other floats (speed, etc.).
    - **Metric:** Sample efficiency and final time. Progress can act as a dense reward proxy and help credit assignment; compare with checkpoint-based reward if you add it.
    - **Why simple:** One extra float in the observation; no new pretrain.

12. **Track embedding: same track vs new track**
    - **What:** Once you have a **track (or trajectory) encoder** (Level 7 below), pretrain it on many tracks. Then run IQN with frozen track embedding on (a) a track seen in pretrain, (b) a new track. Measures transfer.
    - **Metric:** IQN performance on seen vs unseen track; ablation with/without track embedding.
    - **Why later:** Requires implementing the embedding encoder first.

---

Roadmap: experiments from simplest to most complex
--------------------------------------------------

Experiments are ordered by **implementation and data complexity**. Later steps assume you have (or can add) the corresponding code/scripts; “Integration with IQN” explains how each fits the current learner/collector setup.

**Your pretrain ideas (mapped to roadmap)**

- **Idea 1: Простой претрейн картиночного энкодера** — corresponds to **Level 0** (unsupervised visual pretraining): AE/VAE/SimCLR on frames, save encoder, load into ``img_head``. Already in repo.
- **Idea 2: Сжатие данных трассы в вектор и восстановление траектории из реплея** — corresponds to **Level 7** (track/trajectory embedding): encode track or full trajectory into a fixed-size vector (e.g. VAE or GNN on segment graph), train to reconstruct trajectory from replay; use encoder as compact track/trajectory input for IQN or for reward.
- **Idea 3: Вектор текущего состояния машины + позиция на трассе → предсказание оставшейся части траектории из реплея** — corresponds to **Level 8** (trajectory completion): input = (current state, position on track or progress); target = remaining trajectory (positions/actions) from expert replay. Trained as auxiliary head or separate model; can be used for planning or as behavioral prior (e.g. “how would expert continue?”).

**Level 0: Unsupervised visual pretraining**

  - **What:** Pretrain only the **CNN backbone** (IQN ``img_head``) on frames—no actions.
    Tasks: autoencoder (AE), VAE, SimCLR (contrastive).

  - **Package:** ``trackmania_rl/pretrain_visual/`` — modular package with ``PretrainConfig``,
    replay-aware dataset (no cross-replay temporal stacking), Lightning + native training
    paths, and reproducible artifact export.

  - **Scripts:**

    - ``scripts/pretrain_visual_backbone.py`` — train the encoder.
    - ``scripts/init_iqn_from_encoder.py`` — inject encoder into IQN checkpoint.

  - **Configuration:** ``config_files/pretrain_config.yaml`` — YAML file with all defaults.
    Loaded via ``PretrainConfig(BaseSettings)`` (``config_files/pretrain_schema.py``).
    Override priority:

    1. CLI arguments (highest)
    2. Env vars: ``PRETRAIN_<FIELD>``  e.g. ``PRETRAIN_TASK=simclr PRETRAIN_EPOCHS=100``
    3. ``config_files/pretrain_config.yaml``
    4. Field defaults

  - **Framework:** PyTorch Lightning (default; ``framework: lightning`` in
    ``pretrain_config.yaml``). Provides AMP, gradient clipping, early stopping,
    TensorBoard, CSV logger, and best-checkpoint saving out of the box.
    ``native`` and ``lightly`` back-ends remain available via ``--framework``
    for debugging or minimal-dependency setups.

  - **Artifact contract:** every run creates a versioned subdirectory
    ``output_dir/run_NNN/`` (or ``output_dir/<run_name>/``) containing:

    - ``encoder.pt`` — CNN weights only; IQN-compatible ``img_head`` architecture.
    - ``pretrain_meta.json`` — task, image_size, n_stack, stack_mode, in_channels,
      enc_dim, epochs, train/val loss, dataset path, arch_hash, timestamp.
    - ``metrics.csv`` — per-epoch loss history.
    - ``tensorboard/`` — TensorBoard event files.
    - ``csv/`` — Lightning CSV logger output.
    - ``checkpoints/`` — ``best-epoch=NNN.ckpt`` snapshots for resuming training
      (not consumed by ``init_iqn_from_encoder.py``; use ``encoder.pt`` there).

    See ``trackmania_rl/pretrain_visual/contract.py`` for the full schema.

  - **Dataset split:** ``--val-fraction 0.1`` enables track-level train/val split
    (no data leakage). Default ``0`` = no split (original behaviour).

  - **Temporal stacking:** ``ReplayFrameDataset`` enforces within-replay temporal
    windows so no sliding window crosses a replay/track boundary.

  - **Preprocessed data cache (optional, recommended for repeated runs):**
    Raw image decoding (JPEG → grayscale → resize) is the typical CPU bottleneck
    during pretrain I/O.  The cache pipeline pre-processes all frames once and
    stores them as a memory-mappable NumPy array (``train.npy`` / ``val.npy``),
    reducing per-epoch I/O to fast sequential reads from a single large file.

    **Activation:** set ``preprocess_cache_dir`` in ``pretrain_config.yaml``
    (or via ``--preprocess-cache-dir`` CLI arg).  The training script
    automatically validates the cache and rebuilds it when stale.

    **Cache layout** (written to ``preprocess_cache_dir/``)::

      train.npy         (N_train, n_stack, 1, H, W) float32 — memory-mappable
      val.npy           (N_val,   n_stack, 1, H, W) float32 — absent when val_fraction=0
      cache_meta.json   parameters + source fingerprint for validity checks

    **Cache is invalidated (rebuilt) when any of these change:** ``image_size``,
    ``n_stack``, ``val_fraction``, ``seed``, ``data_dir`` path, or the
    source fingerprint (``n_tracks``, ``n_replays``, ``n_frame_files`` in
    ``data_dir``).  Adding or removing replay directories triggers a rebuild.

    **Manual pre-warming** (optional — useful when training will run on a
    machine with slower disk access):

    .. code-block:: bash

       python scripts/prepare_pretrain_data.py \
           --data-dir maps/img \
           --output-dir cache/pretrain_64 \
           --image-size 64 --n-stack 1 \
           --val-fraction 0.1 --seed 42

       # Then in pretrain_config.yaml:
       #   preprocess_cache_dir: cache/pretrain_64

    **RAM loading** (for small datasets that fit in memory): set
    ``cache_load_in_ram: true`` in ``pretrain_config.yaml`` to load the
    arrays fully into RAM at startup instead of memory-mapping.

  **Quick start:**

  .. code-block:: bash

     # All settings come from config_files/pretrain_config.yaml.
     # framework: lightning is the default — no extra flags needed.

     # Step 1: pretrain AE (auto-creates output/ptretrain/vis/run_001/)
     python scripts/pretrain_visual_backbone.py --data-dir maps/img

     # Step 1 alt: SimCLR with track-level val split
     python scripts/pretrain_visual_backbone.py \
         --data-dir maps/img --task simclr --val-fraction 0.1

     # Step 1 alt: label the run for easy reference
     python scripts/pretrain_visual_backbone.py \
         --data-dir maps/img --task ae --run-name ae_v1

     # Output structure (every run):
     #   output/ptretrain/vis/run_001/
     #       encoder.pt            ← CNN weights only → init_iqn_from_encoder.py
     #       pretrain_meta.json    ← full reproducibility record
     #       metrics.csv           ← per-epoch loss history
     #       tensorboard/          ← TensorBoard event files
     #       csv/                  ← CSV metrics log
     #       checkpoints/          ← best-epoch=NNN.ckpt  (resume only, not for IQN)
     #
     # NOTE: encoder.pt ≠ .ckpt
     #   .ckpt  — full Lightning snapshot (encoder + decoder + optimizer) for resuming
     #   encoder.pt — extracted CNN weights only; this is what goes into IQN

     # Step 2: inject encoder into IQN (writes weights1.torch + weights2.torch)
     python scripts/init_iqn_from_encoder.py --encoder-pt output/ptretrain/vis/v1/encoder.pt --save-dir   output/ptretrain/vis/v1/

     # Step 3: start IQN training (learner auto-loads checkpoint)
     python scripts/train.py

     # Use a custom YAML (replaces pretrain_config.yaml):
     python scripts/pretrain_visual_backbone.py --config my_experiment.yaml

     # Override individual fields via env vars (PowerShell):
     $env:PRETRAIN_TASK = "simclr"; $env:PRETRAIN_EPOCHS = "100"
     python scripts/pretrain_visual_backbone.py --data-dir maps/img

     # Validate artifact compatibility without writing any files:
     python scripts/init_iqn_from_encoder.py \
         --encoder-pt output/ptretrain/vis/run_001/encoder.pt --dry-run

  - **Integration with IQN:** ``init_iqn_from_encoder.py`` creates a fresh (or patches
    an existing) IQN network pair, injects ``encoder.pt`` into ``img_head`` of both
    online and target networks, and saves ``weights1.torch``/``weights2.torch``.
    The learner picks these up on startup—no training loop change required.
    For multi-channel encoders (``--stack-mode channel``), the script automatically
    averages first Conv2d kernels to produce a 1-channel weight.

  **Level 0 experiment matrix (minimum viable):**

  +-----------+----------------------------+-------------------------------------------+
  | Run label | Pretrain                   | Note                                      |
  +===========+============================+===========================================+
  | A_scratch | None (IQN from scratch)    | Baseline; keep RL config identical.       |
  +-----------+----------------------------+-------------------------------------------+
  | B1_ae     | AE → IQN              | ``--task ae``                             |
  +-----------+----------------------------+-------------------------------------------+
  | B2_simclr | SimCLR → IQN          | ``--task simclr``                         |
  +-----------+----------------------------+-------------------------------------------+

  **KPIs (record at fixed wall-clock intervals, e.g. 30 min and 60 min):**

  - *Primary:* time to first finish, best eval race time at fixed budget, finish rate.
  - *Secondary:* training loss spread, gradient norms, 2–3 seeds for robustness.

  Use ``scripts/analyze_experiment_by_relative_time.py A_scratch B1_ae B2_simclr``
  to compare the three runs.
**Level 1: Behavioral cloning (BC) — frames → actions**
  - **What:** Supervised learning: input = frame stack (and optionally float features), target = expert action from ``manifest.json`` (e.g. steer/accel/brake or discrete action id). Single policy network (e.g. same CNN as ``img_head`` + action head); loss = cross-entropy (discrete) or MSE (continuous).
  - **Data:** Replays from ``capture_replays_tmnf.py``; read ``manifest.json`` per frame for ``inputs``; align frames ``frame_*_*ms.jpeg`` with inputs (same ``step``/``time_ms``).
  - **Integration with IQN:** (A) **Backbone only:** train BC, save the **encoder** (CNN part); load into ``IQN_Network.img_head`` as in Level 0; then run IQN from that checkpoint. (B) **Full policy as prior:** if the BC network has the same architecture as IQN’s “feature” part (img_head + float head → joint embedding), load both into IQN and optionally use a **warm start**: fill replay buffer with BC policy rollouts, then train IQN with a short BC loss coef (e.g. L = L_IQN + 0.1·L_BC) for a few k steps before dropping L_BC.

**Level 2: BC + temporal model (LSTM / history of actions)**
  - **What:** Same as Level 1 but input includes **last K actions** (or latent states); network = CNN + LSTM(256) or 1D conv over time; predicts next action. Captures “how we got here” and reduces compounding errors somewhat by conditioning on recent behavior.
  - **Data:** Same as Level 1; build sequences from consecutive frames + ``inputs``.
  - **Integration with IQN:** IQN currently has no LSTM. Options: (A) Use the **pretrained CNN** from this BC model as ``img_head`` (discard LSTM); or (B) **Extend IQN** to have an LSTM between ``img_head`` output and the rest (larger change); then load pretrained CNN (and optionally LSTM) and continue IQN training.

**Level 3: DAGGER (iterative BC with expert relabeling)**
  - **What:** Train BC → run current policy in env (or in TMInterface with script from policy) → get new states → **query expert** for actions on those states → add (s, a_expert) to dataset → retrain BC. Repeat. Reduces distribution shift and O(T²ε) → O(Tε).
  - **Requirement:** An “expert” that can return actions for arbitrary states: e.g. human, or a strong bot (e.g. script that replays a given replay’s inputs), or a hybrid (human in the loop).
  - **Integration with IQN:** Use DAGGER to produce a **better BC policy**; then use that policy’s **encoder (and optionally full policy)** as in Level 1/2 to initialize IQN or to fill the buffer and warm-start with L_BC. No change to IQN algorithm; only to how the initial weights and/or buffer are produced.

**Level 4: BC + RL fine-tuning (IQN or PPO)**
  - **What:** Pretrain with BC on 50+ hours of replays → then run **IQN** (current pipeline) or **PPO** with reward = progress along track + penalty for resets. Optionally **combined loss** for a few k steps: L = L_RL + λ·L_BC (e.g. λ = 0.5) so the policy doesn’t drift too far from expert early on.
  - **Integration with IQN:** (1) Train BC (Level 1 or 2), save encoder (and if applicable full feature part). (2) Create IQN network, load pretrained parts into ``img_head`` (and if you added LSTM, into that). (3) Option A: Start learner from this checkpoint; no L_BC. Option B: Add an optional **BC loss** in the learner: on each batch, if the transition has an “expert action” flag (e.g. from a replay buffer filled by expert), add λ·L_BC; then decay λ over time. Option B requires storing “expert action” in the buffer and a small change in ``train_on_batch``.

**Level 5: Latent Action Model (Genie-style) for unlabeled video**
  - **What:** Train a model that predicts **discrete latent actions** from video only (no ``inputs``). Use VQ-VAE–style LAM: encoder(frames) → latent code; decoder(latent) → next frame or state. Then map latent codes to real actions using a **small labeled set** (e.g. 200 replays with ``manifest.json`` inputs).
  - **Data:** Unlabeled: any TrackMania video (e.g. YouTube). Labeled: our captured replays with ``inputs`` for mapping latent → steer/accel/brake.
  - **Integration with IQN:** (A) Use LAM to **generate synthetic (s, a)** from unlabeled video: decode latent → map to discrete action id (from labeled mapping); add to replay buffer or to a BC dataset. (B) Or: use LAM encoder as a **feature extractor** and train a small “latent → action” head on labeled data; then use this as a fixed or finetuned policy to fill buffer or to initialize IQN’s policy (would require defining actions in the same space as IQN). This is the most research-heavy option and likely a new script/repo.

**Level 6: Multimodal / uncertainty-aware BC**
  - **What:** Instead of a single action per state, model **distribution** of actions: Mixture of Experts, Conditional VAE, or quantile regression for continuous actions. Helps when multiple good actions exist (e.g. early vs late apex).
  - **Integration with IQN:** Use the **encoder** from this BC model as ``img_head``; the rest of IQN remains unchanged (IQN already models Q-distribution over actions). Optionally use the BC policy’s **mixture/quantile output** as a prior for exploration (e.g. bias sampling toward high-BC-probability actions early in training).

**Level 7: Track / trajectory embedding (compress track into vector, reconstruct trajectory)**
  - **What:** *“Сжатие данных трассы в вектор и восстановление траектории из реплея”.* Encode the **track** (e.g. as segment graph with jump edges) or **full trajectory** (sequence of states/positions from a replay) into a fixed-size vector (e.g. 128–512 dim). Train via reconstruction: encoder(track or trajectory) → z; decoder(z) → trajectory (or next states). References: SeCTAR (trajectory VAE), track-agnostic embeddings, GNN on track graph. Use the **encoder** as a compact representation of “which track” or “which trajectory” so the policy gets global context without millions of 3D points.
  - **Data:** Replays with state/position per frame (e.g. from ``manifest.json`` or TMInterface API); optionally precomputed track graphs (nodes = segments, edges = connectivity + jump edges).
  - **Integration with IQN:** (A) **Extra input:** Concatenate track/trajectory embedding z to the float features in IQN (requires one more input dim and a way to compute z at runtime: e.g. frozen encoder on current trajectory prefix or on track graph). (B) **Reward or planning:** Use z (or decoder(z)) for reward shaping or high-level planning; policy remains image-based. (C) **Transfer:** Pretrain encoder on many tracks; at test time feed embedding of new track (from its graph or a short rollout) so one policy can adapt to new tracks with a compact input.

**Level 8: State + position → remaining trajectory prediction (trajectory completion)**
  - **What:** *“Вектор текущего состояния машины и позиции на трассе → предсказание оставшейся части траектории из реплея”.* Input: current observation (or its embedding) + **position on track** (e.g. progress, checkpoint index, or segment id). Target: **remaining trajectory** from the expert replay (positions, or actions, or both). Train a model (e.g. CNN+LSTM or Transformer) to predict “how would the expert continue from here?”. This gives a **behavioral prior** or **plan** that can guide RL (e.g. auxiliary loss, or reward for following predicted trajectory, or planning in latent space).
  - **Data:** Same as BC: frames + ``manifest.json`` (inputs, state, position). For each frame t, input = (frame_t, state_t, progress_t); target = (positions or actions from t+1 to end of replay).
  - **Integration with IQN:** (A) **Auxiliary loss:** During IQN training, add a head that predicts “remaining trajectory” (or next K actions) and train it on expert replays in the buffer; shared ``img_head`` benefits. (B) **Planning / reward:** At inference or in the buffer, use the predicted remaining trajectory to compute a “progress-along-expert” reward or to bias action selection toward the predicted action. (C) **No direct weight transfer:** Use only as a separate module that outputs a prior; IQN policy gets image + float inputs as today.

---

Summary table: pretrain → IQN pipeline
--------------------------------------

+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| Level  | Pretrain type                     | Output you get                                    | How to plug into current IQN pipeline                            |
+========+===================================+==================================================+==================================================================+
| 0      | Unsupervised (AE/VAE/SimCLR)       | ``encoder.pt`` (CNN only)                        | Load into ``network.img_head``; save full IQN checkpoint; start learner from it. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 1      | BC (frames → actions)              | Encoder + action head                            | Load encoder → ``img_head``; optional: warm buffer + short L_BC in learner. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 2      | BC + LSTM                         | Encoder + LSTM + action head                     | Use encoder in ``img_head``; or extend IQN with LSTM and load both. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 3      | DAGGER                            | Better BC policy / dataset                       | Same as Level 1/2: better init or better buffer for IQN.         |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 4      | BC + RL (IQN/PPO)                 | Pretrained policy + RL fine-tuning               | Init IQN from BC; optional L = L_RL + λ·L_BC in learner; then pure IQN. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 5      | LAM (Genie)                       | Latent actions from video; mapping to real actions | Synthetic (s,a) for buffer/BC; or LAM encoder as feature extractor for IQN. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 6      | Multimodal BC                     | Encoder + mixture/VAE/quantile head              | Encoder → ``img_head``; optional exploration prior from BC.       |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 7      | Track/trajectory embedding        | Encoder z (track or trajectory → vector)        | Extra float input (z); or reward/planning from z; transfer to new tracks. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+
| 8      | State+position → remaining traj.  | Model “expert continuation” from (s, progress)  | Auxiliary loss; or reward/prior from predicted trajectory; no weight transfer. |
+--------+-----------------------------------+--------------------------------------------------+------------------------------------------------------------------+

---

Practical order of experiments (revised)
----------------------------------------

**Phase A — Visual pretrain and BC (no track embedding yet)**

1. **Level 0:** Run visual pretrain (AE or SimCLR) on ``maps/img``; compare IQN **from scratch** vs **from encoder.pt** (experiments 1, 3, 5, 6 from “Simplest experiments”).
2. **Level 1:** Implement BC (frames → actions from ``manifest.json``); sanity check with minimal data (experiment 7); then scale; load encoder into IQN and compare.
3. **Level 2** (optional): Add LSTM / action history to BC if temporal structure matters.
4. **Level 4:** BC → IQN fine-tuning; optional L_BC for a few k steps. Compare to IQN from scratch and from Level 0 only.
5. **Level 3 (DAGGER)** if you have an expert and want to improve BC before RL.
6. **Level 5 (LAM)** for unlabeled video; **Level 6** if you need multimodal action modeling.

**Phase B — Track representation and observation**

7. **Experiment 11:** Add **progress** (or checkpoint index) to float inputs; compare image-only vs image+progress for sample efficiency and final time.
8. If you introduce **3D representation** (segment graph, jump edges, air corridors): use it first for **reward** and optional planning; then consider feeding a compact encoding (e.g. “current segment” or “distance to next jump”) as extra observation.

**Phase C — Track/trajectory embedding and trajectory completion (your ideas 2 and 3)**

9. **Level 7 (track/trajectory embedding):** Implement encoder that compresses track or trajectory into a vector; train with reconstruction loss on replays. Plug into IQN as extra input (z) or for reward/planning; run **experiment 12** (same track vs new track) to measure transfer.
10. **Level 8 (state + position → remaining trajectory):** Train “expert continuation” model: (current state, progress) → remaining trajectory. Use as auxiliary loss, reward, or behavioral prior; no mandatory weight transfer into IQN.

**Order summary:** 0 → 1 → (2) → 4 → (3,5,6) → experiment 11 → (3D reward/obs) → 7 → 12 → 8.

---

References (from knowledge base)
--------------------------------

- **Imitation learning / BC:** DAGGER (compounding errors O(Tε)); i.i.d. assumption violation in BC; BC + RL (e.g. YSDA practical RL course, Week 10).
- **RL surveys:** Imitation learning as a way to tackle sparse rewards; PPO for fine-tuning BC policies.
- **Genie:** arXiv:2402.15391 — latent action model from video without labels; mapping to actions with small labeled set.
- **Architectures:** CNN for vision; LSTM or Transformer for long-horizon temporal dependencies (e.g. 64 FPS × 60 s).
- **Track representation:** Checkpoint system, LIDAR, images, API coordinates (tmrl, TMInterface); 3D representation (segment graph with jump edges, air corridors, volumetric checkpoints) for jumps and shortcuts.
- **Track/trajectory embeddings:** SeCTAR (ICML 2018, trajectory VAE, planning in latent space); track-agnostic embeddings (CNN/GNN, one model for all tracks); graph-based track embedding (GNN on segment graph, 64–256 dim); Point-ROPE for 3D positional encoding; combination GNN + RoPE for 3D jumps.
