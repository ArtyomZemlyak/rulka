.. _ppo_architecture:

PPO actor-critic architecture
=============================

This page documents the policy and value network used for **on-policy policy
optimization** with a **discrete** shared-trunk actor-critic. The **same** stacks
and ``ppo_wiring`` factory are used when ``training.algorithm`` is ``ppo``,
``dpo``, or ``grpo``; **this page** details the **network** (Variants A/B/C) and
the **PPO** training loop (GAE, clipped surrogate, value loss). **DPO** keeps the
same bodies but trains from preference pairs; **GRPO** uses group-relative
trajectory returns — see :doc:`grpo_architecture`.

When ``training.algorithm`` is ``ppo`` specifically, you get a **shared-trunk
actor-critic** as described below. Implementation lives under
``trackmania_rl.agents.policy_models`` and is wired via
``trackmania_rl.agents.algorithms.ppo_wiring``.

For the value-based baseline (quantile IQN + replay), see :doc:`iqn_architecture`.
**BTR** (:doc:`btr_architecture`) applies to **IQN only**. **Variant A** PPO reads
``nn.vis.cnn`` (same ``_build_img_head`` flags as IQN, without merging ``btr:``).
**Fusion** variants that use the **CNN** vision branch (``infer_vis_branch`` → ``cnn``)
also call ``_build_img_head`` with kwargs resolved from ``nn.vis.cnn`` — the same
single source as IQN and Variant A PPO (``trackmania_rl/nn_build/vis_cnn_head.py``).
``TorchMultimodalActorCritic`` (without policy heads) backs **IQN** when ``training.algorithm`` is ``iqn`` and ``nn.fusion_mode != none``.

YAML knobs for PPO routing and vision (``nn.fusion_mode``, ``nn.vis``, ``nn.float``, ``nn.encoder``): :ref:`nn-yaml-reference` in :doc:`../configuration_guide`. Optional :ref:`nn-rl-parameter-freeze` (e.g. ``nn.vis.freeze``, ``nn.encoder.freeze``, ``nn.decoder.shared_trunk_freeze``) applies to PPO the same way as to IQN where documented.

Why this stack is shaped this way
---------------------------------

**Image + float inputs.** TrackMania gives both rendered frames and a normalized
float vector (geometry, speed, gear, …). Vision learns what the road *looks* like;
the float path carries signals that are tedious to infer from pixels alone.
**Why both feed one trunk:** a single representation is used for action logits and
for the value baseline, so both tasks co-adapt the same features.

**Shared trunk, two heads (policy + value).** The **policy head** defines a
categorical over discrete actions (including multi-offset layouts). The **value
head** predicts expected return from each state. **Why actor-critic:** the critic
feeds GAE and the value loss, which reduces variance of policy gradients compared
to pure Monte Carlo returns.

**On-policy PPO loop.** Data are generated with the *current* policy, then
discarded after a few epochs of updates. **Why not replay (here):** keeps the
off-policy correction simple; PPO’s clipped ratio explicitly limits change w.r.t.
the policy that collected the batch, which stabilizes training when rewards are
dense and correlated.

**Collectors vs learner.** Environments run in parallel processes; inference must
be fast. Collectors only **forward + sample** and enqueue lists; the learner does
**backward + optimizer** on aggregated GPU batches. **Why store ``ppo_log_probs``:**
PPO’s ratio compares :math:`\pi_\theta` to :math:`\pi_{\mathrm{old}}` — the policy
that actually produced the actions on the rollout.

**GAE (:math:`\gamma`, :math:`\lambda`).** Trade off bias vs variance of advantage
estimates using the value function and n-step structure. **Why:** raw one-step
TD is noisy; full Monte Carlo is high-variance; GAE interpolates.

**Clipped surrogate.** Penalty if the new policy assigns much more probability to
the taken actions than the behavior policy did. **Why:** approximate trust region
— large policy jumps on one minibatch tend to break the on-policy assumption.

**Value loss + entropy bonus.** The critic is trained toward return targets; the
entropy term discourages premature collapse to deterministic actions. **Why:** without
entropy, exploration from stochastic sampling fades as logits sharpen.

Routing: which network is built?
--------------------------------

``ppo_wiring.make_network`` chooses **exactly one** implementation (first match wins).
For an **uncompiled** policy on CPU (e.g. BC with ``bc_use_rl_architecture``), the same
routing is implemented by ``ppo_wiring.build_ppo_policy_uncompiled`` (no ``torch.compile`` / forced CUDA).

1. If ``get_config().transformers.fusion_mode`` (i.e. ``nn.fusion_mode``) is **not** ``none`` → **Variant C** — ``TorchMultimodalActorCritic`` (native ``torch.nn.TransformerEncoder`` stacks; HF vision only inside ``vision_transformer`` when ``nn.vis.transformer.use_hf_backbone``).
2. Else if ``nn.vis.transformer`` is set **and** ``use_hf_backbone`` is true → **Variant B** — ``HfActorCritic`` (Hugging Face ``AutoModel`` CLS + float MLP + shared trunk).
3. Else → **Variant A** — ``PpoActorCritic`` (``nn.vis.cnn`` image head via the same ``_build_img_head`` kwargs as IQN, or float-only if ``no_image`` / no CNN).

**Why three variants.** **A** is the default conv + MLP path (fast, full control of
CNN flags). **B** plugs in a **pretrained HF** vision backbone when you want
transfer from large-scale image pretraining. **C** uses **fusion transformers** so
image and float features interact through attention (and optional hub round-trip),
instead of a single early concat — useful when alignment between modalities is
subtle.

.. note::
   If ``fusion_mode: none`` but YAML declares **only** ``nn.vis.transformer`` with ``use_hf_backbone: false`` (no ``cnn``), the CNN branch sees no image stem → **float-only** PPO (zeros image tensor at inference). For CNN PPO, keep ``nn.vis.cnn``.

Training stack (processes and modules)
--------------------------------------

``scripts/train.py`` starts a **learner** process and several **collector** processes.
For ``training.algorithm: ppo``, the learner runs ``learner_ppo``; collectors attach
``PPOInferer`` and push rollouts into multiprocessing queues. The **same** policy
weights exist as a compiled CUDA module in collectors and as trainable parameters
in the learner; after each PPO update the learner copies state dict into the
shared ``uncompiled`` copy under ``shared_network_lock`` (collectors refresh their
view from that copy — same pattern as IQN’s weight sync).

.. graphviz::

   digraph ppo_process_stack {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      train [label="scripts/train.py", style="rounded,filled", fillcolor=lightcyan];
      lp [label="learner_process.py\nif algorithm == ppo → learner_ppo", style="filled", fillcolor=lightyellow];
      cp [label="collector_process.py × N\nis_policy_optimization_algorithm()", style="filled", fillcolor=lightyellow];
      inf [label="PPOInferer\n(forward + sample + log p, V)"];
      lppo [label="learner_ppo.py\nrollout batch → GAE → PPO loss → Adam"];
      pol [label="policy network\n(make_network)", style="filled", fillcolor=lightgreen];
      sh [label="uncompiled_shared_network\n+ shared_network_lock", style="filled", fillcolor=lightpink];
      q [label="rollout_queues\n(multiprocessing)"];
      train -> lp;
      train -> cp;
      lp -> lppo;
      cp -> inf;
      inf -> pol;
      lppo -> pol;
      inf -> q [label="put"];
      q -> lppo [label="get"];
      lppo -> sh [label="load_state_dict"];
      inf -> sh [style=dashed, label="weights for inference"];
   }

**Registry:** ``training.algorithm: ppo`` resolves to ``trackmania_rl.agents.algorithms.ppo_wiring`` via ``registry.get_wiring()`` (same module also serves DPO/GRPO for **network** build only).

**``uncompiled_shared_network`` and the lock.** After each update the learner
writes weights into a shared module; collectors read that snapshot for inference.
**Why:** one authoritative weight tensor for many parallel games without training
inside env processes.

Overview
--------

Like IQN, the model consumes two branches:

- **Image** ``(B, 1, H, W)`` — grayscale frame (or a zero tensor if the image
  head is disabled);
- **Float** ``(B, float_input_dim)`` — the same normalized state vector as IQN
  (waypoints, gear, velocity, etc.).

Outputs:

- **Policy:** logits for a categorical distribution over actions. Single-decision
  mode: ``(B, n_actions)``. Multi-action mode (``rl_action_offsets_ms`` with more
  than one offset): ``(B, n_actions_per_block * n_actions)`` reshaped to
  ``(B, N, n_actions)`` inside ``evaluate_actions``.
- **Value:** scalar ``V(s)`` per sample, ``(B, 1)`` before squeeze.

**Float normalization** :math:`(x-\mu)/\sigma` (running buffers) matches IQN so BC /
IQN / PPO can share statistics. **Why:** stable MLP inputs when raw speeds and
distances have different scales.

.. graphviz::

   digraph ppo_overview {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      img [label="img\n(B,1,H,W)", style="filled", fillcolor=lightblue];
      flt [label="float_inputs\n(B,F)", style="filled", fillcolor=lightblue];
      cnn [label="Image head\n(optional CNN)"];
      mlp [label="Float MLP\n2×Linear+ReLU"];
      nrm [label="(x−μ)/σ\nper feature"];
      cat [label="Concat\n(B, D_vis + D_float)"];
      trk [label="Trunk\n2×Linear+ReLU"];
      pi [label="policy_head\nLinear → logits", style="filled", fillcolor=lightyellow];
      v [label="value_head\nLinear → V(s)", style="filled", fillcolor=lightyellow];

      img -> cnn;
      flt -> nrm -> mlp;
      cnn -> cat;
      mlp -> cat;
      cat -> trk -> pi;
      trk -> v;
   }

Variant A: CNN actor-critic
---------------------------

Class: ``PpoActorCritic`` in ``ppo_actor_critic.py``. Example YAML: ``config_ppo_cnn_mlp.yaml`` (minimal) or ``config_ppo.yaml`` with ``nn.fusion_mode: none`` and ``nn.vis.cnn``.

Image branch
~~~~~~~~~~~~

If ``nn.vis.no_image`` is false and ``nn.vis.cnn`` is present, the stem calls the **same** ``_build_img_head`` as IQN (``trackmania_rl/agents/iqn.py``) with flags taken **directly** from ``nn.vis.cnn``: ``use_impala_cnn``, ``impala_model_size``, ``use_spectral_norm``, ``use_adaptive_maxpool``, ``adaptive_maxpool_size``. The conv output is flattened to ``conv_head_output_dim``.

Unlike IQN, this path does **not** read ``btr:`` — only ``nn.vis.cnn``. (BTR is an IQN-only bundle.)

If ``no_image`` is true or there is no CNN stem, ``img_head`` is omitted and the trunk input is **float-only**.

Float branch
~~~~~~~~~~~~

1. Normalize with buffers ``float_inputs_mean`` / ``float_inputs_std``.
2. Two linear layers with ReLU: ``float_input_dim → float_hidden_dim → float_hidden_dim``.

Width ``float_hidden_dim`` comes from ``get_config().float_hidden_dim`` → ``nn.float.mlp.hidden_dim`` (``encoder.mlp`` override applies to **fusion** PPO only, not this variant).

Fusion and trunk
~~~~~~~~~~~~~~~~

- With image: ``h = concat(CNN(img), float_MLP(float))``.
- Trunk: ``Linear → ReLU → Linear → ReLU`` with width ``dense_hidden_dimension``.

Heads
~~~~~

- ``policy_head``: ``dense_hidden_dimension → n_actions * n_actions_per_block``.
- ``value_head``: ``dense_hidden_dimension → 1``.

At inference and training, ``evaluate_actions`` computes **log-probability** and
**entropy** from the categorical defined by logits (product of ``N`` categoricals
in multi-action mode).

.. graphviz::

   digraph ppo_evaluate {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      fwd [label="forward(img, float)\n→ logits, value"];
      rs [label="reshape logits\n(B,N,A) if multi-action"];
      cat [label="Categorical(logits)\nper head / factor"];
      out [label="log π(a|s), H[π], V(s)", style="filled", fillcolor=lightgreen];
      fwd -> rs -> cat -> out;
   }

Variant B: Hugging Face vision backbone
---------------------------------------

Enabled when ``nn.fusion_mode`` is ``none`` and ``nn.vis.transformer.use_hf_backbone`` is ``true``
(requires ``pip install -e ".[policy]"``). Class: ``HfActorCritic`` in
``hf_actor_critic.py``.

Factory: ``make_hf_ppo_network_pair`` in ``ppo_wiring.make_network``.

.. graphviz::

   digraph ppo_hf {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      img [label="img\n(B,1,H,W)", style="filled", fillcolor=lightblue];
      flt [label="float_inputs\n(B,F)", style="filled", fillcolor=lightblue];
      prep [label="resize / RGB /\nprocessor norm"];
      vit [label="HF AutoModel\n(CLS token)"];
      fmlp [label="Float MLP +\nLinear → hidden"];
      cat [label="Concat\n(B, 2·H)"];
      trk [label="Trunk + heads\n(same idea as CNN PPO)"];
      img -> prep -> vit -> cat;
      flt -> fmlp -> cat;
      cat -> trk;
   }

Pixels are interpolated to the processor’s height/width, duplicated to 3
channels if needed, mapped from ``[-1,1]`` to ``[0,1]``, then normalized with
the processor’s mean/std when available. Float features use the same two-layer
MLP as the CNN variant, then a linear projection to the backbone **hidden_size**
so that **image CLS** and **float** embeddings are concatenated before the shared
trunk.

Variant C: Multimodal fusion (``nn.fusion_mode ≠ none``)
--------------------------------------------------------

When ``nn.fusion_mode`` is one of ``vision_transformer``, ``post_concat``, or ``unified``, ``ppo_wiring.make_network`` builds ``TorchMultimodalActorCritic`` (``multimodal_torch_fusion.py``). The **multimodal bundle** exposed as ``get_config().transformers`` combines ``nn.fusion_mode``, ``nn.init_from_pretrained``, and ``nn.encoder.transformer`` (plus ``nn.vis.transformer`` for the image side).

**Common after fusion:** ``Linear → ReLU → Linear → ReLU`` trunk of width ``nn.decoder.dense_hidden_dimension`` (same config field name as IQN; PPO reads it as ``dense_hidden_dim``), then policy / value linear heads.

**Float width:** ``float_hidden_dim_effective()`` = ``nn.encoder.mlp.hidden_dim`` if set, else ``nn.float.mlp.hidden_dim``.

Sub-modes
~~~~~~~~~

``vision_transformer``
   **Image →** either (a) **native** ``PatchEmbed2d`` + ``nn.TransformerEncoder`` on patch tokens + mean-pool, using ``nn.vis.transformer`` (``d_model``, ``n_layers``, ``n_heads``, ``ff_mult``, ``dropout``, ``patch_size``), or (b) **HF** backbone (CLS) + optional ``vis_refine`` encoder when ``use_hf_backbone: true`` (requires ``transformers``). **Float →** two-layer MLP. **Fusion:** concat(image_emb, float_emb) → ``bridge`` Linear to ``dense_hidden_dim`` → trunk.

``post_concat``
   **Image →** if ``use_image_head`` and the vision branch is **CNN**, ``_build_img_head`` with flags from ``nn.vis.cnn`` (IMPALA / adaptive pool / spectral norm as configured). Native patch or HF vision use ``nn.vis.transformer`` instead. **Float →** ``fused_vector`` layout: two-layer MLP (width ``float_hidden_dim_effective()``), then concat with the vision vector and projection to a token sequence (length ``nn.encoder.transformer.post_concat_seq_len``). ``token_sequence`` layout (e.g. ``float_token_layout: per_feature``) uses raw float tokens without that MLP. Then learned positions, **fusion** ``nn.TransformerEncoder`` from ``nn.encoder.transformer`` (when not ``linear``), pool → ``bridge`` → trunk.

   **Hub round-trip:** fusion ``save_pretrained`` / ``from_pretrained`` JSON may include ``rulka_transformers.vis_cnn`` (dump of ``nn.vis.cnn``) so CNN stems match after reload; older hubs without ``vis_cnn`` fall back to the baseline 4-conv kwargs for the CNN branch.

``unified``
   Single joint encoder over **image token(s)** and **learned float token(s)** (``unified_float_tokens``). **No** separate float MLP in this mode (float raw features go through ``float_to_tokens``). **Native** patch vision: ``vis.transformer.d_model`` must equal ``encoder.transformer.d_model`` (schema). **CNN** vision contributes **one** token; **HF** vision contributes **N** patch tokens (``N`` inferred from the HF backbone when the model is built). Fusion trunk: same ``fusion_encoder`` options as other multimodal modes (default ``native_transformer`` when encoder is not HF; else ``hf_embedding`` per ``infer_fusion_encoder``).

.. graphviz::

   digraph ppo_fusion_modes {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      m1 [label="vision_transformer:\nimage (patch or HF CLS) + float MLP\n→ concat → bridge"];
      m2 [label="post_concat:\nCNN ∥ float MLP → tokenize\n→ Enc_fusion → pool → bridge"];
      m3 [label="unified:\npatch tokens ∥ float tokens\n→ Enc_fusion → pool → bridge"];
      tr [label="Trunk + policy / value heads", style="filled", fillcolor=lightgreen];
      m1 -> tr;
      m2 -> tr;
      m3 -> tr;
   }

**Patch geometry:** ``nn.vis.transformer.patch_size`` must divide ``H_downsized`` and ``W_downsized`` for native ``vision_transformer`` and ``unified``. ``post_concat`` ignores patch size on the image side (CNN stem).

Optional **warm start:** ``nn.init_from_pretrained`` (Rulka fusion ``save_pretrained`` directory) after build; trust flags follow ``nn.encoder.transformer.trust_remote_code``.

Example YAML for **post_concat** + HF two-tower fusion: ``config_ppo_transformer.yaml``. For native ``vision_transformer`` (patch + Linear fuse, no HF), start from ``config_ppo.yaml`` and set ``nn.fusion_mode: vision_transformer`` with ``nn.vis.transformer.use_hf_backbone: false``. See :ref:`nn-yaml-reference` in :doc:`../configuration_guide`.

IQN vs PPO (same inputs, different heads)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Aspect
     - IQN (:doc:`iqn_architecture`)
     - PPO (this page)
   * - Output
     - Distributional **Q(s,a,τ)** via quantile embedding + dueling
     - **π(a|s)** logits + **V(s)**
   * - Training
     - Replay buffer, n-step, quantile Huber, target network
     - On-policy rollouts, **GAE**, clipped surrogate, no replay
   * - Exploration
     - ε-greedy / Boltzmann / NoisyNet (config)
     - Stochastic policy sample from **Categorical**

Training flow (high level) — ``training.algorithm: ppo`` only
--------------------------------------------------------------

The following loop runs in ``trackmania_rl.multiprocess.learner_ppo`` when the
algorithm is **PPO**. It does **not** apply to DPO or GRPO (those learners reuse
collectors and often the same rollout tensor builder, but substitute their own
losses).

.. graphviz::

   digraph ppo_train {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      col [label="Collectors:\nPPOInferer\nnetwork(img, float)"];
      q [label="Rollout queues:\nlog p, V, states, actions"];
      rew [label="ppo_rewards:\nvectorized rewards +\npotential shaping (γΦ'−Φ)"];
      gae [label="learner_ppo:\nGAE → Â, R"];
      loss [label="PPO loss:\nclip + c_v·L_V − c_e·H"];
      col -> q -> rew -> gae -> loss;
   }

**Per-step training objective (schematic):** the learner minimizes a sum of **clipped
policy surrogate**, **value error** (often with clipping vs old values), and
**negative entropy** (i.e. entropy bonus). ``ppo_loss_components`` in
``trackmania_rl/agents/policy_optimization/ppo.py`` implements the algebra; exact
coefficients and schedules come from ``ppo:`` in YAML (:ref:`ppo-config` in
:doc:`../configuration_guide`).

.. graphviz::

   digraph ppo_loss_schematic {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      inp [label="batch:\nlog π_old, V_old, a, r, done\n+ forward: log π_θ, V_θ", style="filled", fillcolor=lightblue];
      rat [label="ratio r = exp(log π_θ − log π_old)"];
      clip [label="L_clip = min(r·Â, clip(r)·Â)"];
      lv [label="L_V: MSE or clipped value vs returns R"];
      ent [label="entropy H[π_θ]\n−c_e · mean(H)"];
      sum [label="loss = −L_clip + c_v·L_V − c_e·H\n(minimize)", style="filled", fillcolor=lightgreen];
      inp -> rat -> clip -> sum;
      inp -> lv -> sum;
      inp -> ent -> sum;
   }

Reading the loss diagram:

- **Clipped policy branch:** if the ratio :math:`r` moves outside
  :math:`[1-\varepsilon, 1+\varepsilon]`, the objective flattens — **so the
  update does not over-reward** actions that the new policy already exploits much
  more than the old one.
- **Value branch:** pulls :math:`V_\theta` toward returns (optionally clipped to
  old values) — **so the critic tracks** what will happen from each state, which
  in turn makes GAE’s advantages more accurate.
- **Entropy branch:** subtracts mean entropy from the loss (i.e. **maximize**
  entropy) — **so sampling stays diverse** early in training.

**Advantage flow:** ``compute_gae`` consumes step rewards, ``V(s_t)`` at collection
time, dones, and a bootstrap value at the rollout tail; it outputs per-step
advantages :math:`\hat{A}_t` and returns :math:`R_t` for the value target.

.. graphviz::

   digraph ppo_gae_flow {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      r [label="rewards r_t", style="filled", fillcolor=lightblue];
      v [label="values V(s_t)", style="filled", fillcolor=lightblue];
      d [label="dones", style="filled", fillcolor=lightblue];
      gae [label="compute_gae\n(γ, λ)"];
      out [label="Â_t , R_t", style="filled", fillcolor=lightgreen];
      r -> gae;
      v -> gae;
      d -> gae;
      gae -> out;
   }

**Rollout payload (per episode chunk):** collectors enqueue Python lists the learner
turns into GPU tensors. DPO/GRPO reuse the same keys for the shared builder
(``policy_rollout_batch``).

.. graphviz::

   digraph ppo_rollout_payload {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      col [label="collector\nPPOInferer step", style="filled", fillcolor=lightyellow];
      f [label="frames[]"];
      sf [label="state_float[]"];
      a [label="actions[]"];
      lp [label="ppo_log_probs[]"];
      pv [label="ppo_values[]"];
      q [label="rollout_queue\n+ end_race_stats", style="filled", fillcolor=lightpink];
      col -> f -> q;
      col -> sf -> q;
      col -> a -> q;
      col -> lp -> q;
      col -> pv -> q;
   }

**Why every queue field matters:** ``frames`` / ``state_float`` reconstruct
:math:`s_t`; ``actions`` are the labels for :math:`\log\pi(a|s)`;
``ppo_log_probs`` are :math:`\log\pi_{\mathrm{old}}` at collection time;
``ppo_values`` bootstrap GAE and the value loss. ``end_race_stats`` drives logging
and some reward edge cases (e.g. finish flags).

1. **Collectors** run the compiled (or eager) actor-critic on CUDA; append
   ``ppo_log_probs``, ``ppo_values``, frames, ``state_float``, actions.
2. **Learner** aggregates rollouts until ``ppo.rollout_steps_per_update``,
   builds tensors on GPU, computes **rewards** aligned with IQN’s dense +
   engineered terms (``reward_vectorized`` + fold; see ``rollout_rewards.py``).
3. **GAE** uses scheduled ``γ`` / ``λ`` (and optional ``ppo_*_schedule`` in
   config).
4. **Optimizer** updates the **same** network used in collectors; weights are
   copied to the shared inference copy under a lock.

Key design notes
----------------

- **Float inputs are always used** when ``float_input_dim > 0``; they are not
  auxiliary metadata. With image off, the policy is float-only.
- **Shared trunk** means gradients from policy and value both affect CNN/float
  representations (unless you freeze modules elsewhere).
- **Schedules:** learning rate and optional PPO coefficients can follow
  piecewise schedules on the global frame counter (see configuration guide).

Implementation references
-------------------------

- ``trackmania_rl/agents/policy_models/ppo_actor_critic.py`` — CNN PPO network.
- ``trackmania_rl/agents/policy_models/multimodal_torch_fusion.py`` — native
  Transformer multimodal fusion (``nn.fusion_mode`` / ``get_config().transformers``); IQN reuses it without policy heads.
- ``trackmania_rl/nn_build/vis_cnn_head.py`` — kwargs for ``_build_img_head`` from ``nn.vis.cnn`` (IQN, PPO CNN, multimodal CNN branch, BC, pretrain Level 0 when ``rl_config_path`` is used).
- ``trackmania_rl/agents/policy_models/hf_actor_critic.py`` — HF backbone PPO.
- ``trackmania_rl/agents/algorithms/ppo_wiring.py`` — factory, ``PPOInferer``,
  compile warmup hook.
- ``trackmania_rl/agents/policy_optimization/ppo.py`` — GAE, clipped loss.
- ``trackmania_rl/agents/policy_optimization/rollout_rewards.py`` — full TM
  rewards for PPO.
- ``trackmania_rl/reward_vectorized.py`` — shared dense reward + potentials.
- ``trackmania_rl/multiprocess/learner_ppo.py`` — PPO learner loop.
- ``trackmania_rl/multiprocess/collector_process.py`` — attaches ``PPOInferer`` for
  any policy-optimization algorithm (``ppo``, ``dpo``, ``grpo``).

See also
--------

- :doc:`grpo_architecture` — same policy network; group-relative trajectory training.
- DPO (preference learning, same network): :ref:`dpo-config` in :doc:`../configuration_guide`.
- :doc:`nn_topology_catalog` — full matrix of supported ``nn`` topologies.
- :doc:`iqn_architecture` — baseline value-based architecture.
- :doc:`btr_architecture` — IQN-only extras (not applied to PPO CNN factory).
- :doc:`../configuration_guide` — ``ppo:``, ``training:``, ``nn:``.
