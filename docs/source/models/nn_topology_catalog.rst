.. _nn-topology-catalog:

NN topology catalog (supported stacks)
=======================================

This page lists **every routing path** the training code can build from YAML ``nn`` + ``training.algorithm`` + (IQN only) ``btr:``. It complements the narrative pages :doc:`iqn_architecture`, :doc:`ppo_architecture`, :doc:`btr_architecture` and the field-by-field :ref:`nn-yaml-reference` in :doc:`../configuration_guide`.

**Authoritative schema:** ``config_files/nn_schema.py`` (``NnConfig``). **Factory:** ``trackmania_rl/agents/policy_models/multimodal_torch_fusion.py`` (``TorchMultimodalActorCritic``, ``build_multimodal_fusion_uncompiled``), ``ppo_wiring.py``, ``iqn.py`` (``build_iqn_network_uncompiled``), ``hf_actor_critic.py``.

**Vision branch name** in code is from ``infer_vis_branch(nn.vis)`` in ``nn_schema``: ``none`` (``no_image``), ``cnn``, ``native_transformer`` (``transformer`` with ``use_hf_backbone: false``), ``hf_transformer`` (``transformer`` with ``use_hf_backbone: true``).

.. warning::

   Most nested ``nn`` models use Pydantic ``extra="ignore"``. Unknown or misspelled keys under ``nn.*`` are **silently dropped** at load — they do **not** error. Prefer this catalog + :ref:`nn-yaml-reference` over guesswork.

1. ``fusion_mode: none`` (no multimodal stack)
----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 22 28 40

   * - Algorithm
     - Vision (effective ``infer_vis_branch``)
     - Built module(s)
     - Notes
   * - **IQN**
     - ``cnn`` (default if you omit ``no_image`` and do not set ``transformer`` — schema adds empty ``cnn``)
     - ``IQN_Network`` (``trackmania_rl/agents/iqn.py``)
     - Optional ``btr:`` (Munchausen, IMPALA CNN knobs merged into ``nn.vis.cnn`` when omitted, LayerNorm / NoisyNet on heads). See :doc:`btr_architecture`.
   * - **IQN**
     - ``none`` (``vis.no_image: true``)
     - ``IQN_Network``
     - Float-only; image tensor can be zeros at runtime.
   * - **IQN**
     - ``hf_transformer``
     - ``IQNSharedBackboneNetwork`` + headless ``HfActorCritic`` (``nn_build/iqn_multimodal.py``, ``hf_actor_critic.py``)
     - Requires ``pip install -e ".[policy]"`` (Hugging Face stack).
   * - **IQN**
     - ``native_transformer`` (``vis.transformer`` without HF) **with** ``fusion_mode: none``
     - —
     - **Not wired:** ``build_iqn_network_uncompiled`` raises. Use ``fusion_mode`` in ``vision_transformer`` / ``post_concat`` / ``unified``, or HF vision with ``use_hf_backbone: true``.
   * - **PPO**
     - ``cnn`` / ``none``
     - ``PpoActorCritic`` (``ppo_actor_critic.py``)
     - CNN kwargs **only** from ``nn.vis.cnn`` (no ``btr:`` merge on this path). ``no_image`` → float-only trunk.
   * - **PPO**
     - ``hf_transformer``
     - ``HfActorCritic`` (``hf_actor_critic.py``)
     - HF CLS + float MLP + shared trunk + policy/value heads.
   * - **PPO**
     - ``native_transformer`` only (``transformer`` present, ``use_hf_backbone: false``, no ``cnn``)
     - ``PpoActorCritic`` (degenerate)
     - **Pitfall:** no conv stem is built → **float-only** behavior (image side zeros). For native patch vision use ``fusion_mode: vision_transformer`` (or another multimodal mode), not ``none``.

2. Multimodal fusion modes
--------------------------

Here ``nn.fusion_mode`` is one of ``vision_transformer``, ``post_concat``, or ``unified``.

**Shared body:** ``TorchMultimodalActorCritic`` (``multimodal_torch_fusion.py``).

* **PPO** — ``include_policy_heads=True`` (trunk + ``policy_head`` / ``value_head``).
* **IQN** — ``include_policy_heads=False``; wrapped by ``IQNSharedBackboneNetwork`` + ``iqn_fc`` + dueling heads (same quantile path as classic IQN after fusion hidden).

**Float MLP width** for fusion builds: ``nn.encoder.mlp.hidden_dim`` if set, else ``nn.float.mlp.hidden_dim`` (``float_hidden_dim_effective()``).

**Fusion trunk kind** (after early tokens / concat): ``nn.encoder.fusion_encoder`` if set, else inferred by ``infer_fusion_encoder`` in ``nn_schema``:

1. If ``fusion_encoder`` is set → use it (must agree with ``encoder.transformer.use_hf_backbone``; schema forbids ``native_transformer`` + HF backbone on the same encoder slot).
2. Else if ``encoder.transformer.use_hf_backbone: true`` → ``hf_embedding`` (HF model with ``inputs_embeds``, e.g. BERT-class; path from ``encoder.transformer.model_name_or_path`` or ``encoder.hf_embedding``).
3. Else if ``fusion_mode == vision_transformer`` → ``linear`` (concat embeddings → ``bridge`` Linear to ``decoder.dense_hidden_dimension``).
4. Else → ``native_transformer`` (``torch.nn.TransformerEncoder`` on the fusion sequence; ``n_layers: 0`` means **no** encoder layer — optional blocks skipped via ``_make_encoder_optional``).

Explicit kinds **``mlp``** / **``cnn``** / **``hf_embedding``** use ``nn.encoder.fusion_mlp``, ``fusion_cnn``, ``hf_embedding`` respectively (see :ref:`nn-yaml-reference`).

``vision_transformer`` mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image → float MLP → fuse (default trunk ``linear`` unless overridden).

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - ``infer_vis_branch``
     - Image path (if ``use_image_head``)
     - Fusion path
   * - ``cnn``
     - ``_build_img_head`` from ``nn.vis.cnn`` → Linear to ``vis.d_model``
     - Default ``linear``: concat(image emb, float MLP) → ``bridge``. If ``fusion_encoder`` is non-linear, vision+float concat is projected to a short **sequence** (length ``encoder.transformer.post_concat_seq_len``) then fusion trunk.
   * - ``native_transformer``
     - ``PatchEmbed2d`` + optional ``vis`` ``TransformerEncoder`` (``patch_size`` must divide ``H_downsized``, ``W_downsized``)
     - Same as above after pooling / embedding.
   * - ``hf_transformer``
     - HF vision backbone + optional ``vis_refine`` (native encoder on tokens)
     - Same default ``linear`` / optional non-linear fusion trunk.
   * - ``none``
     - No image tokens
     - Float-only side still participates in concat / sequence as implemented.

``post_concat`` mode
~~~~~~~~~~~~~~~~~~~~

Tokenize vision + float, then fusion trunk.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - ``encoder.post_concat_layout``
     - Behavior (simplified)
     - Typical vision
   * - ``fused_vector``
     - Image branch (CNN / native / HF) and float MLP produce a **single fused vector** → projected to ``post_concat_seq_len`` tokens at ``fuse_d_model`` → fusion trunk (default ``native_transformer`` unless overridden).
     - CNN, native patch stack, or HF with ``fusion_tokens: summary`` (single vector per image).
   * - ``token_sequence``
     - Vision contributes **one or many** tokens at ``fuse_d_model``; float side is **raw** or **MLP-hidden** tokens (``float_token_input``) in **dense** or **per_feature** layout (``float_token_layout``). ``per_feature`` forces ``float_token_input: raw`` and ``token_sequence`` (schema).
     - CNN → one vision token; native patches → many; HF with ``fusion_tokens: patch_tokens`` → many (requires ``token_sequence`` — schema).

``unified`` mode
~~~~~~~~~~~~~~~~

Joint sequence over image token(s) and learned float token(s).

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - ``infer_vis_branch``
     - Image tokens
     - Constraints
   * - ``cnn``
     - **One** image token (conv → Linear to ``fuse_d_model``)
     - Floats → ``unified_float_tokens`` via ``Linear(float_dim -> K*d)``; joint ``pos_uni``; fusion trunk per ``fusion_encoder``.
   * - ``native_transformer``
     - Patch grid tokens at ``vis.d_model``; must equal ``encoder.transformer.d_model`` (``fuse_d_model``)
     - Schema enforces ``vis.transformer.d_model == encoder.transformer.d_model``.
   * - ``hf_transformer``
     - **N** tokens from HF backbone (count derived from processor / backbone); projected to ``fuse_d_model``
     - Optional native ``vis`` ``TransformerEncoder`` refine; ``n_layers: 0`` skips it. Same joint fusion trunk options as other multimodal modes.

``float_feature_extractor`` (2× MLP on floats) is **omitted** for ``unified`` and for ``post_concat`` + ``token_sequence`` + ``float_token_input: raw`` — floats enter tokenization directly where that path applies.

3. IQN decoder and BTR on heads
-------------------------------

Applies to **classic** ``IQN_Network`` and **shared-backbone** IQN (multimodal / HF vision).

* **Slots** ``decoder.advantage`` and ``decoder.value``: **either** ``mlp`` **or** ``transformer`` (not both per slot). Aliases: ``mlp.layers`` ↔ ``n_hidden_layers``; ``hidden`` ↔ ``hidden_dim``.
* **Transformer slot:** native ``torch.nn.TransformerEncoder`` on chunked state; schema requires ``decoder.shared_input: post_tau`` if any slot uses ``transformer``.
* **BTR** dense-head flags (LayerNorm, NoisyNet, ``noisy_sigma0``) apply via ``iqn_btr_mlp_head_kw_from_config`` (see :doc:`btr_architecture`).

4. Warm start and checkpoints
-----------------------------

* **Multimodal PPO:** ``nn.init_from_pretrained`` — Rulka fusion ``save_pretrained`` dir; loaded after build in ``make_multimodal_fusion_network_pair`` (unless skipped via utility flag; see :ref:`nn-yaml-reference`).
* **Multimodal IQN:** same directory format may exist, but **automatic** hub load is **not** guaranteed to mirror PPO — prefer continuing from ``weights1.torch`` / explicit load in your workflow.
* **Hub JSON** may carry ``rulka_transformers.vis_cnn`` for CNN stems; older bundles without it fall back to default conv kwargs (see :doc:`ppo_architecture`).

5. Reference YAML files
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Role
   * - ``config_default.yaml`` / ``config_btr.yaml``
     - IQN ``fusion_mode: none`` + CNN; ``config_btr.yaml`` enables full ``btr:`` recipe.
   * - ``config_btr_post_concat_cnn_transformer.yaml``
     - IQN + ``post_concat`` + CNN + native fusion ``TransformerEncoder`` + ``btr:``.
   * - ``config_ppo.yaml``
     - PPO baseline (``fusion_mode: none``); starting point for native ``vision_transformer`` (change ``fusion_mode`` + ``vis.transformer`` / remove ``cnn`` as needed).
   * - ``config_ppo_cnn_mlp.yaml``
     - Minimal PPO CNN + float MLP.
   * - ``config_ppo_post_concat_cnn_tf.yaml``
     - PPO ``post_concat`` + CNN + native fusion transformer.
   * - ``config_ppo_transformer.yaml``
     - PPO ``post_concat`` + **HF** timm vision + **HF** fusion encoder (historical filename).

There is **no** single YAML file covering every cell of the tables above; combine :ref:`nn-yaml-reference` with the closest example and edit ``nn`` fields.

See also
--------

- :doc:`iqn_architecture` — IQN routing and tensors
- :doc:`ppo_architecture` — PPO variants A/B/C
- :doc:`btr_architecture` — BTR flags on IQN
- :ref:`nn-yaml-reference` in :doc:`../configuration_guide`
- ``config_files/nn_schema.py`` — validators for mutually exclusive options and geometry
