.. _iqn_architecture:

IQN architecture
================

IQN (implicit quantile networks) is the **distributional off-policy** baseline: quantile Q-values, replay, target network. Implementation centers on ``trackmania_rl/agents/iqn.py``; **which module graph is built** depends on ``nn.fusion_mode`` and ``nn.vis`` (see below).

For optional BTR paper features, see :doc:`btr_architecture`. BTR is **not** a separate ``training.algorithm``; it toggles behavior on top of IQN.

Configuration lives under YAML ``nn`` (:ref:`nn-yaml-reference` in :doc:`../configuration_guide`). RL freeze flags: :ref:`nn-rl-parameter-freeze`.

Which network is built? (routing)
-----------------------------------

``build_iqn_network_uncompiled()`` (used by training and BC when ``algorithm: iqn``) picks **one** topology:

1. **Multimodal fusion** — ``nn.fusion_mode`` is ``vision_transformer``, ``post_concat``, or ``unified``. Builds the same ``TorchMultimodalActorCritic`` body as PPO with ``include_policy_heads=False``, wrapped by ``IQNSharedBackboneNetwork`` (``trackmania_rl/nn_build/iqn_multimodal.py``). Submodule ``fusion`` exposes ``forward_fusion_hidden``; IQN adds ``iqn_fc`` and dueling heads. Shared quantile + dueling math: ``trackmania_rl/nn_build/iqn_quantile_forward.py``. When the multimodal **vision branch** is CNN (default ``nn.vis.cnn``), the conv stem uses ``nn.vis.cnn`` via ``trackmania_rl/nn_build/vis_cnn_head.py`` — same flags as classic ``IQN_Network``. BTR-style **head** options (LayerNorm, NoisyNet) on ``iqn_fc`` / A / V use the flat config via ``trackmania_rl/nn_build/iqn_btr_from_config.py``.

2. **HF vision, fusion off** — ``nn.fusion_mode: none`` and ``nn.vis.transformer.use_hf_backbone: true``. ``HfActorCritic`` without policy heads + ``IQNSharedBackboneNetwork`` (same file as above).

3. **Classic CNN or float-only** — ``nn.fusion_mode: none``, ``nn.vis.cnn`` or ``nn.vis.no_image: true``. Class ``IQN_Network`` — CNN image head (or no image), float MLP, concat, then τ-embedding and dueling heads.

**Not supported:** ``nn.vis.transformer`` with **native** ViT (``use_hf_backbone: false``) while ``fusion_mode: none`` — ``build_iqn_network_uncompiled`` raises; use ``fusion_mode`` multimodal or switch to HF ViT.

**Hub warm-start:** for multimodal fusion, ``nn.init_from_pretrained`` is applied automatically on the **PPO** path; IQN does not run the same hook yet (see :ref:`nn-yaml-reference`).

Vision and tensors by topology
--------------------------------

**Classic ``IQN_Network``**

- Inputs: image ``(B, 1, H, W)``, floats ``(B, float_input_dim)``.
- Outputs: ``Q`` with IQN quantiles — single-action ``(B*K, n_actions)`` or multi-action ``(B*K, N, n_actions)`` for ``n_actions_per_block``.

.. graphviz::

   digraph iqn_overview {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      img [label="img\n(B,1,H,W)", style="filled", fillcolor=lightblue];
      flt [label="float_inputs\n(B,F)", style="filled", fillcolor=lightblue];
      cnn [label="Image head\nCNN or IMPALA"];
      mlp [label="Float head\nMLP"];
      cat [label="Concat\n(B,D)"];
      iqn [label="IQN block\nτ-embedding × state"];
      duel [label="Dueling heads\nA + V"];
      out [label="Q-values\n(B*K, A) or (B*K,N,A)", style="filled", fillcolor=lightgreen];

      img -> cnn -> cat;
      flt -> mlp -> cat;
      cat -> iqn -> duel -> out;
   }

**Shared backbone IQN** (multimodal or HF vision)

- Inputs are the same; ``fusion.forward_fusion_hidden(img, float)`` produces a state vector ``(B, D)`` (width ``nn.decoder.dense_hidden_dimension`` after bridge, or ``HfActorCritic.pre_trunk_feature_dim`` on HF path).
- Then the **same** τ cosine embedding, ``iqn_fc``, Hadamard product, and dueling readout as classic IQN (shared implementation).

Core blocks (classic path)
--------------------------

Image branch
~~~~~~~~~~~~

By default IQN uses a 4-layer CNN image head. The BTR option can replace this
head with IMPALA-CNN, but the interface is unchanged: image branch outputs a
flat embedding per sample.

Float branch
~~~~~~~~~~~~

A two-layer MLP transforms normalized scalar features to ``float_hidden_dim``.

Fusion
~~~~~~

Image and float embeddings are concatenated into ``dense_input_dimension``.

IQN quantile module
~~~~~~~~~~~~~~~~~~~

For each sample, IQN draws/supplies ``K`` quantiles ``τ`` and computes:

1. cosine embedding of ``τ`` (dimension ``iqn_embedding_dimension``),
2. projection to state feature width,
3. element-wise multiplication with repeated fused state embedding.

This yields a quantile-conditioned latent representation ``(B*K, D)``.

.. graphviz::

   digraph iqn_tau {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      tau [label="τ\n(B*K,1)", style="filled", fillcolor=lightblue];
      cos [label="cos(pi*i*τ)"];
      fc [label="Linear + activation\n-> (B*K,D)"];
      st [label="state embed\n(B,D)", style="filled", fillcolor=lightblue];
      rep [label="repeat K\n(B*K,D)"];
      mul [label="Hadamard product"];
      out [label="quantile latent\n(B*K,D)", style="filled", fillcolor=lightgreen];
      tau -> cos -> fc -> mul;
      st -> rep -> mul -> out;
   }

Decoder (``nn.decoder``): MLP vs transformer slots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After fusing image + float into a flat vector (or after fusion hidden state), **advantage** and **value** each use a slot that is **either** ``mlp`` **or** ``transformer`` (mutually exclusive per slot):

- **MLP:** hidden width defaults to ``decoder.dense_hidden_dimension // 2`` if ``mlp.hidden_dim`` is omitted; ``n_hidden_layers`` stacks of Linear (+ optional LayerNorm / NoisyLinear from ``btr``).
- **Transformer:** ``IQNTransformerTrunk`` reshapes the flat vector into tokens ``(B, D/d_model, d_model)`` (so the input width must be divisible by ``d_model``), runs ``torch.nn.TransformerEncoder``, mean-pools to ``d_model``, then a small Linear stack to actions or value.

**``decoder.shared_input``:** if either slot uses ``transformer``, validation requires ``shared_input: post_tau``. Classic ``IQN_Network.forward`` fuses image+float first, then applies quantile mixing (**post-τ**). ``pre_tau`` is schema-only for transformer slots today.

Dueling heads
~~~~~~~~~~~~~

``Q(s,a,τ) = V(s,τ) + A(s,a,τ) - mean_a A(s,a,τ)``.

Multi-action mode factorizes by offset; output ``(B*K, N, n_actions)``.

Training flow (high level)
--------------------------

1. Collectors run an inference copy of the network.
2. Learner samples replay; target branch computes quantile targets.
3. Quantile Huber loss; target network soft/hard updates.

Key design notes
----------------

- **Distributional:** return quantiles, not only mean Q.
- **Dueling:** state value + action advantage.
- **DDQN, NoisyNet, multi-action:** see config and :doc:`btr_architecture`.

Implementation references
-------------------------

- ``trackmania_rl/agents/iqn.py`` — ``IQN_Network``, ``build_iqn_network_uncompiled``, trainer, inferer.
- ``trackmania_rl/nn_build/iqn_multimodal.py`` — ``IQNSharedBackboneNetwork``, fusion/HF factories.
- ``trackmania_rl/nn_build/iqn_quantile_forward.py`` — shared τ + dueling forward.
- ``trackmania_rl/nn_build/vis_cnn_head.py`` / ``iqn_btr_from_config.py`` — config → kwargs for shared CNN / BTR head wiring (see :doc:`ppo_architecture` for PPO side).
- ``trackmania_rl/multiprocess/collector_process.py``, ``learner_process.py``.

See also
--------

- :doc:`nn_topology_catalog` — full matrix of supported ``nn`` topologies.
- :doc:`ppo_architecture` — same multimodal / HF bodies with policy heads.
- :doc:`grpo_architecture` — same bodies under ``ppo_wiring``; GRPO-specific training.
- :doc:`btr_architecture` — IQN extras under ``btr:``.
- :doc:`../experiments/models/iqn` — experiments.
- :doc:`../configuration_guide` — full ``nn`` reference.
