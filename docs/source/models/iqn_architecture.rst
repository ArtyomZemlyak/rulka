.. _iqn_architecture:

IQN architecture
================

This page documents the baseline model used in the project:
``trackmania_rl.agents.iqn.IQN_Network``.

For the BTR variant, see :doc:`btr_architecture`.
Important: BTR in this codebase is **not a separate model family**. It is a
set of optional additions on top of this same IQN architecture.

Overview
--------

The network consumes two input branches:

- image input ``(B, 1, H, W)`` (grayscale frame);
- float features ``(B, float_input_dim)`` (state vector).

It produces distributional Q-values via IQN quantiles:

- single-action mode: ``Q: (B*K, n_actions)``;
- multi-action mode: ``Q: (B*K, N, n_actions)`` where ``N`` is
  ``n_actions_per_block`` from ``rl_action_offsets_ms``.

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

Core blocks
-----------

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
2. projection to ``dense_input_dimension``,
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

Dueling heads
~~~~~~~~~~~~~

The model uses dueling decomposition:

``Q(s,a,τ) = V(s,τ) + A(s,a,τ) - mean_a A(s,a,τ)``.

In multi-action mode, the action head is factorized by future offset and output
is shaped ``(B*K, N, n_actions)``.

Training flow (high level)
--------------------------

1. Collectors generate transitions with an inference copy of the network.
2. Learner samples replay batches (optionally prioritized).
3. Target branch computes bootstrapped quantile targets.
4. Online branch computes current quantile values for sampled actions.
5. Quantile Huber loss is minimized; online weights are updated.
6. Target network is periodically updated from online network.

Key design notes
----------------

- **Distributional learning (IQN):** predicts return quantiles, not only mean Q.
- **Dueling:** separates state value and action advantage.
- **Optional DDQN target selection:** controlled by ``use_ddqn``.
- **Optional multi-action prediction:** controlled by ``rl_action_offsets_ms``.

Implementation references
-------------------------

- ``trackmania_rl/agents/iqn.py`` — model, forward pass, action selection, trainer.
- ``trackmania_rl/multiprocess/collector_process.py`` — rollout-time inference.
- ``trackmania_rl/multiprocess/learner_process.py`` — training loop and target updates.

See also
--------

- :doc:`btr_architecture` — BTR additions over this IQN backbone.
- :doc:`../experiments/models/iqn` — IQN experiment results and ablations.
- :doc:`../configuration_guide` — config parameters for IQN and training.
