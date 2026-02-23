.. _iqn_architecture:

IQN (Implicit Quantile Network) Architecture
===========================================

The diagrams on this page are built with **Graphviz** (the ``sphinx.ext.graphviz`` extension). Building the docs requires Graphviz to be installed with the ``dot`` executable on your PATH.

This page describes the structure of **IQN_Network** (``trackmania_rl.agents.iqn``): what goes in as input, how the blocks are built, and what is produced as output.

Overview: inputs and outputs
----------------------------

**Inputs:**

- **img** — Screen image tensor: shape ``(batch_size, 1, H, W)``, dtype float32/float16. Values are normalized in ``forward()`` as ``(img - 128) / 128`` (if given as uint8, normalization is done in ``Inferer``).
- **float_inputs** — Vector of scalar state features (position, zones, previous actions, etc.): shape ``(batch_size, float_input_dim)``. Normalized in ``forward()`` as ``(float_inputs - mean) / std`` from config.
- **num_quantiles** — Number of quantiles (N or N' in the IQN paper), e.g. 8 during training.
- **tau** (optional) — Tensor of quantiles with shape ``(batch_size * num_quantiles, 1)``. If not provided, quantiles are sampled inside the network (symmetrically around 0.5).

**Outputs:**

- **Q** — Q-values for each (state, quantile): shape ``(batch_size * num_quantiles, n_actions)``.
- **tau** — The quantiles used: shape ``(batch_size * num_quantiles, 1)``.

The network uses a **dueling** layout: from a single shared representation it computes value V and advantages A, then Q = V + A - mean(A).

High-level diagram (main blocks)
--------------------------------

Below is a block diagram of the main components only: what enters the network and how data flows to the Q output.

.. graphviz::

   digraph iqn_high_level {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=11];
      edge [fontname="Helvetica", fontsize=10];

      subgraph inputs {
        node [fillcolor=lightblue, style="filled"];
        img [label="img\n(B, 1, H, W)"];
        floats [label="float_inputs\n(B, float_input_dim)"];
        tau_in [label="τ (quantiles)\noptional"];
      }

      subgraph backbone {
        node [fillcolor=lightyellow, style="filled"];
        img_head [label="Image head\n(CNN)"];
        float_head [label="Float head\n(MLP)"];
        concat [label="Concat"];
        iqn_block [label="IQN: τ-embed × concat"];
        dueling [label="Dueling\nA_head + V_head"];
      }

      subgraph outputs {
        node [fillcolor=lightgreen, style="filled"];
        Q_out [label="Q, τ\n(B×K, n_actions)"];
      }

      img -> img_head;
      floats -> float_head;
      img_head -> concat [label="conv_out"];
      float_head -> concat [label="float_hidden"];
      concat -> iqn_block;
      tau_in -> iqn_block [style=dashed];
      iqn_block -> dueling;
      dueling -> Q_out;
   }

- **Image head** — CNN over the frame; outputs one vector per sample (details below).
- **Float head** — Two-layer MLP over scalar features; output size matches the float branch dimension (details below).
- **Concat** — Concatenation of the two heads’ outputs along the last axis; dimension = ``conv_head_output_dim + float_hidden_dim`` (this is ``dense_input_dimension`` in the code).
- **IQN block** — Quantiles τ are turned into an embedding (cos + linear layer), then element-wise (Hadamard) product with the repeated concat; output shape ``(B×K, dense_input_dimension)`` (details below).
- **Dueling** — From this representation the advantage head A and value head V are computed, then Q = V + A - mean(A) (details below).

Block details
-------------

Image head (CNN)
~~~~~~~~~~~~~~~

Four convolutions with LeakyReLU and Flatten. Channel sizes: 1 → 16 → 32 → 64 → 32. Output is one vector per sample (size depends on H, W; set in config via ``w_downsized``, ``h_downsized``, e.g. 64×64).

.. graphviz::

   digraph img_head {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      in [label="(B, 1, H, W)", fillcolor=lightblue, style="filled"];
      c1 [label="Conv2d 4×4, s=2\n1→16"];
      c2 [label="Conv2d 4×4, s=2\n16→32"];
      c3 [label="Conv2d 3×3, s=2\n32→64"];
      c4 [label="Conv2d 3×3, s=1\n64→32"];
      flat [label="Flatten"];
      out [label="(B, conv_head_output_dim)", fillcolor=lightgreen, style="filled"];
      in -> c1 -> c2 -> c3 -> c4 -> flat -> out;
   }

Each Conv2d is followed by LeakyReLU (inplace). Weights are initialized orthogonally with the appropriate gain for LeakyReLU.

Float head (MLP)
~~~~~~~~~~~~~~~~

Two linear layers with LeakyReLU. Input normalization (mean/std) is applied in ``forward()`` before this head.

.. graphviz::

   digraph float_head {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      in [label="(B, float_input_dim)\nnormalized", fillcolor=lightblue, style="filled"];
      l1 [label="Linear → float_hidden_dim"];
      l2 [label="Linear → float_hidden_dim"];
      out [label="(B, float_hidden_dim)", fillcolor=lightgreen, style="filled"];
      in -> l1 -> l2 -> out;
   }

Config parameter: ``float_hidden_dim`` (e.g. 256). LeakyReLU after each Linear.

IQN: quantile embedding and mixing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantiles τ (shape ``(B×K, 1)``) are mapped to an embedding using the IQN formula: ``cos(π · i · τ)`` for i = 1..iqn_embedding_dimension, then one linear layer + LeakyReLU to dimension ``dense_input_dimension``. The state vector (concat) is repeated K times (one per quantile), then multiplied by the quantile embedding (Hadamard). The result is a representation of shape ``(B×K, dense_input_dimension)`` that depends on both state and τ.

.. graphviz::

   digraph iqn_detail {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      tau [label="τ (B×K, 1)", fillcolor=lightblue, style="filled"];
      concat_in [label="concat (B, D)", fillcolor=lightblue, style="filled"];
      cos [label="cos(π·i·τ)\n(B×K, iqn_embed_dim)"];
      fc [label="Linear + LeakyReLU\n→ (B×K, D)"];
      repeat [label="repeat K times\n(B×K, D)"];
      hadamard [label="× (element-wise)"];
      out [label="(B×K, D)", fillcolor=lightgreen, style="filled"];
      tau -> cos -> fc;
      concat_in -> repeat;
      fc -> hadamard;
      repeat -> hadamard;
      hadamard -> out;
   }

Dueling: A_head and V_head
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the shared representation ``(B×K, dense_input_dimension)`` two heads are computed:

- **A_head**: Linear(D → dense_hidden_dimension//2) → LeakyReLU → Linear → ``(B×K, n_actions)``.
- **V_head**: Linear(D → dense_hidden_dimension//2) → LeakyReLU → Linear → ``(B×K, 1)``.

Then: ``Q = V + A - mean(A, dim=actions)``. This yields Q-values for all actions and all quantiles.

.. graphviz::

   digraph dueling_detail {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      in [label="(B×K, D)\nstate × quantile", fillcolor=lightyellow, style="filled"];
      a1 [label="Linear D→512"];
      a2 [label="Linear 512→n_actions"];
      v1 [label="Linear D→512"];
      v2 [label="Linear 512→1"];
      A [label="A (B×K, n_actions)"];
      V [label="V (B×K, 1)"];
      Q [label="Q = V + A - mean(A)", fillcolor=lightgreen, style="filled"];
      in -> a1 -> a2 -> A;
      in -> v1 -> v2 -> V;
      A -> Q;
      V -> Q;
   }

Config parameter ``dense_hidden_dimension`` (e.g. 1024); then the inner layer of the heads is 512. The final layers of A_head and V_head are initialized orthogonally without extra gain.

Config parameters
-----------------

Main dimensions are set in ``config_default.yaml`` (section ``neural_network``):

- **w_downsized**, **h_downsized** — Input frame size for the CNN (e.g. 64×64).
- **float_hidden_dim** — Output size of the float head (256).
- **dense_hidden_dimension** — Hidden size in the A and V heads (1024).
- **iqn_embedding_dimension** — Dimension of the cos-embedding of quantiles (128).
- **iqn_n** — Number of quantiles during training (8); **iqn_k** — during inference (32).

**float_input_dim** is computed at config load time (depends on number of zones, previous actions, etc.). **conv_head_output_dim** is computed from H, W via ``calculate_conv_output_dim()`` in ``iqn.py``.

See also
--------

- :doc:`iqn` — Experiments on IQN variants (DDQN, embedding size, image size).
- The ``IQN_Network`` class and ``forward()`` method in ``trackmania_rl.agents.iqn``.
