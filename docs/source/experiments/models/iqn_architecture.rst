.. _iqn_architecture:

IQN architecture
================

The diagrams on this page are rendered with **Graphviz** (the ``sphinx.ext.graphviz`` extension). The CI workflow that publishes the docs installs Graphviz so the diagrams render on the site. For a local docs build, install Graphviz and ensure ``dot`` is on your PATH, or you will see raw DOT code instead of images.

This page describes the **structure** of **IQN_Network** (``trackmania_rl.agents.iqn``), **how training works** (data flow, replay, loss), and **why** the main design choices were made.

What we use
-----------

- **IQN** — value-based RL with a *distributional* head: we predict **quantiles** of the return (sum of future rewards), not just its expectation. For each (state, action) we get K values (one per quantile τ ∈ (0,1)); their mean is the usual Q(s,a). This often improves sample efficiency and stability (see *Why distributional* below).
- **Discrete actions** — e.g. 12 classes (steer × accel × brake binned), defined in ``config_files/inputs_list.py`` and ``config.inputs``.
- **Inputs** — (1) **image**: one grayscale frame per step, downscaled (e.g. 64×64); (2) **float state**: scalar features (position along track, zone indices, previous actions, etc.), see ``state_normalization`` and ``float_input_dim`` in config.

Training loop (data flow)
-------------------------

1. **Collectors** — Several game instances run in parallel. Each has an **inference_network** (copy of the policy). The agent observes (frame, float_state), chooses an action (ε-greedy or Boltzmann over mean Q), and sends it to the game. Transitions (state, action, reward, next_state, …) are sent to the learner via a queue.
2. **Learner** — One process. It holds the **online_network** (updated by gradients) and the **target_network** (periodically synced with the online network, e.g. soft update with τ=0.02 or hard update every N steps). It also maintains an **uncompiled_shared_network**: the learner copies online → shared; collectors copy shared → their inference_network. So collectors always use a slightly stale but consistent policy.
3. **Replay buffer** — Transitions are stored in a **ReplayBuffer** (e.g. prioritized). Sampling is done in **mini-batches**; each batch is then passed through **buffer_collate_function**, which implements the **mini-race** logic (see *Why mini-races* below).
4. **train_on_batch** — For each batch we compute TD targets using the **target** network and current rewards/gammas; we compute Q(s,a) for the sampled (s,a) using the **online** network; we minimize the quantile Huber loss between targets and outputs, then backprop and update the online network.

**Why separate online and target?** Standard in DQN: the target is held fixed for many steps so the learning signal is stable; otherwise we would be chasing a moving target (bootstrapping from a network we keep changing).

Why distributional (quantiles)
------------------------------

In standard DQN we learn one number Q(s,a) = E[return]. In **IQN** we learn the *distribution* of the return via its **quantiles**: for τ ∈ (0,1), we predict the τ-quantile (e.g. τ=0.1 = pessimistic, τ=0.5 = median). The network is trained with **quantile Huber loss** so that predicted quantiles match the distribution of TD targets.

**Why this helps:** (1) **Richer signal** — the full distribution captures risk and uncertainty. (2) **Better gradient flow** — multiple quantiles provide more learning signal per transition than a single scalar. (3) **Stability** — distributional methods (IQN, QR-DQN, C51) often reduce overestimation and improve convergence.

We use **implicit** quantiles: τ is sampled (or fixed) per forward pass and embedded via cos(π·i·τ); the state representation is repeated K times and mixed with this embedding. Config: ``iqn_n`` (e.g. 8) quantiles during training; ``iqn_k`` (e.g. 32) during inference for action selection (we average over quantiles then choose argmax).

Why dueling (V + A)
-------------------

We decompose **Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))**. The **value** V(s) is shared across all actions; the **advantage** A(s,a) is per action.

**Why:** In many states the value is similar for all actions (e.g. straight road); learning V(s) once is more sample-efficient than learning each Q(s,a) separately. See Dueling DQN (Wang et al., 2016). The subtraction of mean(A) keeps the decomposition unique (otherwise V and A are underdetermined).

Why Double DQN (optional)
-------------------------

Config: ``use_ddqn: true`` (default). In plain DQN the TD target uses the **target** network both to *choose* the best next action and to *evaluate* it → tends to **overestimate** Q. In **Double DQN** we use the **online** network to *choose* the best next action and the **target** network only to *evaluate* that action → usually reduces overestimation.

**In our code:** In ``train_on_batch``, when ``use_ddqn`` is True we take ``a* = argmax_a Q_online(s', a)`` then form the target as ``r + γ Q_target(s', a*)``; when False we use ``r + γ max_a Q_target(s', a)``.

Why mini-races (clipped horizon)
--------------------------------

When we sample a batch, **buffer_collate_function** does the following: (1) For each transition it draws a **random horizon** (in number of actions) up to ``temporal_mini_race_duration_actions`` (e.g. 7 seconds). This horizon is stored in ``state_float[:, 0]`` so the network sees “time left in this mini-race.” (2) **Rewards** and **gammas** are reindexed so that we only sum rewards *up to that horizon*; beyond the horizon we treat the transition as terminal (gamma=0). (3) **Potential-based shaping** is applied: we add (γ φ(s') − φ(s)) to the reward so that the value of progress is preserved without changing optimal policies (Ng et al.).

**Why:** **Credit assignment** — we only ask “how much reward in this short window?”, which simplifies learning. **Gamma = 1 over the window** — we can use γ=1 within the 7s window because the horizon is fixed and short. **Same buffer, different views** — the same transition can be interpreted as different “mini-races” on different samples, which increases diversity. See ``trackmania_rl.buffer_utilities.buffer_collate_function`` and config ``temporal_mini_race_duration_ms``, ``n_steps``, ``gamma_schedule``.

Normalization
-------------

- **Image** — In ``IQN_Network.forward()`` we do ``(img - 128) / 128`` (assuming input in [0, 255]) → approximately [-1, 1]. This matches Level 0 / BC pretraining when ``image_normalization: "iqn"`` is set, so that loading a pretrained encoder into ``img_head`` does not require renormalization.
- **Float state** — We apply ``(float_inputs - mean) / std`` in ``forward()``; ``mean`` and ``std`` come from config (``state_normalization.float_inputs_mean`` and ``float_inputs_std``).

Pretrained encoder (Level 0 / BC)
---------------------------------

The **image head** (CNN) of ``IQN_Network`` has the same architecture as the encoder saved by **Level 0** (autoencoder/SimCLR) and **Level 1 BC** pretraining. We can **load** a pretrained ``encoder.pt`` into ``img_head`` (config: ``pretrain_encoder_path``). BC pretrain additionally trains the same CNN to predict actions from frames; the encoder is then transferred to IQN’s ``img_head``. See :doc:`../pretrain_replay_roadmap` and :doc:`../pretrain_bc`.

Overview: inputs and outputs
----------------------------

This network does **distributional** RL: it models the *distribution* of the return (sum of rewards), not just its expectation. Standard DQN outputs one Q(s,a) = E[return]; here we predict **quantiles** of that distribution — for each τ ∈ (0,1) we get the τ-quantile (e.g. τ=0.1 = pessimistic, τ=0.5 = median). We get K values per (state, action); averaging them gives the usual Q(s,a), but the full set captures uncertainty and often improves learning (IQN, QR-DQN, C51 are distributional methods).

**Quantiles τ** — In IQN, for each τ ∈ (0,1) we predict the τ-quantile of the return distribution (e.g. τ=0.1 = “pessimistic” scenario, τ=0.5 = median, τ=0.9 = “optimistic”). The network is trained to match these quantiles via the quantile Huber loss. So we get K values Q(s,a,τ₁), …, Q(s,a,τₖ) per state and action instead of one; averaging them gives the usual Q(s,a), but the full set captures uncertainty.

**Replication (“repeating” state K times)** — We have one state representation after concat, shape (B, D). For each state we need Q for K different τ. So we *repeat* that representation K times → (B×K, D), and for each of the B×K rows we compute a τ-dependent embedding and mix it (Hadamard) with the repeated state. The result is (B×K, D): one row per (state, quantile). The A and V heads then output Q of shape (B×K, n_actions). So “replication” is: one state → K rows (one per τ) so we get K quantile estimates per state in one forward pass.

**Dueling** — We decompose Q(s,a) = V(s) + A(s,a), where V(s) is the state value and A(s,a) is the advantage of action a (we use Q = V + A - mean(A) so the decomposition is unique). In many states the value is similar across actions; learning V(s) once and small advantages per action is more sample-efficient than learning each Q(s,a) from scratch. See Dueling DQN (Wang et al., 2016).

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

Other implementation details
-----------------------------

- **Prioritized replay** — Optional (``prio_alpha > 0``): transitions are sampled with probability proportional to TD error; importance weights are applied to the loss so that the update remains unbiased.
- **Gradient clipping** — We clip gradients by value (``clip_grad_value``) and by norm (``clip_grad_norm``) to avoid explosions.
- **Target self-loss clamping** — We scale the per-sample loss so that the target’s “self-loss” (target vs target) does not dominate; this stabilizes quantile regression. See ``target_self_loss_clamp_ratio`` and the running averages in ``Trainer.train_on_batch``.
- **Exploration** — At inference we use ε-greedy or Boltzmann over the **mean** of the K quantile outputs (config: ``exploration`` section). So we still act on a single scalar Q per action, but that scalar is the average of the distributional output.

Config parameters
-----------------

Main dimensions are set in ``config_files/rl/config_default.yaml`` (section ``neural_network``):

- **w_downsized**, **h_downsized** — Input frame size for the CNN (e.g. 64×64).
- **float_hidden_dim** — Output size of the float head (256).
- **dense_hidden_dimension** — Hidden size in the A and V heads (1024).
- **iqn_embedding_dimension** — Dimension of the cos-embedding of quantiles (128).
- **iqn_n** — Number of quantiles during training (8); **iqn_k** — during inference (32).

**float_input_dim** is computed at config load time (depends on number of zones, previous actions, etc.). **conv_head_output_dim** is computed from H, W via ``calculate_conv_output_dim()`` in ``iqn.py``.

See also
--------

- :doc:`iqn` — Experiments on IQN variants (DDQN, embedding size, image size).
- :doc:`../../main_objects` — IQN_Network, buffer, rollout_results.
- :doc:`../../first_training` — How to run training and what to expect.
- :doc:`../../configuration_guide` — All config options (neural_network, training, rewards, etc.).
- The ``IQN_Network`` class and ``forward()`` method in ``trackmania_rl.agents.iqn``.
