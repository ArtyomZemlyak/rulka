.. _btr_architecture:

BTR options (IQN + paper extras)
================================

This page describes how BTR is implemented in this project.

The most important point:
**BTR is not a separate architecture here**.
It is IQN (same ``IQN_Network`` and trainer path) plus a set of optional
enhancements configured in the ``btr`` section of the config.

Baseline and composition
------------------------

- Baseline: :doc:`iqn_architecture`
- BTR in this repo: ``IQN + {Munchausen, IMPALA-CNN, AdaptiveMaxPool, SpectralNorm, LayerNorm, NoisyLinear}``

.. graphviz::

   digraph btr_stack {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      iqn [label="IQN baseline\n(image head + float head + IQN + dueling)", style="filled", fillcolor=lightyellow];
      m [label="Munchausen target"];
      i [label="IMPALA-CNN image head"];
      p [label="Adaptive MaxPool"];
      s [label="SpectralNorm (conv)"];
      l [label="LayerNorm (MLP/heads)"];
      n [label="NoisyLinear exploration"];
      out [label="BTR-configured IQN", style="filled", fillcolor=lightgreen];

      iqn -> out;
      m -> out;
      i -> out;
      p -> out;
      s -> out;
      l -> out;
      n -> out;
   }

Where each BTR option is applied
--------------------------------

The table below maps each BTR feature to the implementation location and effect.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - BTR feature
     - Where in code
     - What changes
   * - Munchausen IQN
     - ``Trainer.train_on_batch`` in ``trackmania_rl/agents/iqn.py``
     - Replaces hard-max/DDQN bootstrapped target with soft-policy value and log-policy reward bonus.
   * - IMPALA-CNN
     - ``_build_img_head`` in ``trackmania_rl/agents/iqn.py``
     - Swaps default 4-conv image encoder with IMPALA residual blocks.
   * - Adaptive MaxPool
     - image head builder in ``trackmania_rl/agents/iqn.py``
     - Produces fixed-size spatial output before flatten.
   * - SpectralNorm
     - image head builder in ``trackmania_rl/agents/iqn.py``
     - Wraps convolution layers with spectral normalization.
   * - LayerNorm
     - float extractor and dueling heads in ``trackmania_rl/agents/iqn.py``
     - Adds layer normalization in MLP/heads.
   * - NoisyLinear
     - ``FactorizedNoisyLinear`` and action selection in ``trackmania_rl/agents/iqn.py``
     - Uses trainable parameter noise; when enabled, rollout action logic does not use epsilon/Boltzmann branches.

Config resolution in code
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Vision CNN** (IMPALA, adaptive pool, spectral norm): YAML **canonical** block is ``nn.vis.cnn``; omitted keys can be filled from ``btr:`` at load (``config_loader._merge_btr_cnn_into_vis``). All call sites that build ``_build_img_head`` (classic IQN, PPO CNN, multimodal CNN branch, BC, pretrain) resolve kwargs through ``trackmania_rl/nn_build/vis_cnn_head.py``.
- **LayerNorm / NoisyNet / ``noisy_sigma0``** on IQN MLP heads (classic and shared-backbone IQN): read from the flat loaded config in ``trackmania_rl/nn_build/iqn_btr_from_config.py`` as ``iqn_btr_mlp_head_kw_from_config``.

BTR data flow vs baseline IQN
-----------------------------

At high level, BTR uses the same collector/learner/replay pipeline as IQN.
Differences are in the model blocks and target computation:

.. graphviz::

   digraph btr_flow {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      s [label="State: image + float", style="filled", fillcolor=lightblue];
      enc [label="Encoder\n(default CNN or IMPALA + SN + AdaptivePool)"];
      body [label="IQN quantile fusion\n(+ optional LayerNorm in MLP/heads)"];
      heads [label="Dueling heads\n(+ optional NoisyLinear)"];
      q [label="Q quantiles"];
      tgt [label="Target computation\n(Standard IQN/DDQN OR Munchausen IQN)", style="filled", fillcolor=mistyrose];
      loss [label="Quantile Huber loss"];

      s -> enc -> body -> heads -> q -> tgt -> loss;
   }

Detailed behavior by component
------------------------------

1) Munchausen IQN target
~~~~~~~~~~~~~~~~~~~~~~~~

With ``btr.use_munchausen: true``, training uses the soft-policy target path:

- compute ``log π(a|s)`` from quantile-mean Q and temperature ``munchausen_entropy_tau``;
- add reward bonus ``alpha * tau * clamp(log π(a_t|s_t), lo, 0)``;
- bootstrap with soft value ``V(s') = Σ_a π(a|s') [Q(s',a) - tau*log π(a|s')]``.

This path is implemented for both single-action and multi-action modes.
If Munchausen is off, code falls back to standard DDQN/max target logic.

Why this is useful:
it replaces brittle ``max(Q)`` style targets with a soft-policy target, which
often reduces optimistic spikes and makes updates less jumpy. The bounded
log-policy bonus also keeps learning signal informative when many actions have
similar value.

2) IMPALA-CNN + Adaptive MaxPool + SpectralNorm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These three options modify only the image branch:

- ``nn.vis.cnn.use_impala_cnn`` selects residual IMPALA-style encoder.
- ``nn.vis.cnn.use_adaptive_maxpool`` changes spatial reduction to fixed size.
- ``nn.vis.cnn.use_spectral_norm`` wraps conv layers for spectral normalization.

The rest of IQN pipeline (float branch, quantile fusion, dueling heads,
replay/training loop) remains unchanged.

Why this is useful:
IMPALA usually gives a stronger visual encoder than a small plain conv stack.
Adaptive max-pool makes spatial output size fixed and less sensitive to raw
resolution details. Spectral norm limits sudden activation amplification, which
often improves stability when targets are noisy.

3) LayerNorm in MLP and heads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_layer_norm``, layer normalization is inserted in:

- float feature extractor;
- advantage/value head MLP blocks.

This is a stabilization feature and does not change tensor contracts.

Why this is useful:
LayerNorm reduces hidden-scale drift across training and usually makes
optimization smoother, especially when image and float branches have different
feature scales.

4) NoisyLinear and exploration semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``use_noisy_linear`` is enabled:

- linear layers in dueling heads become factorized noisy layers;
- rollout policy calls ``reset_noise()`` in exploration mode and
  ``disable_noise()`` in eval mode;
- epsilon/Boltzmann branches in action selection are bypassed.

So in this mode exploration is driven by parameter noise in Q-values.
The epsilon schedules can still be present in config/logging, but they are not
used to choose actions in the noisy branch.

Why this is useful:
exploration becomes state-dependent (through noisy parameters) instead of
uniform random action perturbations. In long runs this often preserves useful
exploration better than a fixed epsilon schedule.

.. graphviz::

   digraph noisy_action {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      cfg [label="use_noisy_linear?"];
      noisy [label="reset_noise / disable_noise\nargmax(noisy Q)"];
      eps [label="epsilon-greedy or Boltzmann\n(only when noisy off)"];
      act [label="chosen action/block", style="filled", fillcolor=lightgreen];
      cfg -> noisy [label="yes"];
      cfg -> eps [label="no"];
      noisy -> act;
      eps -> act;
   }

Configuration section
---------------------

**Vision CNN** (canonical): ``nn.vis.cnn`` — ``use_impala_cnn``, ``impala_model_size``,
``use_adaptive_maxpool``, ``adaptive_maxpool_size``, ``use_spectral_norm``.
The loader can copy missing CNN keys from ``btr:`` into ``nn.vis.cnn`` for backward-compatible minimal YAML; **IQN** and **PPO Variant A** read the merged ``nn.vis.cnn``. Multimodal fusion ``post_concat`` still uses its **own** fixed CNN in ``multimodal_torch_fusion.py``.

**BTR-only flags** (under ``btr:`` in YAML, ``BTRConfig`` in code):

- ``use_munchausen``, ``munchausen_alpha``, ``munchausen_entropy_tau``, ``munchausen_lo``
- ``use_layer_norm``
- ``use_noisy_linear``, ``noisy_sigma0``

``BTRConfig`` still lists the CNN fields for schema/merge; prefer setting them on ``nn.vis.cnn`` in new configs to avoid duplication.

Practical recommendations
-------------------------

- Start from IQN defaults and enable BTR features incrementally if you need
  isolated ablations.
- For full BTR-style runs, enable all six features together.
- Keep in mind that some “paper defaults” are environment-specific; for TrackMania,
  schedule timing, gamma strategy, and batch size may need retuning.

See also
--------

- :doc:`iqn_architecture` — baseline model that BTR augments.
- :doc:`../experiments/models/iqn` — IQN experiment pages.
- :doc:`../configuration_guide` — full config reference; YAML trees :ref:`nn-yaml-reference` and :ref:`btr-yaml-reference`.
