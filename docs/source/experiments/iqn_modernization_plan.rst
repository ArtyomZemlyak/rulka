Experiment Note: IQN Modernization Plan
=======================================

Purpose
-------

This page is a research-backed architecture note for modernizing the current IQN agent used in this project.

The goal is **not** to replace the RL setup blindly, but to identify which improvements are:

- well supported by the IQN / distributional RL literature,
- compatible with the current project design,
- likely to help on the current TrackMania setup,
- worth trying first versus later.

Current Project Baseline
------------------------

The current RL agent is implemented in ``trackmania_rl/agents/iqn.py`` and is architecturally:

- grayscale image encoder (small CNN),
- float-feature MLP,
- IQN cosine embedding + multiplicative conditioning,
- dueling value / advantage heads,
- optional multi-action factorized head for block prediction.

Important project-specific constraints:

- RL images are currently **single-frame grayscale** observations.
- A large amount of state information already exists in the float vector:
  - previous actions,
  - velocity,
  - angular velocity,
  - wheel / contact / gearbox signals,
  - future zone centers in car frame,
  - margin-to-finish and related handcrafted signals.
- This means the agent is **not** a pure pixel-only controller. It already receives strong structured dynamics signals.

That constraint matters when evaluating modern architectures: some ideas that are powerful in generic visual RL are less attractive here, while some targeted upgrades become especially promising.

What The Literature Says Around IQN
-----------------------------------

Main line of development
~~~~~~~~~~~~~~~~~~~~~~~~

The most relevant papers form a fairly clear progression:

1. ``QR-DQN``: *Distributional Reinforcement Learning with Quantile Regression*.
   It learns a fixed set of quantile values.

   Link: `QR-DQN <https://arxiv.org/abs/1710.10044>`_

2. ``IQN``: *Implicit Quantile Networks for Distributional Reinforcement Learning*.
   Instead of a fixed quantile grid, it learns a continuous quantile function ``Q(s, a, tau)``.

   Link: `IQN <https://arxiv.org/abs/1806.06923>`_

3. ``FQF``: *Fully Parameterized Quantile Function for Distributional Reinforcement Learning*.
   This is the most direct and important architectural successor to IQN.
   The key idea is to learn **which quantile fractions matter**, instead of sampling ``tau`` from a fixed or random distribution.

   Link: `FQF <https://arxiv.org/abs/1911.02140>`_

4. ``Munchausen RL`` / ``M-IQN``.
   This is a strong practical extension, but it is mostly a target / reward modification rather than a backbone or head redesign.

   Link: `Munchausen RL <https://proceedings.neurips.cc/paper/2020/hash/2c6a0bae0f071cbbf0bb3d5b11d90a82-Abstract.html>`_

5. ``TQC`` and related quantile-critic methods in continuous control.
   These methods keep the quantile-distribution idea, then add truncation and critic ensembles to control overestimation.

   Link: `TQC <https://proceedings.mlr.press/v119/kuznetsov20a.html>`_

Main takeaway:

- The strongest direct evolution of IQN was **not** "replace the encoder with a Transformer".
- The strongest direct evolution was **better parameterization of the return distribution itself**, especially ``FQF``.

Common high-value additions around IQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, IQN-like agents are often improved by combining them with "surrounding" DQN-style architectural or agent-level tricks:

- ``dueling`` heads,
- ``NoisyNet`` / noisy linear layers,
- ``n-step`` returns,
- ``prioritized replay``,
- recurrence for partial observability,
- image augmentation for pixel-based RL.

Some of these are not unique to IQN, but they repeatedly show up because they complement quantile-based value learning well.

Key references:

- `Dueling Network Architectures <https://ar5iv.labs.arxiv.org/html/1511.06581>`_
- `Noisy Networks for Exploration <https://arxiv.org/abs/1706.10295>`_
- `R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning <https://openreview.net/forum?id=r1lyTjAqYX>`_
- `DrQ: Image Augmentation Is All You Need <https://arxiv.org/abs/2004.13649>`_
- `GTrXL: Stabilizing Transformers for Reinforcement Learning <https://arxiv.org/abs/1910.06764>`_

What Looks Most Promising For This Project
------------------------------------------

Below is a project-specific reading of the literature for the current TrackMania setup.

1. Stronger IQN trunk without changing the RL principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These changes keep the general training logic, replay structure, and input format mostly intact.

Recommended upgrades:

- Replace the current image head with a **small residual CNN** rather than a plain shallow conv stack.
  - Prefer a lightweight IMPALA-style or small-ResNet-style encoder.
  - This is a better first bet than a ViT for ``64x64`` grayscale inputs.
- Strengthen the post-fusion latent trunk:
  - one or two residual MLP blocks after image+float fusion,
  - add ``LayerNorm`` or ``RMSNorm``,
  - keep the model compact enough for many collectors and large batches.
- Improve quantile conditioning:
  - deeper ``iqn_fc`` projection,
  - scale-and-shift / FiLM-style conditioning instead of only multiplicative modulation,
  - optional repeated quantile conditioning in more than one trunk stage.

Why this is attractive here:

- It preserves the basic IQN training setup.
- It addresses likely under-capacity in representation learning.
- It avoids the engineering cost of sequence replay or large foundation backbones.

2. Multi-action head redesign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this project specifically, the multi-action path is a high-value architecture target.

Current intuition:

- Predicting a block of future actions with a very compact factorized head is elegant and fast.
- However, neighboring offsets in a block are strongly coupled in driving tasks.
- A very shallow fused output may under-model that structure.

Promising directions:

- add **offset embeddings** for each future action slot,
- predict **per-offset value terms** instead of one shared scalar value,
- add a lightweight **offset mixer** between shared latent and final per-offset logits:
  - tiny MLP per slot,
  - 1D convolution over offsets,
  - GRU over offsets,
  - very small self-attention over offsets.

Expected upside:

- better coordination between near-future actions,
- improved use of the multi-action decision block,
- more direct benefit to the current project than replacing the whole model family.

3. FQF as the most direct IQN upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the goal is: "improve IQN as IQN", then ``FQF`` is the first paper to study seriously.

Why it matters:

- IQN learns quantile values but samples ``tau`` from a predefined distribution.
- FQF learns both:
  - the quantile values,
  - the quantile fractions on the probability axis.

In other words:

- ``QR-DQN`` fixed the fraction axis and learned values,
- ``IQN`` sampled the fraction axis and learned a continuous value function,
- ``FQF`` learned both axes.

Why it is attractive:

- It is the most research-grounded upgrade if the target is the distribution head itself.
- It is more "on theme" than switching to a generic bigger backbone.

Why it is not the first low-risk experiment:

- It changes more of the training logic than a trunk-only refactor.
- It requires implementing the fraction proposal network and its associated losses.

Recommendation:

- Treat ``FQF`` as the top **medium-risk / high-upside** modernization path.

4. Recurrent memory: try LSTM/GRU before Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model is limited by partial observability, recurrence is more compelling than a Transformer-first approach.

Why:

- RL observations are currently single-frame on the image side.
- Some temporal information exists in float inputs, especially previous actions and kinematic signals.
- If that still proves insufficient, the next natural step is a recurrent latent module.

Recommended ordering:

1. ``CNN/float encoder -> fused latent -> GRU/LSTM -> IQN heads``
2. Only later consider Transformer memory if recurrence is clearly insufficient

Why not Transformer first:

- higher engineering and tuning cost,
- replay + sequence handling becomes harder,
- benefits are strongest when long-horizon memory is clearly the bottleneck,
- the current project already injects useful short-horizon dynamics through float inputs.

Literature note:

- ``R2D2`` is the practical value-based reference for recurrent replay.
- ``GTrXL`` is the most relevant Transformer-for-RL reference, but it does **not** imply that a Transformer is the best first modernization for this project.

5. Visual backbone strategy: prefer in-domain pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project already has a solid foundation for visual and BC pretraining. That is a stronger fit than dropping in a generic frozen foundation model.

Recommended visual strategy:

- First preference:
  - keep an IQN-compatible or near-compatible small CNN,
  - pretrain it on project-specific data,
  - transfer the encoder into RL.
- Second preference:
  - experiment with a lightweight mobile CNN if runtime or throughput becomes critical.
- Lower priority:
  - frozen generic encoders such as DINOv2-style features.

Why generic foundation backbones are not the first recommendation here:

- inputs are grayscale and low-resolution,
- the domain is highly specialized,
- transfer from generic natural-image representations is less certain,
- the project already benefits from in-domain pretraining paths.

Relevant vision reference:

- `DINOv2 <https://arxiv.org/abs/2304.07193>`_

What Probably Matters More Than A Fancy Backbone
------------------------------------------------

The literature around visual RL repeatedly suggests that some simple methods can outperform larger architecture changes.

High-value examples:

- image augmentation / random crop regularization,
- better exploration parameterization (for example noisy linear layers),
- more stable latent trunk design,
- recurrence when memory is truly needed,
- better quantile parameterization (FQF).

For this project, these may matter more than switching to a ViT or a large pretrained vision encoder.

Recommended Roadmap
-------------------

Tier 1: Best first experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the most practical first modernization steps.

1. **Stronger residual IQN trunk**
   - small residual CNN,
   - stronger fused MLP trunk,
   - norm layers,
   - improved quantile conditioning.

2. **Multi-action head redesign**
   - offset embeddings,
   - per-offset value or richer shared-to-slot mapping,
   - lightweight offset mixer.

3. **Noisy linear layers in the value / advantage heads**
   - especially attractive if exploration remains a bottleneck.

4. **Enable and evaluate stronger image augmentation**
   - especially if generalization or sample efficiency from pixels is limiting progress.

Tier 2: High-upside, moderate engineering cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5. **FQF-style learned quantile fractions**
   - best direct successor to IQN,
   - strongest literature-grounded upgrade of the quantile head itself.

6. **Project-specific visual pretraining with the chosen RL backbone**
   - especially useful if representation quality appears to cap performance.

Tier 3: Only after evidence of memory limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7. **GRU/LSTM over fused latent**
   - sequence replay and burn-in likely needed,
   - best next step when partial observability is clearly limiting.

8. **Transformer / GTrXL-style memory**
   - interesting research path,
   - but not the recommended first modernization for this project.

Summary Recommendations
-----------------------

If the project wants the most research-supported, project-compatible plan, the order should be:

1. strengthen the current IQN trunk,
2. redesign the multi-action head,
3. add cheap high-value tricks such as noisy layers and stronger augmentation,
4. consider ``FQF`` as the main "true IQN successor" experiment,
5. add recurrent memory before trying Transformers,
6. prefer in-domain pretraining over generic frozen visual backbones.

Short version
~~~~~~~~~~~~~

- **Best direct IQN upgrade from the literature:** ``FQF``.
- **Best practical upgrade for this project right now:** stronger trunk + better multi-action head.
- **Best temporal upgrade if needed:** ``GRU/LSTM`` before Transformer.
- **Best visual strategy:** small project-aligned backbone + in-domain pretraining.
- **Least convincing first move:** replacing everything with a generic Transformer or large frozen foundation encoder.

Suggested Follow-up Experiments
-------------------------------

If this page is turned into an implementation roadmap, a reasonable first batch is:

- ``iqn_residual_trunk_v1``
- ``iqn_multi_action_mixer_v1``
- ``iqn_noisy_heads_v1``
- ``iqn_augmented_pixels_v1``
- ``fqf_trackmania_v1``
- ``iqn_gru_latent_v1``

These names are only placeholders, but they reflect a sensible experimentation order from low-risk to higher-risk changes.
