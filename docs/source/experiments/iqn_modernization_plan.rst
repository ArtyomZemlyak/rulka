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

- grayscale image encoder (small 4-layer CNN: channels ``[1, 16, 32, 64, 32]``, LeakyReLU, no normalization, ``Flatten`` output),
- float-feature MLP (2 linear layers, 256 hidden, LeakyReLU),
- IQN cosine embedding + purely multiplicative Hadamard conditioning,
- dueling value / advantage heads,
- optional multi-action factorized head for block prediction (single shared linear reshaped to ``N × n_actions``).

Important project-specific constraints:

- RL images are currently **single-frame grayscale** ``64×64`` observations.
- A large amount of state information already exists in the float vector (~264 dims):
  - previous actions (25 steps × 4),
  - velocity, angular velocity,
  - wheel / contact / gearbox signals,
  - future zone centers in car frame (40 × 3),
  - margin-to-finish and related handcrafted signals.
- This means the agent is **not** a pure pixel-only controller. It already receives strong structured dynamics signals.

That constraint matters when evaluating modern architectures: some ideas that are powerful in generic visual RL are less attractive here, while some targeted upgrades become especially promising.

Current training configuration of note:

- **Prioritized replay is disabled** (``prio_alpha = 0``, uniform sampling).
- **Image augmentation is disabled** (``apply_randomcrop_augmentation = false``), though random-crop is already implemented.
- Exploration: epsilon-greedy + Boltzmann noise, no NoisyNets.
- N-step returns: 3.
- Discount schedule: ``γ = 0.999 → 1.0``.
- Batch size: 4096.
- Double DQN: enabled.

These disabled-but-implemented features represent low-hanging fruit before any architecture changes.

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
   Extends IQN by learning **which quantile fractions matter** via a fraction proposal network,
   instead of sampling ``tau`` from a fixed or random distribution.

   Link: `FQF <https://arxiv.org/abs/1911.02140>`_

4. ``Munchausen RL`` / ``M-IQN``.
   Adds a scaled log-policy term to the reward in bootstrapped targets.
   Converts DQN to Soft-DQN and uses soft expectation instead of hard max for next-state values.
   Despite being "just" a target modification, it has outsized practical impact on training stability,
   action gaps, and policy churn. **Used in BTR (ICML 2025), the current non-recurrent SOTA on Atari.**

   Link: `Munchausen RL <https://proceedings.neurips.cc/paper/2020/hash/2c6a0bae0f071cbbf0bb3d5b11d90a82-Abstract.html>`_

5. ``Non-Crossing Quantile Networks`` (NQ-Networks, 2025).
   Address the quantile crossing problem that affects both IQN and FQF: predicted quantile values
   can violate monotonicity (i.e. ``Q(tau_1) > Q(tau_2)`` for ``tau_1 < tau_2``).
   NQ-Networks use non-negative activation functions to guarantee monotonic distributions.

   Link: `NQ-Networks <https://arxiv.org/abs/2504.08215>`_

6. ``TQC`` and related quantile-critic methods in continuous control.
   These methods keep the quantile-distribution idea, then add truncation and critic ensembles to control overestimation.

   Link: `TQC <https://proceedings.mlr.press/v119/kuznetsov20a.html>`_

Main takeaway:

- The strongest direct evolution of IQN was **not** "replace the encoder with a Transformer".
- The strongest **practical** evolution was combining IQN with Munchausen targets, as proven by BTR (ICML 2025).
- The strongest **theoretical** evolution of the distribution head was FQF (learned fractions) and later NQ-Networks (monotonicity guarantees).

Key modern reference: BTR (Beyond The Rainbow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BTR`` (ICML 2025) is the most important recent reference for this project. It achieves non-recurrent SOTA on Atari-60
(IQM 7.4) on a single desktop PC in 12 hours, and successfully trains agents for Mario Kart, Super Mario Galaxy,
and Mortal Kombat.

BTR's recipe is: **IQN + Munchausen + IMPALA-CNN (scale=2) + Adaptive MaxPooling + Spectral Normalization +
NoisyNets + Dueling + N-step + Prioritized Replay + Vectorized Environments**.

BTR ablation highlights (on Atari Phoenix):

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15 15

   * - Metric
     - Full BTR
     - w/o Munchausen
     - w/o IQN
     - w/o Spectral Norm
     - w/o IMPALA
   * - Action Gap
     - 0.282
     - 0.055
     - 0.180
     - 0.274
     - 0.215
   * - Action Swaps %
     - 36.6%
     - 47.7%
     - 42.2%
     - 40.3%
     - 41.1%
   * - Policy Churn %
     - 3.8%
     - 11.0%
     - 0.5%
     - 3.3%
     - 4.5%
   * - Score (ε=0)
     - 330k
     - 184k
     - 187k
     - 296k
     - 21k

Key observations:

- **IMPALA encoder** had the largest single impact (+142% IQM).
- **Munchausen** had the largest impact on training stability: without it, action gap collapses 5×, policy churn triples.
- **Spectral Normalization** improves robustness to observation noise.
- BTR does **not** use FQF — plain IQN with Munchausen was sufficient.
- BTR later found **Layer Normalization** on dense layers to be additionally beneficial (Appendix H of the paper).

Link: `BTR <https://arxiv.org/abs/2411.03820>`_

Common high-value additions around IQN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, IQN-like agents are often improved by combining them with "surrounding" DQN-style architectural or agent-level tricks:

- ``dueling`` heads,
- ``NoisyNet`` / noisy linear layers,
- ``n-step`` returns,
- ``prioritized replay``,
- ``Munchausen`` targets,
- ``spectral normalization`` on convolutional layers,
- ``layer normalization`` on dense layers,
- image augmentation for pixel-based RL,
- recurrence for partial observability.

Some of these are not unique to IQN, but they repeatedly show up because they complement quantile-based value learning well.

Key references:

- `Dueling Network Architectures <https://ar5iv.labs.arxiv.org/html/1511.06581>`_
- `Noisy Networks for Exploration <https://arxiv.org/abs/1706.10295>`_
- `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_
- `R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning <https://openreview.net/forum?id=r1lyTjAqYX>`_
- `DrQ: Image Augmentation Is All You Need <https://arxiv.org/abs/2004.13649>`_
- `GTrXL: Stabilizing Transformers for Reinforcement Learning <https://arxiv.org/abs/1910.06764>`_
- `BTR: Beyond The Rainbow <https://arxiv.org/abs/2411.03820>`_
- `Impoola: Average Pooling for Image-Based Deep RL <https://arxiv.org/abs/2503.05546>`_

What Looks Most Promising For This Project
------------------------------------------

Below is a project-specific reading of the literature for the current TrackMania setup.

1. Munchausen IQN (M-IQN): highest-ROI single change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Munchausen RL modifies the bootstrapped target by:

- converting Q-values to a soft policy via softmax with temperature ``τ``,
- adding ``α · τ · ln(π(a_t | s_t))`` to the reward (scaled log-policy of the **taken** action),
- using soft expectation ``Σ_a' π(a'|s') · (Q(s',a') - τ·ln(π(a'|s')))`` instead of hard max for next-state value.

This implicitly introduces KL regularization and entropy bonuses without requiring a separate entropy term.

Why this is the top recommendation for this project:

- **Proven at scale:** BTR (ICML 2025, SOTA on Atari) uses exactly IQN + Munchausen.
- **Massive stability gain:** BTR ablation shows 5× larger action gaps and 3× lower policy churn with Munchausen.
- **Minimal code change:** ~50 lines of modification to the target computation in ``train_on_batch``.
  The network architecture does not change at all.
- **Replaces Double DQN:** Munchausen uses soft expectation, making the ``argmax``-based Double DQN obsolete.
  This simplifies the target computation (one forward pass instead of two).
- **Complements NoisyNets:** Munchausen provides implicit exploration via entropy regularization;
  combined with NoisyNets, this covers both policy-level and parameter-level exploration.

Implementation sketch:

.. code-block:: python

   # In train_on_batch target computation:
   tau_munch = 0.03   # softmax temperature
   alpha_munch = 0.9  # Munchausen scaling

   # Soft policy from target network Q-values
   log_pi_next = F.log_softmax(q_next_target / tau_munch, dim=-1)
   pi_next = log_pi_next.exp()

   # Soft expectation for next-state value (replaces max / DDQN argmax)
   v_next = (pi_next * (q_next_target - tau_munch * log_pi_next)).sum(dim=-1)

   # Munchausen bonus: log-policy of the action actually taken
   log_pi_current = F.log_softmax(q_current_target / tau_munch, dim=-1)
   munch_bonus = alpha_munch * tau_munch * log_pi_current.gather(1, actions)
   munch_bonus = munch_bonus.clamp(min=-1.0)  # stability clamp

   target = rewards + munch_bonus + gamma * v_next

Risk: **very low**. The network, replay, and input pipeline are untouched. Only the target computation changes.

2. Stronger IQN trunk without changing the RL principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These changes keep the general training logic, replay structure, and input format mostly intact.

Recommended upgrades:

a) **IMPALA-style residual CNN** replacing the current plain 4-layer conv stack.

   - The IMPALA encoder is a 15-layer ResNet with 3 residual blocks (conv → max-pool → 2× [conv → conv + skip]).
   - BTR uses IMPALA with width scale=2, which was their single most impactful component (+142% IQM).
   - ``Impoola`` (2025) further showed that replacing ``Flatten`` with **Global Average Pooling** improves generalization.
   - BTR uses **Adaptive Max Pooling (6×6)** after conv layers, reducing parameters by 77% and decoupling from input resolution.

   Recommended: IMPALA-CNN (scale=1 or 2) + Adaptive Pooling (e.g. 4×4 or 6×6) instead of Flatten.

b) **Spectral Normalization** on all convolutional layers.

   - Normalizes weight matrices by their largest singular value, controlling the Lipschitz constant.
   - BTR shows improved training stability and robustness to observation noise.
   - Particularly important when scaling to larger networks (prevents instability from increased capacity).

   Link: `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_

c) **Layer Normalization** on dense layers (after fusion, between hidden layers, before heads).

   - BTR discovered this to be beneficial after initial publication (Appendix H).
   - Standard practice in modern RL: BTR best practices recommend LayerNorm after the stem of each
     residual block and between dense layers.

d) **Stronger post-fusion MLP trunk:**

   - One or two residual MLP blocks after image+float concatenation.
   - ``LayerNorm → Linear → activation → Linear + skip``.
   - Keeps the model compact enough for large batches and many collectors.

e) **Improved quantile conditioning:**

   - Deeper ``iqn_fc`` projection (2 layers instead of 1).
   - Scale-and-shift / FiLM-style conditioning instead of purely multiplicative modulation.
     Current: ``z = state_embedding * quantile_net``.
     FiLM: ``z = quantile_gamma * state_embedding + quantile_beta``.
   - The current weight initialization already compensates for cosine variance (``√2 × gain``),
     suggesting the multiplicative-only path was known to be lossy.

Why this is attractive here:

- It preserves the basic IQN training setup.
- It addresses likely under-capacity in representation learning.
- It avoids the engineering cost of sequence replay or large foundation backbones.
- Each sub-component (a-e) can be tested independently.

3. Enable already-implemented features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project already has code for features that are currently disabled. These are essentially free experiments:

a) **Prioritized Experience Replay** (``prio_alpha``).

   Currently ``prio_alpha = 0`` (uniform sampling). PER is a core component of both Rainbow and BTR.
   BTR uses ``α = 0.2`` (recommended by Toromanoff et al. 2019 specifically for IQN).

   Recommended: test with ``prio_alpha = 0.2``, then ``0.4``, ``0.6``.

b) **Random crop augmentation** (``apply_randomcrop_augmentation``).

   Currently disabled. Random crop / random shift is one of the cheapest and most effective
   regularizers in visual RL (DrQ, SVEA, BTR all use it). BTR uses 4-pixel padding with 84×84 crops.
   The project already implements 2-pixel padding with random crop.

   Recommended: enable with current settings, then experiment with larger crop ranges.

4. Noisy linear layers
~~~~~~~~~~~~~~~~~~~~~~

Replace standard ``Linear`` layers in the value and advantage heads with ``NoisyLinear`` layers (factorized Gaussian noise).

Why:

- The current exploration uses epsilon-greedy + Boltzmann noise, which is state-independent.
- NoisyNets provide **state-dependent exploration**: the noise magnitude is learned per-parameter.
- BTR, Rainbow, and most high-performing value-based agents use NoisyNets.
- Combined with Munchausen (which provides policy-level entropy regularization), this covers
  both parametric and policy-level exploration.

Risk: **low**. Drop-in replacement for ``nn.Linear`` in V/A heads. No change to training loop.

Link: `Noisy Networks for Exploration <https://arxiv.org/abs/1706.10295>`_

5. Multi-action head redesign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this project specifically, the multi-action path is a high-value architecture target.

Current situation:

- The multi-action head uses a single shared ``Linear(a_head_hidden, N × n_actions)`` reshaped to ``(batch, N, n_actions)``.
- Neighboring offsets in a driving block are strongly coupled (steering at t+10ms affects what's optimal at t+20ms).
- The fully-factorized output may under-model this temporal coupling.

Promising directions:

- add **offset embeddings** (learned positional encoding for each future action slot),
- predict **per-offset value terms** instead of one shared scalar value,
- add a lightweight **offset mixer** between shared latent and final per-offset logits:
  - tiny MLP per slot,
  - 1D convolution over the offset dimension,
  - GRU over offsets (captures temporal ordering),
  - very small self-attention over offsets (captures pairwise interactions).

Expected upside:

- better coordination between near-future actions,
- improved use of the multi-action decision block,
- more direct benefit to the current project than replacing the whole model family.

6. Distribution head upgrades: M-IQN first, then FQF or NQ-Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plan for improving the distribution head itself should follow this priority:

**Step 1: M-IQN** (Section 1 above). This is the highest-priority change overall.

**Step 2 (if needed): Non-Crossing Quantile Networks (NQ-Networks).**

Both IQN and FQF suffer from **quantile crossing**: the network can predict ``Q(tau=0.3) > Q(tau=0.7)``
for the same state-action pair, violating the monotonicity of the CDF. This is a real issue that causes
inconsistent value estimates and noisy gradients.

NQ-Networks (2025) solve this by constraining the quantile function to be monotonic via non-negative activations.
This is architecturally simpler than FQF (no separate fraction proposal network) while addressing a more
fundamental problem.

Link: `NQ-Networks <https://arxiv.org/abs/2504.08215>`_

**Step 3 (research experiment): FQF.**

FQF learns which quantile fractions matter via a fraction proposal network trained with the 1-Wasserstein gradient.
This is theoretically elegant, but:

- BTR (SOTA) does not use FQF — plain IQN + Munchausen was sufficient.
- FQF requires a second small network and a separate loss, adding training complexity.
- The gains over IQN on Atari were modest in the original paper.

FQF remains an interesting research experiment, but is no longer the top recommendation for the distribution head.

7. Plasticity maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~

Long training runs (millions of steps) are susceptible to **plasticity loss**: the network gradually loses its ability
to learn from new data. This is increasingly recognized as a major issue in deep RL (Abbas et al. 2023, Lyle et al. 2024).

Symptoms: rising dormant neuron count, increasing weight norms, decreasing effective rank of activations.

BTR's ablation shows that their component combination (especially Spectral Norm + IMPALA) keeps dormant neurons low
and weight norms stable. Several additional mitigation strategies are worth considering:

- **Spectral Normalization** (already recommended above) — controls weight growth.
- **Layer Normalization** (already recommended above) — stabilizes activations.
- **CReLU (Concatenated ReLU)** — ``CReLU(x) = [ReLU(x), ReLU(-x)]`` — doubles width but eliminates
  the dying neuron problem. Shown effective in Abbas et al. 2023.
- **Monitor dormant neurons and effective rank** during training (add to tensorboard metrics).
- **Periodic soft reset** (shrink-and-perturb) as a last resort if metrics show clear plasticity degradation.

Key references:

- `Loss of Plasticity in Continual Deep RL <https://proceedings.mlr.press/v232/abbas23a.html>`_
- `Understanding Plasticity in Neural Networks <https://arxiv.org/abs/2306.13812>`_

8. Recurrent memory: try LSTM/GRU before Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model is limited by partial observability, recurrence is more compelling than a Transformer-first approach.

Why:

- RL observations are currently single-frame on the image side.
- Some temporal information exists in float inputs, especially previous actions and kinematic signals.
- If that still proves insufficient, the next natural step is a recurrent latent module.

Recommended ordering:

1. ``CNN/float encoder → fused latent → GRU/LSTM → IQN heads``
2. Only later consider Transformer memory if recurrence is clearly insufficient.

Why not Transformer first:

- higher engineering and tuning cost,
- replay + sequence handling becomes harder (R2D2-style burn-in needed),
- benefits are strongest when long-horizon memory is clearly the bottleneck,
- the current project already injects useful short-horizon dynamics through float inputs.

Literature note:

- ``R2D2`` is the practical value-based reference for recurrent replay.
- ``GTrXL`` is the most relevant Transformer-for-RL reference, but it does **not** imply that a Transformer is the best first modernization for this project.
- ``Mamba`` (2025 benchmarks) achieves 4.5× higher throughput than LSTM with comparable performance on partially observable tasks. Worth considering if recurrence becomes needed.

9. Visual backbone strategy: prefer in-domain pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project already has a solid foundation for visual and BC pretraining. That is a stronger fit than dropping in a generic frozen foundation model.

Recommended visual strategy:

- First preference:
  - keep an IQN-compatible or near-compatible small CNN (IMPALA-style),
  - pretrain it on project-specific data (AE, VAE, SimCLR — all already implemented),
  - transfer the encoder into RL.
- Second preference:
  - experiment with a lightweight mobile CNN if runtime or throughput becomes critical.
- Lower priority:
  - frozen generic encoders such as DINOv2-style features.

Why generic foundation backbones are not the first recommendation here:

- inputs are grayscale and low-resolution (64×64),
- the domain is highly specialized,
- transfer from generic natural-image representations is less certain,
- the project already benefits from in-domain pretraining paths.

Relevant vision reference:

- `DINOv2 <https://arxiv.org/abs/2304.07193>`_

What About Completely Different Approaches?
-------------------------------------------

A natural question is: should the project switch away from IQN entirely?

After surveying the 2024–2025 landscape, the answer is **no**:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Approach
     - Applicability
     - Assessment
   * - SAC (continuous)
     - Low
     - TMRL uses SAC for TrackMania, but this project uses discrete actions with multi-action blocks.
       Switching to continuous control = full rewrite of action space, replay, and heads.
   * - Dreamer-v3 (world model)
     - Low
     - 200M parameters, week-long training. Fundamentally different architecture.
       Powerful but impractical for this project's compute constraints.
   * - Diffusion-based critics
     - Very low
     - Novel (2025), models inverse CDF via diffusion bridges. No production-ready discrete-action implementations.
   * - Flow-based distributional RL
     - Very low
     - Normalizing flows for return distributions. Research-stage, no practical implementations.
   * - **BTR (IQN + surrounding improvements)**
     - **High**
     - **This is the target.** SOTA on Atari on a desktop. Discrete, value-based, IQN-based.
       The roadmap below is essentially "adapt BTR's recipe to this project's specifics."

The IQN framework is the right one. The upgrades should be **around** it, not replacing it.

What Probably Matters More Than A Fancy Backbone
------------------------------------------------

The literature around visual RL repeatedly suggests that some simple methods can outperform larger architecture changes.

High-value examples for this project, roughly ordered by expected ROI:

- Munchausen targets (massive stability improvement, ~50 lines of code),
- enabling prioritized replay and image augmentation (already implemented, just disabled),
- spectral normalization and layer normalization (training stability),
- noisy linear layers (state-dependent exploration),
- residual CNN encoder (representation capacity),
- stronger fused MLP trunk (post-fusion processing),
- improved quantile conditioning (FiLM-style).

For this project, these collectively matter more than switching to a ViT or a large pretrained vision encoder.

Recommended Roadmap
-------------------

Tier 0: Free experiments (already implemented, just enable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These require no new code — only configuration changes. Run these first to establish a stronger baseline.

1. **Enable random crop augmentation** (``apply_randomcrop_augmentation: true``)
   - Already implemented with 2-pixel padding. Test immediately.

2. **Enable Prioritized Experience Replay** (``prio_alpha: 0.2``)
   - Already implemented including importance-weight correction. Start with ``α = 0.2`` (BTR's choice for IQN).

Tier 1: Best first experiments (low risk, high value)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of these is an independent experiment. They can be combined incrementally.

3. **Munchausen IQN (M-IQN)**
   - Modify target computation in ``train_on_batch``. ~50 lines.
   - Replaces Double DQN (simplifies code).
   - Largest expected stability improvement based on BTR evidence.

4. **IMPALA-style residual CNN + Adaptive Pooling**
   - Replace the 4-layer conv stack with IMPALA-CNN (3 residual blocks).
   - Replace ``Flatten`` with ``AdaptiveMaxPool2d(6, 6)`` or ``AdaptiveAvgPool2d(4, 4)`` + Flatten.
   - Largest expected representation improvement based on BTR evidence.

5. **Spectral Normalization on conv layers**
   - Wrap each ``Conv2d`` with ``torch.nn.utils.spectral_norm()``.
   - Stabilizes training, especially important with larger networks.

6. **Layer Normalization on dense layers**
   - Add ``LayerNorm`` after fusion, between MLP hidden layers, and before V/A heads.
   - Complements spectral norm: SN controls conv weights, LN stabilizes dense activations.

7. **Noisy linear layers in V / A heads**
   - Replace ``nn.Linear`` with ``NoisyLinear`` (factorized Gaussian) in the value and advantage heads.
   - Provides state-dependent exploration, complementing Munchausen's entropy regularization.

Tier 1.5: Still low risk, refine the trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

8. **Stronger post-fusion MLP trunk**
   - Add 1–2 residual MLP blocks after image+float concatenation.
   - ``LayerNorm → Linear → activation → Linear + skip``.

9. **Improved IQN quantile conditioning**
   - Deeper ``iqn_fc`` (2 layers instead of 1).
   - FiLM-style scale+shift conditioning instead of purely multiplicative.

Tier 2: High-upside, moderate engineering cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

10. **Multi-action head redesign**
    - Offset embeddings + per-offset value + lightweight offset mixer.
    - Requires careful ablation since the current factorized head is a core project feature.

11. **Non-Crossing Quantile Networks**
    - Monotonicity constraint on the quantile function via non-negative activations.
    - Addresses a real issue (quantile crossing) that both IQN and FQF have.
    - Architecturally simpler than FQF.

12. **FQF-style learned quantile fractions**
    - Fraction proposal network + 1-Wasserstein loss.
    - Theoretically strongest parameterization of the distribution.
    - Higher implementation and tuning cost than NQ-Networks.

13. **Project-specific visual pretraining with the chosen RL backbone**
    - Especially useful if representation quality appears to cap performance.
    - Re-run pretraining pipelines (AE/VAE/SimCLR) with the new IMPALA encoder.

Tier 3: Only after evidence of specific limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

14. **Plasticity maintenance interventions**
    - CReLU activations, periodic soft resets, dormant neuron monitoring.
    - Only needed if training curves show clear plasticity degradation after many millions of steps.

15. **GRU/LSTM over fused latent**
    - Sequence replay and burn-in (R2D2-style) likely needed.
    - Best next step when partial observability is clearly limiting.
    - Consider Mamba as a throughput-efficient alternative to LSTM.

16. **Transformer / GTrXL-style memory**
    - Interesting research path.
    - But not the recommended first modernization for this project.

Summary Recommendations
-----------------------

If the project wants the most research-supported, project-compatible plan, the order should be:

1. enable already-implemented features (augmentation, PER) — free baseline improvement,
2. add Munchausen targets — highest-ROI single change,
3. upgrade to IMPALA-style residual CNN with spectral norm and layer norm,
4. add noisy linear layers for state-dependent exploration,
5. strengthen the post-fusion trunk and quantile conditioning,
6. redesign the multi-action head,
7. consider NQ-Networks or FQF if distribution quality is still limiting,
8. add recurrent memory only if temporal information proves insufficient,
9. prefer in-domain pretraining over generic frozen visual backbones.

Short version
~~~~~~~~~~~~~

- **Best single change right now:** ``Munchausen IQN`` (proven in BTR, ~50 lines, massive stability gain).
- **Best architecture upgrade:** ``IMPALA-CNN + Spectral Norm + Layer Norm`` (BTR's core recipe).
- **Best free improvements:** enable PER and random crop augmentation (already implemented).
- **Best distribution head upgrade (if needed):** ``NQ-Networks`` (simpler than FQF, solves a real problem).
- **Best temporal upgrade (if needed):** ``GRU/LSTM`` before Transformer.
- **Best visual strategy:** small project-aligned backbone + in-domain pretraining.
- **Best overall target to aim for:** adapt the BTR recipe (IQN + Munchausen + IMPALA + SN + NoisyNets + PER) to this project's specifics.
- **Least convincing first move:** replacing everything with a generic Transformer or large frozen foundation encoder.

Suggested Follow-up Experiments
-------------------------------

If this page is turned into an implementation roadmap, a reasonable experimentation order is:

- ``exp_enable_augmentation_v1`` — enable random crop, compare to baseline.
- ``exp_enable_per_v1`` — enable PER (α=0.2), compare to baseline.
- ``exp_munchausen_iqn_v1`` — M-IQN target modification.
- ``exp_impala_encoder_v1`` — IMPALA-CNN + adaptive pooling, replacing the 4-layer conv stack.
- ``exp_spectral_norm_v1`` — add spectral normalization to conv layers.
- ``exp_layer_norm_v1`` — add layer normalization to dense layers.
- ``exp_noisy_heads_v1`` — NoisyLinear in V/A heads.
- ``exp_residual_trunk_v1`` — residual MLP blocks after fusion.
- ``exp_film_conditioning_v1`` — FiLM-style quantile conditioning.
- ``exp_multi_action_mixer_v1`` — offset embeddings + temporal mixer in multi-action head.
- ``exp_nq_network_v1`` — non-crossing quantile constraints.
- ``exp_fqf_v1`` — FQF fraction proposal network.
- ``exp_gru_latent_v1`` — recurrent latent after fusion.

These reflect a sensible experimentation order from free/low-risk to higher-risk changes.
Each experiment should be compared against the best accumulated baseline from previous successful experiments.
