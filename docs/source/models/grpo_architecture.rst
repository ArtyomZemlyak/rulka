.. _grpo_architecture:

GRPO: network and training
===========================

This page documents **GRPO** (group-relative policy optimization) when
``training.algorithm`` is ``grpo``: the **same discrete shared-trunk actor-critic**
as PPO (:doc:`ppo_architecture`), with a **trajectory-level** objective based on
**group-relative returns** instead of PPO’s per-step GAE and clipped ratio.

Implementation: ``trackmania_rl.agents.policy_optimization.grpo`` (advantages and
policy objective), ``trackmania_rl.multiprocess.learner_grpo`` (learner loop).
``trackmania_rl.agents.algorithms.registry`` maps ``"grpo"`` to
``trackmania_rl.agents.algorithms.ppo_wiring`` — **identical** network factory and
``PPOInferer`` rollout path as PPO.

YAML under ``grpo:`` (flat ``grpo_*`` on ``get_config()``): :ref:`grpo-config` in
:doc:`../configuration_guide`. Reference config: ``config_files/rl/config_grpo.yaml``.

What GRPO is doing here (why each idea)
----------------------------------------

**Reuse ``ppo_wiring`` and the same actor-critic.** You keep one vision + float
design (Variants A/B/C on :doc:`ppo_architecture`) and the same checkpoints /
collector contract. Only the **learner** changes: no GAE, no PPO clip — so you can
experiment with group-based credit without redefining ``nn``.

**Trajectory scalar :math:`R_i`.** Each rollout is turned into per-step rewards
(same dense + engineered shaping as PPO). Summing them gives one number per
trajectory segment. **Why:** GRPO compares whole chunks of behavior (e.g. how
far you got on the map in that run), not individual timesteps, so the signal is
aligned with “this run was good/bad *relative to other runs collected now*.”

**Wait for ``grpo_group_size`` valid batches.** Short or malformed segments are
dropped (same minimum length as the shared tensor builder). **Why:** advantages
are defined only *within* a fixed group; partial groups would bias which
trajectories enter training.

**Group-relative advantages :math:`A_i`.** Subtract the group mean (and
optionally scale by group std) so :math:`\sum_i A_i = 0`. **Why:** you learn from
*which trajectory beat the others in this batch*, not from absolute return scale
(which drifts with reward schedules and map difficulty). Better-than-average runs
get positive :math:`A_i` and are reinforced; worse-than-average are discouraged.

**Policy term :math:`-A_i \sum_t \log\pi(a_{t}\mid s_t)`.** Classic REINFORCE on
the full trajectory, weighted by :math:`A_i`. **Why no PPO ratio:** data are always
on-policy for the *current* :math:`\theta` inside each inner epoch; the code
recomputes :math:`\log\pi` with the live policy, so there is no stale behavior
policy to correct with a ratio.

**Recompute :math:`\log\pi` for ``grpo_update_epochs`` passes.** **Why:** multiple
Adam steps on the same :math:`K` trajectories extract more from expensive env
interaction, similar in spirit to PPO epochs — but still without a clip, so
``grpo_max_grad_norm`` and moderate learning rates matter.

**Entropy term.** Same role as in PPO: encourage stochasticity so the policy
does not collapse to a single action mode too early.

**Optional ``ref_policy`` + ``grpo_ref_kl_coef``.** A frozen copy (periodically
synced) evaluates :math:`\log\pi_{\mathrm{ref}}`. **Why:** penalize deviation from
a reference snapshot or slowly moving anchor — useful if you want conservative
updates or started from a strong prior. If the coefficient is ``0``, no extra
forward passes on the reference.

**Value head unused in the loss.** Still computed at collection so queues and
``build_policy_rollout_tensors`` stay **one** code path with PPO/DPO. **Why not
strip it:** less duplication and easier switching between ``ppo``, ``dpo``, and
``grpo``; only the learner ignores :math:`V` for GRPO’s objective.

**Shared network + lock after each group update.** Collectors must act with
weights the learner just produced. **Why:** same real-time sync story as PPO —
parallel envs, single source of truth for inference weights.

Algorithm placement (same code paths as PPO)
---------------------------------------------

GRPO does not introduce a new ``nn`` topology: **config → registry →
``ppo_wiring`` → ``make_network``** builds the same classes as PPO (Variant A/B/C
on :doc:`ppo_architecture`). Only **``learner_grpo``** replaces **``learner_ppo``**.

.. graphviz::

   digraph grpo_code_stack {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      yaml [label="YAML:\ntraining.algorithm = grpo\n+ grpo: block", style="filled", fillcolor=lightcyan];
      reg [label="registry.get_wiring(\"grpo\")\n→ ppo_wiring module"];
      mk [label="ppo_wiring.make_network()\nPpoActorCritic | HfActorCritic |\nTorchMultimodalActorCritic", style="filled", fillcolor=lightgreen];
      col [label="collector_process\nPPOInferer"];
      lr [label="learner_grpo.py\n(group loss)", style="filled", fillcolor=lightyellow];
      yaml -> reg -> mk;
      mk -> col;
      mk -> lr;
   }

Training stack (processes)
--------------------------

Same multiprocess layout as PPO: collectors fill queues; **one** learner process
consumes rollouts. The learner holds the trainable policy, a **frozen**
``ref_policy`` copy (optional KL term), and syncs weights into
``uncompiled_shared_network`` after each **group** update.

**Why a separate ``ref_policy`` node in the diagram:** KL regularization needs
two forwards — trainable :math:`\pi_\theta` and fixed :math:`\pi_{\mathrm{ref}}`
— without mixing gradients into the reference. Periodic ``load_state_dict`` from
the live policy decides how “stale” the anchor is.

.. graphviz::

   digraph grpo_process_stack {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      train [label="scripts/train.py", style="rounded,filled", fillcolor=lightcyan];
      lp [label="learner_process.py\nalgorithm == grpo → learner_grpo", style="filled", fillcolor=lightyellow];
      cp [label="collector_process.py × N"];
      inf [label="PPOInferer\n(forward + sample + log p, V)"];
      lgr [label="learner_grpo.py\nK rollouts → advantages → loss"];
      ref [label="ref_policy\n(deepcopy, eval, no grad)\noptional KL", style="filled", fillcolor=lightsteelblue];
      pol [label="policy (trainable)", style="filled", fillcolor=lightgreen];
      sh [label="uncompiled_shared_network\n+ lock", style="filled", fillcolor=lightpink];
      q [label="rollout_queues"];
      train -> lp;
      train -> cp;
      cp -> inf -> pol;
      inf -> q [label="put"];
      q -> lgr;
      lp -> lgr;
      lgr -> pol;
      lgr -> ref [style=dashed, label="forward no grad"];
      lgr -> sh [label="load_state_dict"];
      inf -> sh [style=dashed, label="inference weights"];
   }

Policy network (identical to PPO)
---------------------------------

All tensor routing (image + float → trunk → logits + :math:`V`) is on
:doc:`ppo_architecture`. The **conceptual** forward at collection and training is:

.. graphviz::

   digraph grpo_forward {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      img [label="obs_img\n(T,1,H,W)", style="filled", fillcolor=lightblue];
      fl [label="obs_float\n(T,F)", style="filled", fillcolor=lightblue];
      act [label="actions\n(from rollout)", style="filled", fillcolor=lightblue];
      trunk [label="shared trunk\n(same as PPO)", style="filled", fillcolor=lightyellow];
      ev [label="evaluate_actions\n(img, float, actions)", style="filled", fillcolor=wheat];
      out [label="log p, entropy, V\nGRPO: sum log p over T;\nV unused in loss", style="filled", fillcolor=lightgreen];
      vnote [label="value head still runs\n(grad flows only via\npolicy + entropy paths)", shape=note, fontsize=9];
      img -> trunk;
      fl -> trunk;
      trunk -> ev;
      act -> ev;
      ev -> out;
      ev -> vnote [style=dotted];
   }

**Collection:** collectors still store ``ppo_values`` for parity with the tensor
builder; **GRPO** ignores them in the objective. Entropy can be averaged per
trajectory inside the learner when stacking the group.

Rollout → GPU batch (one trajectory)
------------------------------------

``build_policy_rollout_tensors`` (``policy_rollout_batch.py``) aligns frames,
``state_float``, ``actions``, ``ppo_log_probs``, and ``ppo_values``, then calls
``ppo_rewards_and_dones_from_rollout`` for per-step rewards (dense + engineered,
same as PPO). Invalid segments (too few steps) return ``None`` and are dropped.

.. graphviz::

   digraph grpo_rollout_batch {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      env [label="Env rollout dict:\nframes[], state_float[],\nactions[], ppo_log_probs[],\nppo_values[]", style="filled", fillcolor=lightblue];
      build [label="build_policy_rollout_tensors"];
      t [label="GPU tensors:\nobs_img, obs_float, actions\nrewards (T,), dones\nold_logp, old_values", style="filled", fillcolor=lightyellow];
      R [label="R_i = sum_t rewards[t]\n(scalar per trajectory)", style="filled", fillcolor=lightgreen];
      env -> build -> t -> R;
   }

Forming a group and group-relative advantages
---------------------------------------------

The learner buffers valid batches until it has **exactly** ``grpo_group_size``
trajectories :math:`\tau_1,\ldots,\tau_K`. Each has a scalar return
:math:`R_i = \sum_t r_{i,t}`. Advantages are **detached** and **zero-mean** across
the group:

- ``mean``: :math:`A_i = R_i - \frac{1}{K}\sum_j R_j`.
- ``mean_std``: center, then divide by group std (with stabilizer). **Why
  ``mean_std``:** when absolute return spread changes a lot across training, scaling
  keeps gradient scale more stable than centering alone.

.. graphviz::

   digraph grpo_group_adv {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      b1 [label="batch τ_1 → R_1", style="filled", fillcolor=lightblue];
      b2 [label="batch τ_2 → R_2", style="filled", fillcolor=lightblue];
      bk [label="batch τ_K → R_K", style="filled", fillcolor=lightblue];
      dot [label="...", shape=plaintext];
      grp [label="stack [R_1..R_K]"];
      adv [label="group_relative_advantages\nmean | mean_std", style="filled", fillcolor=lightyellow];
      out [label="A_1..A_K\n(detached)", style="filled", fillcolor=lightgreen];
      b1 -> grp;
      b2 -> grp;
      dot -> grp;
      bk -> grp;
      grp -> adv -> out;
   }

Policy loss (inner epochs)
--------------------------

For each ``grpo_update_epochs`` pass, the learner recomputes
:math:`\log\pi_\theta(\tau_i)=\sum_t \log\pi_\theta(a_{i,t}\mid s_{i,t})` with the
**current** :math:`\theta`. The policy term is
:math:`\mathcal{L}_\pi = -\frac{1}{K}\sum_i A_i \log\pi(\tau_i)`.
Optional **reference** term uses ``ref_policy`` with ``torch.no_grad()`` on the
reference branch; ``ref_policy`` is refreshed from the live policy every
``grpo_ref_sync_every_updates`` group updates.

.. graphviz::

   digraph grpo_loss_detail {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      batches [label="K batches\n(obs, actions)", style="filled", fillcolor=lightblue];
      subgraph cluster_pi {
         label="Trainable policy (each traj i)";
         style=dashed;
         e1 [label="policy.evaluate_actions\n→ logp (T,), ent"];
         s1 [label="traj_logp_i = sum_t logp"];
         e1 -> s1;
      }
      subgraph cluster_ref {
         label="Optional ref (grpo_ref_kl_coef > 0)";
         style=dashed;
         e2 [label="ref_policy.evaluate_actions\n(no grad)"];
         kl [label="mean over steps of\n(log pi_theta - log pi_ref)"];
         e2 -> kl;
      }
      batches -> e1;
      batches -> e2;
      advn [label="A_i (detached)", style="filled", fillcolor=lightcyan];
      lpi [label="L_pi = -mean_i A_i * traj_logp_i"];
      le [label="- grpo_ent_coef * mean(entropy)"];
      lkl [label="+ grpo_ref_kl_coef * KL term"];
      tot [label="total loss -> backward\nGradScaler + clip_grad_norm\nAdam step", style="filled", fillcolor=lightgreen];
      advn -> lpi;
      s1 -> lpi;
      lpi -> tot;
      le -> tot;
      kl -> lkl -> tot;
   }

End-to-end training loop (summary)
----------------------------------

.. graphviz::

   digraph grpo_train {
      rankdir=TB;
      node [shape=box, fontname="Helvetica", fontsize=10];
      col [label="Collectors:\nPPOInferer\n(same as PPO)"];
      q [label="Rollout queues:\nframes, float, actions,\nlog p, V"];
      b [label="build_policy_rollout_tensors\n(step rewards, dones)"];
      buf [label="Buffer until K valid\n(grpo_group_size)"];
      adv [label="group_relative_advantages\non R_i = sum r"];
      ep [label="grpo_update_epochs ×\nforward + loss + step"];
      sync [label="sync ref_policy\nevery N updates"];
      sh [label="shared network\nfor collectors"];
      col -> q -> b -> buf -> adv -> ep -> sh;
      ep -> sync [style=dashed];
   }

PPO vs GRPO (architecture vs credit assignment)
----------------------------------------------

**Architecture** (CNN / HF / fusion, two inputs, two heads) is the same; **only**
the learner’s use of outputs differs.

.. graphviz::

   digraph ppo_vs_grpo {
      rankdir=LR;
      node [shape=box, fontname="Helvetica", fontsize=10];
      subgraph cluster_ppo {
         label="PPO learner";
         style=filled;
         fillcolor="#f0f8ff";
         p1 [label="Many steps in rollout buffer"];
         p2 [label="GAE per timestep\nÂ_t, R_t"];
         p3 [label="Clipped ratio + value loss + H"];
         p1 -> p2 -> p3;
      }
      subgraph cluster_grpo {
         label="GRPO learner";
         style=filled;
         fillcolor="#fff8f0";
         g1 [label="K full trajectories"];
         g2 [label="Scalar R_i per traj\nA_i within group"];
         g3 [label="−A_i Σ log π + H\n(+ optional ref)"];
         g1 -> g2 -> g3;
      }
      net [label="Same actor-critic\n(ppo_wiring)", style="filled", fillcolor=lightgreen];
      net -> p1;
      net -> g1;
   }

.. list-table::
   :header-rows: 1
   :widths: 18 38 40

   * - Aspect
     - PPO (:doc:`ppo_architecture`)
     - GRPO (this page)
   * - Credit assignment
     - **GAE** on step rewards with :math:`\gamma`, :math:`\lambda`
     - **Scalar return** :math:`R_i` per trajectory; **group-relative** :math:`A_i`
   * - Policy objective
     - Clipped **importance ratio** vs behavior policy
     - **REINFORCE-style** :math:`-A_i \sum_t \log\pi(a_t|s_t)` (no ratio clip)
   * - Value head
     - **Used** (value loss + bootstrap for GAE)
     - **Not used** in the loss (still computed at collection)
   * - Batch shape
     - ``ppo.rollout_steps_per_update`` then minibatches
     - **Exactly** ``grpo_group_size`` trajectories per update

Implementation references
-------------------------

- ``trackmania_rl/agents/policy_optimization/grpo.py`` — group centering and policy objective.
- ``trackmania_rl/multiprocess/learner_grpo.py`` — GRPO learner loop, reference policy, TensorBoard.
- ``trackmania_rl/multiprocess/policy_rollout_batch.py`` — ``build_policy_rollout_tensors``, ``grpo_scheduled_float``.
- ``trackmania_rl/agents/algorithms/ppo_wiring.py`` — network factory (shared with PPO/DPO).
- ``trackmania_rl/multiprocess/collector_process.py`` — policy-optimization collectors (PPO/DPO/GRPO).
- ``config_files/config_schema.py`` — ``GRPOConfig``.

See also
--------

- :doc:`ppo_architecture` — full actor-critic topology (Variants A/B/C), PPO process stack, GAE and clipped-loss diagrams.
- DPO (same ``ppo_wiring``, preference loss): :ref:`dpo-config` in :doc:`../configuration_guide`.
- :doc:`nn_topology_catalog` — ``nn`` routing; GRPO uses the **PPO** rows.
- :doc:`../configuration_guide` — ``grpo:`` and rollout :math:`\gamma`.
