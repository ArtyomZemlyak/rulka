Model architectures
===================

This section describes **what tensors flow where** for **IQN** and for **policy
optimization** (**PPO**, **DPO**, **GRPO** — the last two reuse the PPO actor-critic
and ``ppo_wiring``). YAML knobs live in :ref:`nn-yaml-reference` and
:ref:`btr-yaml-reference` (:doc:`../configuration_guide`). For a **tabular catalog
of every supported stack** (fusion modes, vision branches, fusion trunks, IQN
decoder, pitfalls), see :doc:`nn_topology_catalog`.

Which stack when?
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - You set
     - Algorithm
     - Architecture page / module
   * - ``training.algorithm: iqn``, ``nn.fusion_mode: none``, ``nn.vis.cnn`` or ``no_image``
     - IQN
     - :doc:`iqn_architecture` — ``IQN_Network`` (``iqn.py``); optional :doc:`btr_architecture`.
   * - ``training.algorithm: iqn``, ``nn.fusion_mode: none``, ``nn.vis.transformer.use_hf_backbone: true``
     - IQN
     - :doc:`iqn_architecture` — ``IQNSharedBackboneNetwork`` + headless ``HfActorCritic`` (``hf_actor_critic.py``).
   * - ``training.algorithm: iqn``, ``nn.fusion_mode`` in ``vision_transformer`` / ``post_concat`` / ``unified``
     - IQN
     - :doc:`iqn_architecture` — ``IQNSharedBackboneNetwork`` + headless ``TorchMultimodalActorCritic`` (``multimodal_torch_fusion.py``).
   * - ``training.algorithm: ppo``, ``nn.fusion_mode: none``, ``nn.vis.cnn`` (or ``no_image``)
     - PPO
     - :doc:`ppo_architecture` — ``PpoActorCritic`` (``ppo_actor_critic.py``).
   * - ``training.algorithm: ppo``, ``nn.fusion_mode: none``, ``nn.vis.transformer.use_hf_backbone: true``
     - PPO
     - :doc:`ppo_architecture` — ``HfActorCritic`` (``hf_actor_critic.py``); extra ``pip install -e ".[policy]"``.
   * - ``training.algorithm: ppo``, ``nn.fusion_mode`` in ``vision_transformer`` / ``post_concat`` / ``unified``
     - PPO
     - :doc:`ppo_architecture` — ``TorchMultimodalActorCritic`` (``multimodal_torch_fusion.py``).
   * - ``training.algorithm: dpo``, same ``nn`` choices as PPO (CNN / HF / fusion)
     - DPO
     - :doc:`ppo_architecture` — **same** built modules; training / pairs under :ref:`dpo-config` in :doc:`../configuration_guide`.
   * - ``training.algorithm: grpo``, same ``nn`` choices as PPO (CNN / HF / fusion)
     - GRPO
     - :doc:`grpo_architecture` — **same** modules as PPO; :doc:`ppo_architecture` for tensor routing diagrams.

**IQN** is **distributional off-policy** (quantile Q, replay, target net). **PPO** is **on-policy actor-critic** (logits + V, GAE, clipped objective, no replay). **DPO** keeps that actor-critic but trains from **preference pairs** (chosen vs rejected trajectories; :ref:`dpo-config`). **GRPO** is **on-policy** with the **same** actor-critic but **group-relative trajectory returns** and no PPO clip (see :doc:`grpo_architecture`). Multimodal **IQN** and **PPO** (and **DPO** / **GRPO**) share the same fusion body when ``nn.fusion_mode`` matches; IQN swaps policy/value heads for ``iqn_fc`` + dueling heads. PPO/DPO/GRPO never use IQN’s target net slot ``weights2.torch``.

**BTR** is **not** a separate ``training.algorithm``: it is optional flags on top of **IQN** (same ``IQN_Network``). See :doc:`btr_architecture`.

Reference configs (``config_files/rl/``): ``config_default.yaml`` / ``config_btr.yaml`` (classic IQN); ``config_btr_post_concat_cnn_transformer.yaml`` (BTR + multimodal ``post_concat`` + CNN + fusion transformer). ``config_ppo.yaml`` and siblings define ``nn`` layouts usable for **multimodal IQN** if you set ``training.algorithm: iqn``; ``config_dpo.yaml`` / ``config_grpo.yaml`` mirror the PPO stack with ``dpo:`` / ``grpo:`` blocks.

Contents
--------

.. toctree::
   :maxdepth: 2

   nn_topology_catalog
   iqn_architecture
   ppo_architecture
   grpo_architecture
   btr_architecture
