Model architectures
===================

This section describes **what tensors flow where** for IQN and PPO. YAML knobs live in :ref:`nn-yaml-reference` and :ref:`btr-yaml-reference` (:doc:`../configuration_guide`). For a **tabular catalog of every supported stack** (fusion modes, vision branches, fusion trunks, IQN decoder, pitfalls), see :doc:`nn_topology_catalog`.

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

**IQN** is **distributional off-policy** (quantile Q, replay, target net). **PPO** is **on-policy actor-critic** (logits + V, GAE, no replay). Multimodal **IQN** and **PPO** share the same fusion body; IQN swaps policy/value heads for ``iqn_fc`` + dueling heads. PPO never uses IQN’s target net slot ``weights2.torch``.

**BTR** is **not** a separate ``training.algorithm``: it is optional flags on top of **IQN** (same ``IQN_Network``). See :doc:`btr_architecture`.

Reference configs (``config_files/rl/``): ``config_default.yaml`` / ``config_btr.yaml`` (classic IQN); ``config_btr_post_concat_cnn_transformer.yaml`` (BTR + multimodal ``post_concat`` + CNN + fusion transformer). PPO YAMLs in that directory also define ``nn`` layouts usable for **multimodal IQN** if you set ``training.algorithm: iqn``.

Contents
--------

.. toctree::
   :maxdepth: 2

   nn_topology_catalog
   iqn_architecture
   ppo_architecture
   btr_architecture
