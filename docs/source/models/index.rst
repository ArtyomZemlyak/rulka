Model architectures
===================

This section describes the model architectures used in the project.

The only value-based RL backbone in the training stack is **IQN** (:doc:`iqn_architecture`).
**BTR (Beyond The Rainbow)** is **not** a separate architecture or ``training.algorithm``:
it is a **bundle of optional improvements** (Munchausen targets, IMPALA-CNN, pooling, spectral norm,
LayerNorm, NoisyNets) toggled under the ``btr:`` section of the same YAML and wired into
``IQN_Network`` / the IQN trainer. See :doc:`btr_architecture` for the feature map.

Contents
--------

.. toctree::
   :maxdepth: 2

   iqn_architecture
   btr_architecture
