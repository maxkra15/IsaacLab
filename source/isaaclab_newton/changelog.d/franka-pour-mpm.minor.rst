Added
^^^^^

* Added isolated-world and bounded sparse-grid options to
  :class:`~isaaclab_newton.physics.MPMSolverCfg`, selective solver-state reset,
  per-world builder extensions, clone-prototype copies, and sparse-solver CUDA
  graph lifecycle forwarding to :class:`~isaaclab_newton.physics.NewtonManager`.

Fixed
^^^^^

* Fixed Newton articulation poses written during environment reset not reaching
  Fabric until a later physics step.
