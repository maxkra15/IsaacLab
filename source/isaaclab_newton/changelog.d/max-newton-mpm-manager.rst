Added
^^^^^

* Added :class:`~isaaclab_newton.physics.MPMSolverCfg` and
  :class:`~isaaclab_newton.physics.NewtonMPMManager` for Newton implicit MPM
  simulations.
* Added teapot-pour and granular MPM demos using
  :class:`~isaaclab_newton.physics.NewtonMPMManager`.
* Added the :class:`~isaaclab_newton.physics.NewtonManager` subclass hooks
  :meth:`~isaaclab_newton.physics.NewtonManager._register_builder_attributes`
  (register a solver's Newton custom builder attributes) and
  :meth:`~isaaclab_newton.physics.NewtonManager._prepare_builder_for_finalize`
  (allow a solver to normalize imported builder data before finalization) and
  :meth:`~isaaclab_newton.physics.NewtonManager._supports_cuda_graph_capture`
  (opt a solver out of CUDA graph capture).
* Added :attr:`~isaaclab_newton.physics.MPMSolverCfg.project_outside_colliders`
  (default ``False``): when set,
  :class:`~isaaclab_newton.physics.NewtonMPMManager` runs
  ``SolverImplicitMPM.project_outside`` after each substep to push particles out
  of collider interiors. It is opt-in so scenes can skip the per-substep
  projection pass unless they need that correction; the MPM demos enable it
  because they use thin/exact mesh colliders.
* Added ``visual_update_frequency`` to MPM particle spawner configs so Kit USD
  point-cloud visualization can be throttled independently from physics.

Changed
^^^^^^^

* :meth:`~isaaclab_newton.physics.NewtonManager.create_builder`,
  :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation`, and
  :meth:`~isaaclab_newton.physics.NewtonManager.instantiate_builder_from_stage`
  now invoke the active manager's solver-specific builder-attribute hook so MPM custom
  attributes (``mpm:young_modulus``, ...) are registered on the builder
  before particles are added or the model is finalized.
* :class:`~isaaclab_newton.physics.NewtonMPMManager` now clears mass and inertia
  on kinematic bodies before finalization so Newton implicit MPM treats imported
  kinematic rigid assets as massless colliders.
* The MPM demos now use the particle visualization update cadence for both
  Newton viewer particle logging and Kit USD point-cloud sync.
* :meth:`~isaaclab_newton.physics.NewtonManager._capture_or_defer_graph` now
  skips CUDA graph capture when the active solver reports it is unsupported,
  so sparse/dense-grid MPM falls back to eager execution.
