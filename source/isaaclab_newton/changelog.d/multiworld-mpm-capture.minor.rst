Added
^^^^^

* Added :attr:`~isaaclab_newton.assets.MPMObject.particle_offsets` for mapping per-environment
  particle buffers without private asset access.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.reset_solver_state`,
  :meth:`~isaaclab_newton.physics.NewtonManager.prepare_cuda_graph_capture`, and
  :meth:`~isaaclab_newton.physics.NewtonManager.is_cuda_graph_active` as public lifecycle seams.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.check_solver_status` for application-owned
  outer graphs to inspect sticky device failures after replay.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_builder_world_hook` and
  :meth:`~isaaclab_newton.physics.NewtonManager.unregister_builder_world_hook` so tasks can extend
  replicated Newton worlds without mutating manager internals.

Changed
^^^^^^^

* Changed Newton CUDA graph setup to use each solver's public capture-capability and preparation
  contracts, including recursive coupled-solver preparation.
* Changed Newton environment reset masks to boolean arrays and updated the Kamino manager to use
  Newton's public solver reset contract.

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.physics.NewtonCoupledManager` so kinematic rigid bodies selected
  for implicit MPM work as massless colliders and refresh forward kinematics before stepping.
* Fixed coupled implicit-MPM particle projection to operate on authoritative entry state and
  reconcile through Newton's public coupled-state API.
* Fixed selective reset to update both manager state buffers and finish from the authoritative
  input state without exposing solver internals.
* Fixed CUDA graph allocation warmup to restore solver-private state before recording the first
  real step and reject sparse-capacity failures before advancing simulation time.
* Fixed Newton teardown to discard MPM object registrations before constructing another scene.
* Fixed local Warp source overrides to expose checkout-built native libraries to the dynamic loader.
