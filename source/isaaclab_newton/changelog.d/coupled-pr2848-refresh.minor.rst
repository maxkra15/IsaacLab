Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.CoupledProxyCfg.proxy_relaxation` to expose Newton's
  lagged proxy feedback relaxation factor, which damps the impulse exchange when light free
  bodies interact with fast-flowing media.
* Added :attr:`~isaaclab_newton.physics.CoupledSolverEntryCfg.preserve_shape_ids` to control
  whether a coupled sub-solver sees parent-model or entry-local shape ids.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonCoupledManager` to preallocate entry-local
  filtered contact buffers via Newton's ``SolverCoupled.prepare_contacts`` after contact
  allocation, replacing the removed ``prepare_graph_capture`` hook.
