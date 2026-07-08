Added
^^^^^

* Added implicit MPM entries to
  :class:`~isaaclab_contrib.coupling.CoupledSolverEntryCfg`, including entry-local
  substeps, in-place stepping, and named solver/view accessors.

Fixed
^^^^^

* Fixed coupled implicit-MPM entries ignoring
  :attr:`~isaaclab_newton.physics.MPMSolverCfg.project_outside_colliders`.
