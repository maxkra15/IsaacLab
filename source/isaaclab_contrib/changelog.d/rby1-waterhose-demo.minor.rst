Added
^^^^^

* Added :class:`~isaaclab_contrib.cable.CableObject` and
  :class:`~isaaclab_contrib.cable.CableObjectCfg` for simulating Newton rod
  cables authored as ``UsdGeomBasisCurves``, including replication, reset, and
  curve-visual synchronization support.
* Added VBD configuration for contact friction smoothing, rigid-contact
  history and capacity, AVBD penalty updates, and rigid-joint constraint
  tuning.
* Added ``use_solver_effective_mass`` to
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerEntryCfg` so proxy
  coupling can preserve the model-view inertia of small virtual bodies.

Fixed
^^^^^

* Fixed procedural cable rods replacing a builder's configured shape and
  contact defaults with Newton's standalone defaults.
* Fixed coupled CUDA graph capture starting before environment action and
  controller managers complete lazy initialization.
