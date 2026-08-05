Added
^^^^^

* Added :class:`~isaaclab_newton.ik.NewtonIKJointPostureObjective` and
  :class:`~isaaclab_newton.ik.NewtonIKJointPostureObjectiveCfg` for regularizing
  selected scalar joints toward a reference posture.

Fixed
^^^^^

* Fixed Newton inverse kinematics for assets whose articulation root is below
  the asset prim and for prototype builders that share mesh geometry with the
  finalized simulation model.
* Fixed coupled VBD CUDA graph capture when rigid-contact history buffers need
  eager initialization.
