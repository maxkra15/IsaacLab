Added
^^^^^

* Added :class:`~isaaclab_newton.ik.NewtonIKJointPostureObjective` and
  :class:`~isaaclab_newton.ik.NewtonIKJointPostureObjectiveCfg` for regularizing
  selected scalar joints toward a reference posture.
* Added
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg` for
  authoring cable stretch, bend, damping, and density properties.
* Added named Newton pre-render callbacks for procedural visuals that cannot
  use rigid-body or particle transform synchronization.

Fixed
^^^^^

* Fixed Newton inverse kinematics for assets whose articulation root is below
  the asset prim and for prototype builders that share mesh geometry with the
  finalized simulation model.
* Fixed Kit and Newton-viewer rendering of instanced articulated assets and
  render-only static USD meshes, including compatibility with the Cubric 0.2
  interface.
* Fixed solver-history resets after state teleports and VBD contact-history
  allocation when an explicit rigid-contact capacity is configured.
