Fixed
^^^^^

* Moved the core Franka Reach, Lift, Reorient, Drawer, and deformable tasks to one Menagerie asset
  with compact backend-specific collision and physics variants.
* Selected the native MuJoCo physics payload for Newton Franka tasks so the gripper imports one
  finger-mimic constraint instead of combining the generic and PhysX mimic schemas.
* Normalized absolute differential IK position actions to the configured Reach command workspace,
  so a zero policy action targets the workspace center instead of a pose near the world origin.
* Made the Franka cabinet end-effector frame selectors compatible with both the legacy and Menagerie
  asset hierarchies.
* Reset rigid, deformable, and cable lift environments on non-finite physics state and kept their
  terminal rewards finite, preventing one unstable environment from aborting vectorized training.
* Restored symmetric point-cloud observation noise in the Lift and Reorient ADR curriculum.
* Restored the Franka deformable-camera training contract with stationary RGB normalization, a
  spatial-softmax encoder, and the stable fixed learning-rate schedule.
