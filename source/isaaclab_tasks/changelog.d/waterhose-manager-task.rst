Added
^^^^^

* Added the manager-based RBY1 waterhose task with Newton proxy-coupled
  physics, task-space IK actions, and scripted demonstration recording.
* Added ``proxy_mass_scale`` to tune RBY1 waterhose Newton proxy coupling.
* Added ``cable_num_segments`` to resample authored RBY1 waterhose cables
  without changing their placement.
* Added ``command_frame`` to the RBY1 waterhose task-space IK action
  configuration.
* Added accumulated task-space targets for interactive RBY1 waterhose
  teleoperation.

Changed
^^^^^^^

* Changed the RBY1 waterhose task configuration to derive Newton builder timing
  from ``SimulationCfg.dt`` instead of a separate task ``fps`` field.
* Changed RBY1 waterhose launch scripts to use a shared Newton/Kit
  import-order helper.
* Changed the RBY1 waterhose Newton solver, robot drive, and cable material
  defaults to follow Newton's proxy-coupled cable robot example.
* Changed standalone SpaceMouse teleoperation to use simpler translation and
  yaw controls by default for the Newton waterhose task.
* Changed standalone SpaceMouse simple-mode axes to match direct gripper
  motion for the Newton waterhose task.

Fixed
^^^^^

* Fixed Newton viewer startup for standalone RBY1 waterhose scripts when
  ``DISPLAY`` is unset.
* Fixed RBY1 waterhose cable placement after manager environment resets.
* Fixed RBY1 waterhose random-agent and scripted-demo startup when using the
  Kit visualizer.
* Fixed the RBY1 waterhose play variant to use balanced Newton stepping
  defaults for interactive teleoperation.
* Fixed RBY1 waterhose gripper target updates to close gradually instead of
  snapping to the closed joint target in one simulation step.
