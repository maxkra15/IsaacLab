Added
^^^^^

* Added the manager-based RBY1 waterhose task with Newton proxy-coupled
  physics, task-space IK actions, and scripted demonstration recording.

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
