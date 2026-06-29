Added
^^^^^

* Added the RBY1DF waterhose cable-plug insertion demo under
  ``isaaclab_tasks.contrib.waterhose`` (tasks ``Isaac-Waterhose-Coupled-v0``,
  ``Isaac-Waterhose-Coupled-Teleop-v0`` and the experimental ``Isaac-Waterhose-Admm-v0``).
  It drives a scripted grasp → carry → align → insert → release → back-off arc on a Newton
  MuJoCo-Warp + VBD proxy-coupled scene, with differential-IK actions and native
  keyboard/SpaceMouse plus XR teleoperation.

Changed
^^^^^^^

* Changed the coupled waterhose defaults to the validated 8 substeps and 16 VBD iterations, with
  positive-integer overrides through ``WATERHOSE_SUBSTEPS`` and ``WATERHOSE_VBD_ITERS``.

Fixed
^^^^^

* Restored right-gripper-only robot collision filtering and full-scene episode resets so unintended
  robot contacts and state carried between episodes do not affect the task.
