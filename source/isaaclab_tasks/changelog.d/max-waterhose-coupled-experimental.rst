Added
^^^^^

* Added the RBY1DF waterhose cable-plug insertion demo under
  ``isaaclab_tasks.contrib.waterhose`` (tasks ``Isaac-Waterhose-Coupled-v0``,
  ``Isaac-Waterhose-Coupled-Teleop-v0`` and the experimental ``Isaac-Waterhose-Admm-v0``).
  It drives a scripted grasp → carry → align → insert → release → back-off arc on a Newton
  MuJoCo-Warp + VBD proxy-coupled scene, with differential-IK actions and native
  keyboard/SpaceMouse plus XR teleoperation.
