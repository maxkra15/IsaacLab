Added
^^^^^

* Added the RBY1DF waterhose cable-plug insertion demo under
  ``isaaclab_tasks.contrib.waterhose`` (tasks ``Isaac-Waterhose-Coupled-v0`` and
  ``Isaac-Waterhose-Coupled-Teleop-v0``). It drives a scripted grasp → carry →
  align → insert → snap-lock → release → regrasp → pull-out arc on a Newton
  MuJoCo-Warp + VBD proxy-coupled scene, with multi-body Newton IK actions and
  native keyboard/SpaceMouse plus XR teleoperation.
