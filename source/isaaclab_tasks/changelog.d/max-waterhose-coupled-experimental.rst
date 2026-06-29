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

* Made coupled substeps and VBD iterations configurable through validated positive-integer
  ``WATERHOSE_SUBSTEPS`` and ``WATERHOSE_VBD_ITERS`` overrides while retaining the high-fidelity
  defaults of 10 and 20, respectively.

Fixed
^^^^^

* Restored right-gripper-only robot collision filtering and full-scene episode resets so unintended
  robot contacts and state carried between episodes do not affect the task. Cable resets now also
  restore every VBD-owned hose segment pose and velocity before coupled-solver history is refreshed.
