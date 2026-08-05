Added
^^^^^

* Added the RBY1 waterhose insertion task, a scripted robot demo, and bimanual
  Apple Vision Pro teleoperation using pure-contact Newton proxy coupling. The
  required waterhose assets are distributed separately from the Git
  repository.

* Added a dedicated Isaac Lab Mimic task that replays the recorded 20D
  ``processed_actions`` as direct bimanual wrist poses and explicit hand-joint
  targets, without applying the Apple Vision Pro clutch a second time.

Changed
^^^^^^^

* Migrated the waterhose from the legacy articulation-based hose path to
  Isaac Lab's native :class:`~isaaclab.assets.CableObject`. A scoped task
  extension adds the connector, damping, and tail attachment to each Newton
  world while retaining the standard cable state and rendering paths.

* Configured Waterhose Mimic annotation to preserve the built Newton solver
  buffers between episodes while still restoring each recorded initial state.
  The generic annotation hard reset remains enabled for other tasks.

Fixed
^^^^^

* Restored the published one-step RBY1 gripper response for Apple Vision Pro
  teleoperation and recording compatibility.

* Included both authored fridge props in the robot-only MJWarp collision proxy
  without adding them to the cable VBD contact space.

* Restored the standard Apple Vision Pro Play, Stop, and Reset controls for the
  waterhose task. Resetting an episode also clears the relative wrist
  calibration so both arms re-clutch from the reset pose.

* Fixed batched waterhose simulation by creating and resetting each cable,
  connector, and tail attachment in its matching environment. Native cable
  state is recorded under ``cable_object/cable1`` as per-segment poses and
  velocities.

* Added backward-compatible Mimic replay for recordings whose cable state was
  stored under the legacy ``articulation/cable1`` path. The robot state and
  actions are preserved while the native cable is initialized from its task
  default.

* Added an explicit Waterhose capability check for ``--use_skillgen`` so the
  unsupported bimanual deformable-contact path fails before environment or
  cuRobo initialization. Standard MimicGen remains supported.
