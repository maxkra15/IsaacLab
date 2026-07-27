Added
^^^^^

* Added the RBY1 waterhose insertion task, a scripted robot demo, and bimanual
  Apple Vision Pro teleoperation using pure-contact Newton proxy coupling. The
  required waterhose assets are distributed separately from the Git
  repository.

* Added a dedicated Isaac Lab Mimic task that replays the recorded 20D
  ``processed_actions`` as direct bimanual wrist poses and explicit hand-joint
  targets, without applying the Apple Vision Pro clutch a second time.

Fixed
^^^^^

* Restored the standard Apple Vision Pro Play, Stop, and Reset controls for the
  waterhose task. Resetting an episode also clears the relative wrist
  calibration so both arms re-clutch from the reset pose.
