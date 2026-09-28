Fixed
^^^^^

* Fixed Franka Reorient training to keep one nontrivial pose target per episode, so success-driven ADR no longer
  advanced on an easy resampled target. Added a separate completed-episode success metric; existing Reorient
  checkpoints should be requalified against the updated task.
* Removed additive object inertia from Franka rigid Lift and Reorient so small shapes retain their geometric inertia
  after mass randomization. Retrain and requalify both tasks' checkpoints because their training dynamics have changed.
* Removed initial finger-object penetration from Franka Reorient's per-shape pre-grasps, including lateral reset
  jitter along the gripper's closing axis. Retrain Reorient checkpoints because the reset distribution changed.
