Fixed
^^^^^

* Fixed Franka Reorient training to keep one nontrivial pose target per episode, so success-driven ADR no longer
  advanced on an easy resampled target. Added a separate completed-episode success metric; existing Reorient
  checkpoints should be requalified against the updated task.
