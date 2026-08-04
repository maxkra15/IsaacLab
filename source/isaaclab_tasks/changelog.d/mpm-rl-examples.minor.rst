Added
^^^^^

* Added Newton MPM Franka pouring and UR10 particle-pushing reinforcement
  learning tasks with reset-safe particle randomization and rigid-particle
  coupling.
* Added rollout-boundary synchronization of adaptive Franka Pour reset-dataset
  evidence across distributed RSL-RL workers.
* Added bounded sparse MPM configurations for both tasks, with CUDA graph
  capture for fixed-payload Pour and eager public-API activation for
  variable-payload Push.
* Added randomized pile footprint, payload, and lateral placement to the UR10
  particle-pushing task, with partial-progress and split-pile resets for
  multi-pass manipulation.
