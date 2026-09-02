Added
^^^^^

* Added ``IsaacContrib-Drone-Waypoint-FLARE-DirectCTBR``, a rigid-drone
  variation of the manager-based Direct-CTBR waypoint task. It reuses the same
  FLARE drone, randomized route geometry, route objective, 100 Hz collective-
  thrust/body-rate interface, rate PID, mixer, and motor model while removing
  the payload, cable, load-only MDP terms, and coupled VBD solver. Evaluation
  metrics explicitly mark suspended-load channels as unavailable. Direct-CTBR
  policies used isotropically normalized body-frame velocity and route-tube-
  scaled cross-track error, requiring fresh checkpoints while leaving the
  paper-aligned and enhanced policy observations unchanged. The rigid-drone
  runner inherits the shared 500-step, ``gamma=lambda=0.999`` Direct-CTBR PPO
  horizon and 400-update control-step budget under its own fresh checkpoint
  namespace.
