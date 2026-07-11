Added
^^^^^

* Added the ``Isaac-Pour-Franka-v0`` contributed task, where a Franka pours
  granular MPM media between scene-owned cups using proxy-coupled MJWarp and
  implicit-MPM solvers.
* Added backward-curriculum and OmniReset-style reset-mixture training presets
  with collision-screened Newton IK resets, broad tabletop object and arm
  randomization, asymmetric observations, general fixed-weight rewards, and
  particle-based sustained success metrics.
* Added CUDA-graph-captured sparse-grid training with isolated MPM worlds,
  fixed-grid playback, visible MPM particles, video-friendly camera framing,
  and SpaceMouse teleoperation presets for Franka Pour.
