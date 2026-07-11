Added
^^^^^

* Added the ``Isaac-Pour-Franka-v0`` contributed task, where a Franka pours
  granular MPM media between scene-owned cups using proxy-coupled MJWarp and
  implicit-MPM solvers.
* Added backward-curriculum and OmniReset-style reset-mixture training presets
  with collision-screened Newton IK resets, broad tabletop object and arm
  randomization, asymmetric observations, and particle-based success metrics.
* Added sparse-grid training with independent MPM and rigid stepping rates,
  fixed-grid CUDA-graph playback, visible MPM particles, video-friendly camera
  framing, and SpaceMouse teleoperation presets for Franka Pour.
