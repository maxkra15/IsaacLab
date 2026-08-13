Added
^^^^^

* Added a state-based Franka task for order-invariant, end-to-end three-cube
  stacking with measured-state relative joint-position control and a binary
  gripper.
* Added a deployable Franka RGB-camera task and a teacher-student distillation
  task. The camera actor receives RGB and robot proprioception only; simulator
  object state remains confined to the critic or frozen teacher.
* Added one state-based KUKA-Allegro task with 8 cm cubes, independent control
  and proprioception for all 7 arm and 16 hand joints, and a wrist-diverse
  65,536-row reset table.
* Added validated physical reset tables with randomized table layouts,
  intermediate manipulation states, role permutations, and target-rate
  sampling through the shared success monitor.

Changed
^^^^^^^

* Configured both stack robots for Newton physics with explicit contact
  materials, measured-state residual actions, gravity compensation, and
  manipulation-specific solver capacities.
* Added a visible-top contact surface for the Seattle table and native colored
  cuboids with semantic labels for consistent physics and rendering.
* Centralized episode reset metadata in one typed runtime-state owner shared by
  observations, rewards, success contexts, and curriculum terms.
* Configured the camera task with a reset-safe two-frame 128 by 128 RGB history,
  fixed visible cube roles, camera-calibration randomization, and a
  spatial-softmax policy encoder.
* Configured distillation with recipe-balanced behavior cloning, per-step
  DAgger collection, type-appropriate arm and binary-gripper losses, auxiliary
  visual-state supervision, and held-out success metrics for every reset phase.
* Configured the KUKA-Allegro policy to observe the complete hand state,
  fingertip geometry, and continuous cube orientation while reset-authored
  grasps use the validated index-thumb pinch.

Fixed
^^^^^

* Fixed stack completion to require a stable, released three-cube tower and to
  terminate immediately after the physical hold, preventing reward cycling.
* Fixed reset-authored grasps, action target handoff, and reset sampling hot
  paths so vectorized physical states remain valid without per-reset host
  synchronization.
* Fixed camera distillation ambiguity by making cube roles identifiable from
  RGB and excluding held-out evaluation episodes from adaptive reset evidence.
* Fixed non-finite state handling and task-space bounds so one invalid world
  cannot contaminate a complete policy rollout.
