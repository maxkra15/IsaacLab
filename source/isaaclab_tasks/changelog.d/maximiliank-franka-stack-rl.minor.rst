Added
^^^^^

* Added an RSL-RL task for learning order-invariant, end-to-end three-cube Franka stacking with
  direct relative joint-position control and a binary gripper.
* Added a validated reset-state table with randomized deployment starts, physical intermediate
  states, cube-role permutations, and adaptive epsilon sampling.
* Added a Newton-MJWarp configuration with DexSuite-calibrated Franka impedance, model-based
  gravity compensation, and manipulation-specific contact settings.
* Added one Kuka-Allegro stack task with lightweight 8 cm cubes, continuous
  control and complete proprioception for all 23 arm and hand joints, and a
  65,536-state epsilon reset table spanning three thumb/opposing-finger
  grasp pairs, broad wrist rotations, and randomized cube layouts.

Changed
^^^^^^^

* Changed policy playback to use the environment configuration's
  :meth:`~isaaclab.envs.ManagerBasedRLEnvCfg.play_mode` override.
* Consolidated branch-specific stack registration to one Franka RL task and
  one Kuka-Allegro RL task. The Kuka task retains the trained large-cube,
  diverse-reset, full-hand observation and action interfaces.
* Changed reset and success-context state to use typed owners while retaining
  the initial environment tensor names as compatibility aliases.
* Changed cube spawning to author display colors on standard cuboid geometry
  instead of maintaining duplicate collision and render cubes.
* Removed inactive stack-RL experiments and their compatibility exports.
* Split the Kuka-Allegro acquisition curriculum into a sparse TABLE
  grasp/lift ladder while FIRST_PICK resets retain the independently screened
  25 mm lift target.
* Made the TABLE-row curriculum target configurable for staged acquisition
  training while preserving each stack task's existing default.
* Removed the unused below-table ground plane from the Kuka-Allegro task so
  headless Newton training no longer resolves an
  external environment USD.

Fixed
^^^^^

* Fixed stack completion to require a stable, released three-cube tower and terminate immediately,
  preventing transient contacts and post-success reward cycling.
* Fixed reset-authored grasps, joint residual targets, and Newton contact capacities so vectorized
  reset states remain physically valid throughout policy rollouts.
* Fixed reset and adaptive curriculum hot paths to avoid per-reset CUDA-to-CPU
  synchronization and duplicate sampling-distribution work.
* Fixed Kuka-Allegro diverse resets to decorrelate cube layouts from wrist
  modes, preserve authored reset strata after IK repair, cover the
  first-cube release/retract transition, and apply validated two-finger
  contact preload without reset-time interpenetration.
* Fixed large-cube PICK reset validity by separating table-supported partial
  closure rows from dynamically screened suspended-grasp rows.
* Fixed the full-hand preload handoff with a one-way reset-only grasp anchor
  while retaining DexSuite's measured-state relative action semantics, so
  reset grasps remain stable until a deliberate release without an unobserved
  target integrator saturating.
