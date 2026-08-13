Added
^^^^^

* Added an RSL-RL task for learning order-invariant, end-to-end three-cube Franka stacking with
  direct relative joint-position control and a binary gripper.
* Added a validated reset-state table with randomized deployment starts,
  physical intermediate states, cube-role permutations, and target-rate
  sampling through the shared success monitor.
* Added a Newton-MJWarp configuration with DexSuite-calibrated Franka impedance, model-based
  gravity compensation, and manipulation-specific contact settings.
* Added one Kuka-Allegro stack task with lightweight 8 cm cubes, continuous
  control and complete proprioception for all 23 arm and hand joints, and a
  65,536-state adaptive reset table spanning three thumb/opposing-finger
  grasp pairs, broad wrist rotations, and randomized cube layouts.
* Added a Franka RGB-camera training variant with stationary image scaling, a
  spatial-softmax actor, an asymmetric privileged critic, and only
  real-controller proprioception exposed to the actor for sim-to-real use.
* Added a three-stage Franka sim-to-real workflow: the proven state policy is
  used as a frozen teacher, an RGB-plus-proprio student is distilled through
  behavior cloning, and the compatible student actor is then fine-tuned with
  asymmetric PPO without exposing simulator state at deployment.

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
* Changed camera-policy distillation to use an identifiable visible role
  convention (blue base, red first, green second), a short behavior-cloning
  warmup, and globally annealed DAgger with persistent episode controllers.
  Supervision is balanced across reset recipes and uses type-appropriate arm
  and binary-gripper losses.
* Changed the deployable camera observation to a reset-safe two-frame RGB
  history, exposing object motion without simulator state, and report
  deterministic student success from held-out starts for every reset recipe.
* Changed the Franka camera input to 128 by 128 pixels and bounded its
  teacher-only pretraining period so DAgger collects corrective labels from
  student-visited states as soon as the easiest phase is learnable.
* Removed inactive stack-RL experiments and their compatibility exports.
* Split the Kuka-Allegro acquisition curriculum into a sparse TABLE
  grasp/lift ladder while FIRST_PICK resets retain the independently screened
  25 mm lift target.
* Made the TABLE-row curriculum target configurable for staged acquisition
  training while preserving each stack task's existing default.
* Removed the unused below-table ground plane from the Kuka-Allegro task so
  headless Newton training no longer resolves an
  external environment USD.
* Changed Kuka-Allegro arm actions to use matching 0.12 rad scaling and target
  limits, removing hidden many-to-one saturation from PPO's action contract.
* Changed stack event rewards to exact timestep-independent episode impulses
  while preserving their established values at the existing control rate.

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
* Fixed camera distillation's hidden reset-time cube-role permutation, which
  previously asked identical RGB observations to imitate incompatible teacher
  plans, and excluded held-out evaluation episodes from adaptive reset
  sampling.
* Fixed visible Franka cube/table overlap by adding a native contact surface
  aligned with the Seattle tabletop instead of relying on its 3 mm recessed
  collision proxy.
* Fixed the Kuka-Allegro cubes to use the same visual-top contact surface and
  geometric resting height. Existing Kuka-Allegro policies may require a short
  fine-tune for the corrected table contact and arm-action mapping.
* Fixed native cube replacement to retain semantic labels, migrated the task
  camera hint off the deprecated viewer configuration, and added validation
  for Newton timing, solver capacity, action, contact, and camera contracts.
