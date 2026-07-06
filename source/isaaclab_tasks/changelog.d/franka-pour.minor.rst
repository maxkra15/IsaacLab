Added
^^^^^

* Added the ``Isaac-Pour-Franka-v0`` (``-Play-v0`` / ``-Teleop-v0``) contributed task: a Franka
  grasps a small hollow-cube bowl full of granular MPM media and pours it into a fixed target bowl
  using Newton-generated rigid contacts resolved by a proxy-coupled MJWarp and implicit-MPM
  solver, with isolated per-environment sparse grids, CUDA graph replay, capacity scaling, and
  selective reset.
* Added a success-driven five-stage backward curriculum for Franka Pour that progresses from
  pouring a grasped cup to the complete grasp, lift, carry, and pour task, teaches fixed-cup
  stand-off reach and grasp, then adds collision-safe source/target placement randomization with a
  prevalidated Newton-IK reset bank while preserving a fixed robot base and asynchronous resets.

Changed
^^^^^^^

* Changed Franka Pour cups and media to scene-owned assets, kept solver-only collision proxies in
  the per-world Newton builder hook, and added selective public-state reset plus two-world Kit and
  Newton visualization coverage. Registered task users need no migration; custom scripts should
  access the cups through ``env.scene["source_cup"]`` and ``env.scene["target_cup"]``.
* Changed the Franka Pour RSL-RL runner to log to the ``franka-pour-mpm`` W&B project by default;
  set ``WANDB_MODE=offline`` to retain runs locally without syncing.
* **Breaking:** Changed Franka Pour training from a seven-dimensional Cartesian DiffIK action to a
  nine-dimensional phase, joint-residual, and gripper action, added joint-limit clipping, and made
  reward shaping depend on measured cup motion. Existing checkpoints must be retrained. The
  separate eight-dimensional ``-Teleop-v0`` preset accepts seven joint targets plus an operator
  gripper command through ``teleop_franka_pour_spacemouse.py``.
* **Breaking:** Changed the Franka Pour policy observation ABI from a 91-dimensional simulator-state
  vector to a 72-dimensional sensor-compatible vector with canonical cup-relative grasp geometry
  and individual finger state, and moved 20 dimensions of exact media and dwell state to an
  asymmetric critic group. Existing checkpoints must be retrained.
* Changed Franka Pour particle shaping to signed progress toward the active held-delivery target:
  credit is capped at that target, repaid when particles leave the receiver or an attempt fails,
  and retained only by the same stable success used for curriculum advancement. Each table-level
  spill is still penalized once and an environment terminates after more than ten percent spill.
* Changed Franka Pour shaping to bounded signed open-approach, contact-grasp, lift, and alignment
  potentials, replaced persistent and one-sided tilt rewards with target-directed signed tilt
  progress through a validated 150-degree drain, added a held-only stage-zero reference for its
  validated joint trajectory, projected the supplied-grasp pour/carry stages onto safe one-way
  trajectories so the first curriculum problem is a scalar tilt while lift and later stages retain
  full joint control, and retained 10 percent of resets on the preceding curriculum stage.
* Changed Franka Pour gripper control from a Gaussian-thresholded binary command to a continuous
  reset-relative symmetric finger target, exposed reset-relative origins and filtered joint-target
  state to the actor, and bounded PPO actions.
* Changed Franka Pour training to use the stock RSL-RL runner and an exact 4,096-outcome rolling
  curriculum window with stage, randomization, success, reset-bank, delivery, and mastery metrics.
  Its PPO update uses the standard Franka base learning rate and update depth with a fixed schedule
  suited to the task's low-KL residual policy, plus the standard Gaussian action distribution and
  a small entropy bonus. The 3,000-iteration curriculum budget is calibrated for
  ``--num_envs 512``. Resume training at a saved stage by setting
  ``env.curriculum_start_stage`` to the last logged stage.
* Enlarged the Franka Pour source cup to a 60 mm square outer footprint while keeping its visible
  geometry, particle collider, and grasp proxy aligned and within the Panda gripper opening.
* Changed Franka Pour MPM collider sampling to the particle-bounded ``pic27`` basis while retaining
  the compact Q1 velocity solve.
* Changed the receiving cup to one scene-owned rigid body proxied into MPM, so rendering, rigid
  contacts, particle collision, and randomized resets share one authoritative pose.
* Shortened Franka Pour training episodes from 12 seconds to 5 seconds so failed attempts recycle
  promptly.

Fixed
^^^^^

* Fixed the source cup's rigid grasp proxy to match the visible cup instead of extending above its
  rim and creating phantom contacts.
* Fixed Franka Pour startup memory scaling by deriving sparse-grid capacity from its fixed particle
  count and sharing the Q1 velocity space with particle colliders, eliminating a capacity-sized
  cross-basis interpolation matrix.
* Fixed long-running Franka Pour training failures by bounding finite particle excursions and
  reserving upper sparse-grid hierarchy nodes independently from active cells.
* Fixed non-finite terminal transitions contaminating Franka Pour rewards before the affected
  environment is selectively reset.
* Fixed extreme but finite rigid-body states reaching the policy by terminating and selectively
  resetting affected environments before observations are returned.
* Fixed success, reward, termination, and curriculum disagreeing on one-step particle transfers by
  sharing per-particle held-delivery state in one failure-aware success predicate with a
  configurable 0.15-second dwell interval.
* Fixed finite-horizon control ambiguity by exposing remaining time, the lost-grasp debounce, and
  the active delivery threshold to the policy, and made successful termination mutually exclusive
  with timeout.
* Fixed finite-horizon RSL-RL startup assigning shortened deadlines without corresponding physical
  states, and exposed the randomized reset bank's applied filtered joint reference to the policy.
* Fixed transient or airborne particle dumps retaining delivery credit at an unsuccessful timeout;
  every unsuccessful completion now receives the same one-time failure penalty.
* Fixed scale-dependent MPM instability in large training batches by colocating isolated physics
  worlds as intended by environment-partitioned FEM topology; worlds still cannot interact, while
  small viewer layouts retain their requested spacing.
* Fixed command-line overrides for Franka Pour MPM resolution, iteration count, substeps, coupling,
  proxy settings, and CUDA graph use being ignored by the nested Newton solver configuration.
* Fixed open fingers passively compressed by source-cup contact being mistaken for a grasp by
  requiring commanded preload in grasp shaping and requiring the cup to remain preloaded, held, and
  lifted for alignment and particle-delivery reward and throughout the success dwell interval.
* Fixed early curriculum resets commanding a fully closed gripper on the first policy step,
  retained the authored preload throughout pre-grasped curriculum stages, scaled continuous
  gripper actions to its physical travel, and filtered high-frequency arm and gripper targets
  before the stiff coupled drives.
* Fixed nested source and target cups counting source-contained particles as delivered; a particle
  now contributes to target observations, reward, success, and curriculum only after leaving the
  source cup.
* Fixed curriculum promotions saturating the policy with zero-variance empirical-normalization
  outliers by replacing running actor/critic normalization with fixed physical observation scales.
* Fixed the randomized curriculum exposing full receiver displacement and reset-arm jitter at its
  first level by defining each frontier over the combined source, receiver, and TCP perturbation;
  later levels replay the immediately preceding randomization frontier instead of the fixed-layout
  stage.
