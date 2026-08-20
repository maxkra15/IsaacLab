<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka RJ45 tasks

This package provides two distinct reset-driven tasks. The original
`IsaacContrib-Franka-RJ45-Insertion` starts with the cable already grasped near
the socket. `IsaacContrib-Franka-RJ45-Pick-Insert` adds the complete open-hand
approach, grasp, transport, alignment, and insertion problem described below.

Both tasks combine Newton's mechanically latched RJ45 plug example with the
reset-dataset learning workflow used by the contributed Franka tasks. The
socket, plug, latch, source cable curve, SDF settings, cable rod, four
plug-relative cable anchors, pinned tail, latch joint, and contact parameters
come from Newton commit
`7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a`. The unmodified source USD and its
Apache-2.0 attribution are kept in `physics/assets/`.

Both tasks add a Menagerie Franka driven by MJWarp and couple its hand and
finger collision proxies into the VBD-owned RJ45 assembly. A massless,
invisible, finger-only collider covers the plug's rear housing so the gripper
can transfer force without changing the original plug/socket/latch contact
geometry. The original connector remains frictionless, while the cable retains
Newton's friction and bending behavior. The insertion-only assembly retains
Newton's local support plane for the cable; every Franka collider is filtered
from that plane so it does not become part of the robot dynamics.

## Full pick-and-insert task

`IsaacContrib-Franka-RJ45-Pick-Insert` is a separate long-horizon task, not a
play configuration of the insertion-only environment. Its task state contains
48 bodies in the fixed order socket, plug, latch, then 45 cable segments. The
cable keeps Newton's original 35 segments and adds a ten-segment tail so a
randomized pickup can be transported without shortening the source geometry.
The socket is resettable, plug rotation is free, and insertion and success are
measured in each randomized socket goal frame.
The actor observes seven approximately arclength-uniform cable centers and
their linear velocities in that same socket frame, so the extended cable's
shape and motion remain available during transport.

The pick-only arm action follows the reset-driven Franka-stack controller: each
policy delta is EMA-filtered and integrated exactly once per control step into
a persistent absolute joint target seeded from the stored reset actuator
target. A literal zero action clears the EMA tail and holds that absolute target
bitwise. Per-joint target-error limits terminate non-finite or untrackable
commands. The pick-only Franka enables native MJWarp joint
`actuatorgravcomp`; action-level inverse dynamics and global-model RNEA remain
disabled, so the cable model is never passed through inverse dynamics. A 7D
arm-target-error term exposes the controller state to the policy, making the
actor input 135D and the asymmetric critic input 138D. These choices are
fail-closed in task contract 6, pick-insert semantics 6, reset-dataset schema 3,
and validation-report schema 5.

The scene uses the Seattle Lab table USD at the Franka-stack pose
`pos=(0.5, 0.0, 0.0)`, `rot=(0.0, 0.0, 0.707, 0.707)`. It is made recursively
editable and all authored collisions are disabled, so the imported asset is
visual only. Contact comes from a separate invisible kinematic cuboid at
`pos=(0.3439, 0.0, -0.02)` with size `(1.28, 0.91, 0.04)` m, whose top face is
at `z=0`. Its Newton material uses static/dynamic friction `1.0/0.8`, zero
restitution, torsional/rolling friction `0.002/0.0001`, contact stiffness
`1.0e4` N/m, and contact damping `200.0` N s/m. This slab replaces the original
task-local support plane for the pick-and-insert variant.

### Pinned external scene assets

The Franka and Seattle table remain NVIDIA-authored external assets and are not
vendored into this repository. Production pickup generation, validation,
training, and playback require their exact 19-file Isaac 6.1 dependency closure
(83,718,325 bytes) at a stable local path. Materialize it from an existing local
Isaac 6.1 asset tree without network access, then export the task-specific root:

```bash
uv run python scripts/tools/prepare_franka_rj45_asset_closure.py materialize \
  --source-tree /path/to/Assets/Isaac/6.1 \
  --cache-root "${XDG_CACHE_HOME:-$HOME/.cache}/isaaclab/asset-closures"

export ISAACLAB_FRANKA_RJ45_ASSET_CLOSURE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/isaaclab/asset-closures/sha256/060d50ba665ae9850a8ef19c1b42a2a55579d4f8924cadfc37c1ec83ca726a76"
```

Environment startup verifies every byte and rejects missing, modified,
symlinked, or extra entries before scene construction. Reset artifacts store
only the normalized `external_assets` contract v1 (logical URIs plus the tree
digest), never a host-local cache path. Configuration discovery and diagnostic
contract inspection remain available without the environment variable.

### Six reset phases

The generator creates the same number of rows for each phase (96 per phase by
default). All rows are physically settled and robot-recoverable; cable bodies
are never independently perturbed.

| Phase | Name | Initial gripper state | Reset meaning |
| --- | --- | --- | --- |
| 0 | `near_insertion` | Grasped | Plug is 6 mm short of its seated goal on the goal-local +Y insertion axis. |
| 1 | `preinsertion` | Grasped | Plug is 30 mm short of the seated goal on that axis. |
| 2 | `transport` | Grasped | Plug is lifted along a midpoint path between pickup and pre-insertion. |
| 3 | `postgrasp` | Grasped | Plug has just been lifted 25 mm from the pickup pose. |
| 4 | `pregrasp` | Open | TCP is 45 mm above the plug and must acquire a physical grasp. |
| 5 | `full_pick` | Open | Franka starts near its noisy home pose, away from the randomized pickup. |

Normal adaptive training reserves an exact long-run 35% of resets for phase 5
`full_pick` rows. The remaining 65% comes from phases 0--4 using the
success-aware continuation sampler; its own uniform replay fraction is a
separate setting. This keeps complete deployment starts represented even when
the competence frontier concentrates on later phases.

### Generate and validate pick-and-insert resets

Run the generator and independent validator from the repository root:

```bash
uv run python scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py \
  --visualizer none --device cuda:0

uv run python scripts/tools/validate_franka_rj45_pick_insert_resets.py \
  --visualizer none --device cuda:0
```

The generator writes
`datasets/franka_rj45_pick_insert/reset_dataset.pt`. It first constructs a
central fully seated state using Newton's +35 mm insertion drive, disables the
drive, passively settles it, and proves at least 60 simulated seconds of
canonical cold, drive-free replay. Each row then stores all 48 body poses and
velocities, both ordered VBD previous-pose buffers, robot state and persistent
absolute actuator targets, its phase, and a rigidly transformed copy of that
canonical goal at the sampled socket pose. Reset first materializes public
robot/task state and keeps only the latest staged histories per environment.
Immediately before the next policy step it queues both named-entry VBD
histories; each is applied exactly once after normal input and proxy
rebaselining and before that world's first VBD solve. The independent validator
requires exact entry/body/world ordering plus one world application and 48 body
applications for the canonical goal and every row.

The pick topology also preserves the direct-VBD input semantics of the cable's
44 authored moving bodies. Anchor synchronization and capsule orientation
alignment are accepted as pose inputs without adding pose-delta velocity or
rewinding the bodies. Capsule alignment runs immediately after every coupled
solve and before state swapping or collision generation; the fixed terminal
body is excluded. This projection is pick-only, and the legacy insertion task
keeps its existing coupling and alignment behavior unchanged.

Open pickup rows use the true 0.04 m finger posture. Acquisition is
fail-closed: the robot first lifts to the 0.22 m safe world-height route, moves
overhead in at most 50 mm Cartesian steps, descends coarsely to the measured
45 mm pregrasp, calibrates there, and then advances in 1 mm guarded steps.
Every open sample must remain collision-free, proxy-contact-free, drive-free,
finite, and within 0.5 mm plug drift. Closing takes 0.8 s, holds for 1.5 s,
and must establish bilateral proxy contact before a further one-second
drive-free settle. Grasped transport preserves the finger preload and follows
staged lift, midpoint, overhead, and destination waypoints with at most 2 mm
translation, 2 degree rotation, and 0.02 rad raw-IK joint increments; no hidden
plug or cable drive is enabled.

Every validator invocation writes a timestamped evidence report under
`logs/rsl_rl/franka_rj45_pick_insert/validation/` named
`reset_validation_YYYYMMDDTHHMMSSZ_<digest12>_full.json` (or `_quick.json` for
quick validation). Only a passing, non-quick replay of the fixed goal and every
dataset row atomically publishes the stable training gate:

```text
logs/rsl_rl/franka_rj45_pick_insert/validation/reset_validation.json
```

Quick or sampled validation never updates that stable path. Environment startup
requires the default dataset and stable report to agree on the complete content
digest, row phases, 48-body topology, scene, and task contract. Schema 5 also
binds the immutable single-seed, sampler-free validator policy, the complete
repository source snapshot, the verified external-asset closure, and
independently recomputed phase-5 diversity evidence. Full validation supports
atomic per-batch checkpoints; resuming a completed ``stable-published``
checkpoint strictly verifies and idempotently republishes the same report
before removing the checkpoint unless ``--keep-checkpoint`` was requested.

### Train and play pick-and-insert

The current unified Isaac Lab CLI writes all runs below the project-level
`logs/rsl_rl/franka_rj45_pick_insert/` directory:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-RJ45-Pick-Insert \
  --num_envs 256 --device cuda:0 --max_iterations 8000 --visualizer none
```

Adjust `--num_envs` for the available GPU memory. Playback needs the same reset
artifact and stable validation gate as training:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Franka-RJ45-Pick-Insert \
  --checkpoint latest --num_envs 1 --device cuda:0
```

## Insertion-only reset artifact

Training intentionally refuses to start without a physically generated reset
artifact. From the repository root, generate the fixed goal and progressively
harder near-goal rows, then replay every row:

```bash
uv run python scripts/tools/generate_franka_rj45_reset_dataset.py \
  --visualizer none --device cuda:0

uv run python scripts/tools/validate_franka_rj45_resets.py \
  --visualizer none --device cuda:0
```

The default artifact is
`datasets/franka_rj45_insertion/reset_dataset.pt`. Each row stores the Franka
arm and finger positions, velocities, actuator targets, and all 37 RJ45 body
poses and velocities; reconstructing the cable from the plug pose alone is not
a valid reset. Saving the measured arm position and its loaded actuator target
separately preserves the closed-grasp equilibrium. The goal is a single fixed,
fully seated state. The default bank contains 64 rows in each of five
axial-distance bands from 1--25 mm, with bounded robot, connector, and cable
variation. A row is accepted only after drive-free coupled replay and scripted
Franka recovery.

Goal construction follows the Newton example's insertion oracle: the unplugged
plug is driven 35 mm along +Y with a slow ramp, the generator disables that
drive, and the latched state settles passively for at least ten simulated
seconds. While constructing an unplugged grasp, the generator temporarily holds
the plug at Newton's authored start pose, matching the example's viewer spring.
It then requires bilateral finger contacts, disables the hold, and proves that
the closed Franka alone carries the connector before any snapshot is saved.
VBD contact history is intentionally not serialized. The generator therefore
restores both state buffers, invalidates the solver/contact history, settles the
state, and recaptures it until a subsequent cold replay reaches the same
physical fixed point. This prevents apparently stable warm states from entering
the reset bank with incompatible hidden solver history.
Validation resets both Newton state buffers, clears outer and proxy-local
contact history, replays the fixed goal, and checks every reset for finite
state, normalized quaternions, joint limits, bilateral grasp validity, contact
validity, bounded full-cable drift, and robot-only recoverability. Recovery must
enter the task's exact sparse-success geometry and speed predicate at capture
and remain there at every sample of the 0.15-second runtime-equivalent dwell;
the fixed goal must satisfy that same predicate at every sample of its full
ten-second passive replay. Timestamped
evidence is written below
`logs/rsl_rl/franka_rj45_insertion/validation/`. A successful full replay also
atomically publishes `reset_validation.json` there. Training refuses to load an
artifact unless that stable report proves a non-quick replay of the exact
artifact digest, its fixed goal, and every reset row.

## Train insertion-only task

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-RJ45-Insertion \
  --num_envs 2048 --device cuda:0 --max_iterations 3000 --visualizer none
```

Training and validation share the project log root
`logs/rsl_rl/franka_rj45_insertion/`. Policy reward is terminal-sparse: stable
fully seated insertion receives the success pulse, any unsuccessful termination
or timeout receives a failure pulse, and only small action magnitude/rate costs
are otherwise applied. Every reset is validated as recoverable within the
episode horizon, so this prevents a motionless timeout from becoming preferable
to insertion exploration. A separate non-reward progress monitor supplies
row-local evidence to the adaptive reset sampler. The policy uses Gaussian arm
exploration and a true Bernoulli gripper distribution so the binary action has
the correct likelihood.

Isaac Lab `develop` currently pins Newton 1.5. The insertion-only task requests the legacy
`rigid_contact_hard=False` compliant formulation on that release and also sets
`rigid_compliant_alm=True`, which the signature-aware manager forwards when run
with Newton versions exposing the newer unified compliant-ALM API. The reset
artifact records the complete physics/version contract and is rejected after an
incompatible change.

The shared Franka impedance gains follow `maximiliank/franka-newton-stack`.
The insertion-only task retains its reset-relative targets and recorded
target-minus-measured equilibrium bias. The pick-and-insert task instead uses
the persistent reset-seeded target and robot-scoped native MJWarp gravity
compensation contract described above. Global-model inverse dynamics remains
off for both tasks because Newton 1.5 rejects RNEA on a model containing cable
joints.
