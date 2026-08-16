<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka RJ45 insertion

`IsaacContrib-Franka-RJ45-Insertion` combines Newton's mechanically latched RJ45
plug example with the reset-dataset learning workflow used by the contributed
Franka tasks. The socket, plug, latch, source cable curve, SDF settings, cable
rod, four plug-relative cable anchors, pinned tail, latch joint, and contact
parameters come from Newton commit
`7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a`. The unmodified source USD and its
Apache-2.0 attribution are kept in `physics/assets/`.

The task adds a Menagerie Franka driven by MJWarp and couples its hand and finger
collision proxies into the VBD-owned RJ45 assembly. A massless, invisible,
finger-only collider covers the plug's rear housing so the gripper can transfer
force without changing the original plug/socket/latch contact geometry. The
original connector remains frictionless, while the cable retains Newton's
friction and bending behavior. The assembly also retains Newton's local support
plane for the cable; every Franka collider is filtered from that plane so it
does not become part of the robot dynamics.

## Reset artifact

Training intentionally refuses to start without a physically generated reset
artifact. From the repository root, generate the fixed goal and progressively
harder near-goal rows, then replay every row:

```bash
uv run python scripts/tools/generate_franka_rj45_reset_dataset.py \
  --headless --device cuda:0

uv run python scripts/tools/validate_franka_rj45_resets.py \
  --headless --device cuda:0
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

## Train

```bash
uv run python scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl \
  --task IsaacContrib-Franka-RJ45-Insertion \
  --num_envs 2048 --device cuda:0 --max_iterations 3000 --headless
```

Training and validation share the project log root
`logs/rsl_rl/franka_rj45_insertion/`. Policy reward is terminal-sparse: stable
fully seated insertion receives the success pulse, unsafe termination receives
a failure pulse, and only small action magnitude/rate costs are otherwise
applied. A separate non-reward progress monitor supplies row-local evidence to
the adaptive reset sampler. The policy uses Gaussian arm exploration and a true
Bernoulli gripper distribution so the binary action has the correct likelihood.

Isaac Lab `develop` currently pins Newton 1.5. The task requests the legacy
`rigid_contact_hard=False` compliant formulation on that release and also sets
`rigid_compliant_alm=True`, which the signature-aware manager forwards when run
with Newton versions exposing the newer unified compliant-ALM API. The reset
artifact records the complete physics/version contract and is rejected after an
incompatible change.

The Franka impedance gains and reset-relative target semantics follow
`maximiliank/franka-newton-stack`. Model inverse-dynamics gravity feedforward is
disabled because Newton 1.5 rejects inverse dynamics on a global model that
contains cable joints; the recorded target-minus-measured reset bias preserves
the demonstrated gravity-loaded equilibrium instead.
