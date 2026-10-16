<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Franka Pour

This package provides one artifact-owned reset-dataset learning workflow,
`IsaacContrib-Franka-Pour`.

The reset-dataset task follows the same separation of concerns as Franka Stack:

- the actor receives a terminal-sparse task reward;
- one offline cache supplies a connected path through the task;
- a separate Boolean local-progress signal trains the reset sampler; and
- terminal particle transfer remains the policy-success and evaluation metric.

The actor receives one current timestep of robot, cup, target, gripper, and previous-action state,
plus compact permutation-invariant MPM summaries. Its 84 inputs contain no temporal stack or
constant padding. The asymmetric critic adds two task-state inputs for a total of 86.

The artifact is still required for the reset-dataset task because it is the reset curriculum, not
a cache of simulated fluid outcomes. It amortizes NewtonIK and static collision rejection, gives
every physical start a stable row ID for outcome-aware replay, and binds training runs to one
reproducible rigid-state distribution. Runtime reuses one configured source-local particle lattice
across reset rows and resets the complete MPM solver state; it does not replay cached stress or
particle trajectories.

## Train and regenerate

The task downloads the canonical 20,000-row artifact from the standard Isaac Lab asset root on
first use and reuses the local asset cache afterward. Training therefore needs no artifact setup:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 --max_iterations 3000
```

The checked-in generator remains the executable reference for creating or modifying the reset
distribution. It takes about two minutes on an L40S-class GPU and writes a 20,000-row local artifact:

```bash
uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0

uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --num_envs 2048 --device cuda:0 \
  env.reset_dataset_path=datasets/franka_pour/reset_dataset.pt
```

Relative local paths resolve from the repository root. The generator exposes the same 14 connected
reset phases and phase quotas as the published distribution. Newton IK, joint-limit, table/object,
self-collision, grasp-seating, and particle-workspace checks reject unsafe proposals; there is no
second validation or artifact-promotion command.

The generator and task load the Franka Pour cups USD from the standard Isaac Lab Nucleus asset
root. Set `ISAACLAB_FRANKA_POUR_CUPS_USD_PATH` only to use a compatible local copy instead. The
published reset artifact lives alongside those assets at `Contrib/MPM/Pour/reset_dataset.pt`.
Setting `ISAACSIM_ASSET_ROOT` redirects all three dependencies to a compatible local or self-hosted
asset tree. The reset artifact records the Franka path relative to the Isaac Lab asset root so its
task contract does not depend on the configured asset host. Regenerate custom reset datasets made
with an older task version so they use the root-relative contract.

The `datasets/` directory is ignored by Git. Both downloaded and locally generated payloads validate
their stored content digest, so no digest override is needed for ordinary training or playback. For
an exactly reproducible custom run, the generator also prints the optional
`env.reset_dataset_content_sha256` pin.

## Fill and success levels

`env.source_fill_level` is the initial media height as a fraction of the source-cup cavity. It
defaults to `0.70` and accepts values in `(0, 1]`. The particle count is derived from the requested
height, quantized to complete lattice layers; it is not a separate fixed task setting. The 15 mm
voxel and 6 mm particle spacing stay fixed across fill levels. Although the reset artifact records
particle layouts, runtime restores its rigid start states with the currently configured lattice.
Changing the fill therefore does not require regenerating the artifact. This analytic fill volume
is non-colliding. The visible cup mesh remains a particle-only collider so it can physically contain
the media. The receiver uses a solid analytic box only for robot contact; it is invisible to particles,
which continue to collide with the hollow receiver mesh. An invisible particle-only plane at ground
height catches media falling off the table without adding rigid contacts or visible geometry.

The particle generator and MPM solver both use a 15 mm voxel. Setting
`particles_per_cell=2.5` gives 6 mm particle spacing at every fill level, while the
bounded sparse grid, particle-backed automatic warm start, and two MPM entry substeps keep
the configuration CUDA-graph compatible. Collider projection remains disabled because manager-level
post-step projection is not supported inside a coupled MPM entry; contact is resolved by the MPM
solve.

The proxy presents the supported or grasp-constrained 50 g source cup as 50 kg to the MPM collision
view. This prevents the roughly 140 g payload from producing unphysical rigid-cup recoil without
changing the cup's authored mass or the robot dynamics in MJWarp.

`env.pour_target_frac` independently sets the fraction of the initial payload that must be inside
the receiver. Success is normalized by the live particle count, so changing the source fill does
not require changing the success check. For example, this starts half-full and requires 80% of that
payload to be delivered:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour --device cuda:0 \
  env.source_fill_level=0.50 env.pour_target_frac=0.80
```

## Play

The canonical task is registered with import-light string entry points. Registration does not
download the artifact; environment startup retrieves and validates it when creating the environment.
Play mode uses one bounded sparse-grid world, disables observation corruption, freezes the sampler,
and moves the interactive viewer closer:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Franka-Pour \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --device cuda:0 --visualizer kit
```

Use `--visualizer newton_gl` instead for the lightweight Newton viewer. No external callback or
particle-setting override is required for either visualizer.

## Material fidelity and throughput experiments

This feature branch pins Newton fork commit `d183ffaad4d49a74f75a0bdc70e6c31d86d3f58b`
and Warp 1.18.0 in `pyproject.toml` and `uv.lock`. Normal `uv run` commands resolve these
dependencies; no source-path overlay is required.

An experimental Newton build with `SolverImplicitMPM.Fidelity` supports an approximate
position-based particle stage through `MPMSolverCfg(fidelity="particles", particle_iterations=2)`.
The full material remains `fidelity="mpm"`. Both modes use the same scene particles, ordering,
masses, policy observations (84 actor / 86 critic), eight actions, task reward, and reset dataset.
The robot and cups continue to use the same MuJoCo entry and proxy-coupling settings. Particle
contacts return approximate reaction wrenches through the existing proxy interface; they do not
reproduce MPM's continuum constitutive forces. Match the modes with task measurements before
treating them as interchangeable training distributions.

The default fill creates 468 particles with approximately 2.906 mm radius and 137.8 g total mass.
Finite-radius particle contact places their centers approximately 6.756 mm from the cup mesh,
compared with MPM's 3.85 mm signed contact offset. This reduces the source's available center
width from 34.3 mm to 28.488 mm, so packing and outflow differ even with identical particle arrays.
The approximate pair friction defaults to 0.5, while this task's MPM internal friction is 0.7;
these describe different contact and constitutive models. Use `--particle_friction 0.7` to test
an explicit fit, and compare two versus four contact iterations before adjusting attraction.

The MPM adapter converts convex meshes to particle-compatible triangle meshes only when their
collision flags include particles. Robot hulls that collide only with other rigid shapes retain
their authored convex representation. Both material modes use this adapter correction; its
effect on shared-scene collision work is separate from the material-solver optimization.

The current proxy-coupling implementation harvests feedback from the final media substep. With
the task's two media substeps, it does not accumulate every impact impulse across that interval.
Both compared modes retain this existing limitation. These experiments establish throughput and
policy-interface compatibility; they do not calibrate absolute robot reaction forces.

The standalone workflow applies these settings to the existing `media` entry, keeping default
MPM's `pic27` contacts and 24 Jacobi sweeps as its baseline. Run each case in a fresh process:

```bash
uv run python scripts/benchmarks/benchmark_franka_pour_fidelity.py \
  --mode mpm --num_envs 64 --checkpoint /path/to/model.pt \
  --output logs/mpm_fidelity/mpm-64.json

uv run python scripts/benchmarks/benchmark_franka_pour_fidelity.py \
  --mode particles2 --num_envs 64 --checkpoint /path/to/model.pt \
  --output logs/mpm_fidelity/particles2-64.json
```

Use `--mode particles4` to test a larger particle-contact iteration budget. `--mode s2` and
`--mode s2_cached` compare S2 MPM contacts with the same Gauss-Seidel budget
(`--mpm_iterations 5` by default), changing only the optional spatial contact-impulse cache.
The task's existing `pic27` basis already retains particle contact history. The new S2 cache is
therefore a separate numerical experiment, not an automatic change to the checkpoint baseline.

The JSON records environment transitions per second and policy batches per second separately.
Rollout timing includes policy inference, physics, observations, rewards, resets, and curriculum
work after warmup. `--train_iterations 5` also continues PPO from the checkpoint and reports total
training throughput including optimization, logging, and final checkpoint saving. Run progressively
larger environment counts in separate processes until memory or latency becomes limiting; a
successful short run does not establish long-horizon task quality. Initial particle mode avoids
allocating the MPM grid until full MPM is first requested.

The reports include terminal success and spill counts, finite-state and movement checks, and the
final mean particle fractions in each cup and the spill region. These final fractions describe
the current episodes after automatic resets; a freshly reset full source cup does not establish
that earlier episodes avoided spilling. Source fingerprints and checkpoint hashes identify
archived deployments even when their Git metadata is absent.

Record the same checkpoint and seed in separate one-world runs with `--video --num_envs 1
--steps 360 --freeze_reset_sampler`. Clips are written alongside each JSON result. Rendering is
included in that run's timing, so use non-video runs for throughput comparisons.

For an existing environment, the public manager exposes transitions between physics steps:

```python
from isaaclab_newton.physics import NewtonMPMManager

NewtonMPMManager.set_fidelity("particles", entry_name="media")
NewtonMPMManager.set_voxel_size(0.012, entry_name="media")
NewtonMPMManager.set_particle_radius(0.003, entry_name="media")
NewtonMPMManager.set_fidelity("mpm", entry_name="media")
```

The manager resolves the current entry-local state and schedules CUDA graph capture again.
Voxel changes preserve material history; fidelity handoffs reset stress and deformation history
while preserving positions and velocities. A uniform radius change preserves particle mass and
count, so MPM density changes as `1 / radius**3`. It does not split or merge particles. Radius
changes currently require the selected MPM entry to own all scene particles.

An optional success-gated curriculum can perform these transitions inside one training run:

```python
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import PourMediaFidelityStageCfg, _resolve_pour_solver_tree

cfg.media_fidelity_min_episodes = 128
_resolve_pour_solver_tree(cfg).media_solver.collider_basis = "S2"
cfg.media_fidelity_stages = (
    PourMediaFidelityStageCfg(success_rate=0.70, fidelity="mpm", voxel_size=0.020),
    PourMediaFidelityStageCfg(success_rate=0.80, voxel_size=0.015),
)
```

Each stage checks a cohort of completed episodes against actual terminal particle-transfer
success. It is independent of the reset sampler's local-progress signal. A transition applies to
all worlds at a reset boundary and retains the policy and optimizer. Empty stages disable the
curriculum, and frozen playback does not advance it. Distributed ranks currently evaluate their
own cohorts. The workflow offers a one-stage CLI variant with `--switch_success_rate 0.70
--switch_voxel_size 0.015`; `--switch_particle_radius` is available for explicitly mass-fixed radius
experiments. Changing solver voxel spacing does not change the configured particle spawn lattice.
Use an S2 contact basis for voxel curricula (`--collider_basis S2` in the workflow). Coarsening
the grid can exceed the fixed 27-particles-per-cell bound of the default `pic27` contact basis;
the default bound is not a general guarantee for a changed grid or denser packing. Add
`--warmstart_contact_impulses` to enable the S2 contact cache in an initially particle-based run.

## Learning contract

The actor reward contains only:

- `+5` on terminal particle-transfer success;
- `-1` on a non-timeout safety failure;
- `-1e-4` action magnitude; and
- `-1e-4` action-rate regularization.

An ordinary timeout is neutral. Nonfinite state, extreme rigid state, source/receiver overlap,
spill, particle out-of-bounds, and the configured collision guards remain terminal safety checks.
Success terminates on the first policy step with the configured fraction of particles delivered to
the receiver (70% by default).

The sampler does not use terminal success as its learning signal. For each restored row, a
non-terminating context asks whether the policy made meaningful forward physical progress from
that starting state. Near the end of the manifold it requires actual particle-transfer success.
The result latches after a safe milestone crossing and is recorded when that episode finishes;
spill, overlap, out-of-bounds, collision, or abnormal terminal transitions cannot create positive
curriculum evidence. This curriculum-only Boolean never contributes policy reward.

Each row retains its exact 50 most recent local-progress outcomes. Adaptive priority is
`sqrt(p * (1 - p)) + epsilon`, so uncertain rows near 50% progress are emphasized without starving
rows at 0% or 100%. A shuffled cyclic stream receives 50% of assignments throughout training,
which keeps persistently difficult reaching and grasp-acquisition rows represented alongside the
adaptive competence frontier. There are no runtime stages or region probabilities.

The sampler reports its effective pool size, concentration, and cyclic-replay fraction. Offline
reset regions remain in the artifact as generation provenance, but runtime does not maintain a
second set of per-region counters. Evaluate frozen policies separately rather than inferring
performance from training-time samples.

Each worker applies completed outcomes immediately to its own sampler. In distributed training,
the curriculum therefore adapts independently on each rank while PPO still learns from all rank
rollouts. Actor and critic empirical observation normalization is disabled because task
observations are physically scaled at their source.
