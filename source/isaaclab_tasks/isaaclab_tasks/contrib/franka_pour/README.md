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
  --checkpoint pretrained \
  --num_envs 1 --device cuda:0 --visualizer newton_gl
```

This resolves the [official pretrained MPM policy](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/IsaacContrib-Franka-Pour_newtonmjwarpmpmproxy_none_rsl_rl.pt)
for the task's default physics backend. On October 16, 2026, the published file contained
iteration 7450 with SHA-256
`52cbab93a55a8d153844ba93ad496c69838e1bf1082b4c4f4fc4bba43e3a3643`.
Use `--checkpoint /path/to/model.pt` for a local continuation, or `--visualizer kit` for the Kit
viewer. Neither visualizer requires an external callback or particle-setting override.

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
MPM's `pic27` contacts and a 24-sweep Jacobi budget as its baseline. Run each case in a fresh process:

```bash
uv run python scripts/benchmarks/benchmark_franka_pour_fidelity.py \
  --mode mpm --num_envs 64 --checkpoint /path/to/model.pt \
  --output logs/mpm_fidelity/mpm-64.json

uv run python scripts/benchmarks/benchmark_franka_pour_fidelity.py \
  --mode particles2 --num_envs 64 --checkpoint /path/to/model.pt \
  --output logs/mpm_fidelity/particles2-64.json
```

The standalone benchmark accepts a checkpoint file path. To use the official policy, pass the
cached path printed by `isaaclab play --checkpoint pretrained`, or download the linked file above.
The benchmark does not resolve the literal value `pretrained`.

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

For a handoff inside one run, choose an environment count that also fits full MPM. Lazy particle
allocation does not guarantee that the later MPM grid will fit. A larger particle-only training
batch can instead save the policy and optimizer, then resume with fewer MPM environments; each
environment retains the same 84 actor inputs, 86 critic inputs, and eight actions.

The reports include terminal success and spill counts, finite-state and movement checks, and the
final mean particle fractions in each cup and the spill region. These final fractions describe
the current episodes after automatic resets; a freshly reset full source cup does not establish
that earlier episodes avoided spilling. Source fingerprints and checkpoint hashes identify
archived deployments even when their Git metadata is absent.

### Official pretrained checkpoint: PPO continuation

The October 16, 2026 measurements below resume the official iteration-7450 weights under this
branch's current task settings. They retain 468 particles per environment, 84 actor inputs,
86 critic inputs, eight actions, and the saved optimizer. The asset supplies weights and optimizer
state; it does not bundle the original run's complete material configuration.

Each fresh process used seed 42, one CPU thread, 16 warmup policy steps, and 96 timed inference
steps before collecting five PPO iterations of 96 transitions per environment. Four minibatches
and five epochs give 20 Adam updates per iteration. The PPO rate includes collection, returns,
optimization, logging, and checkpoint saving. The preceding inference window is timed separately.
The reset sampler remained adaptive. Warp 1.18.0, PyTorch 2.11.0+cu130, and the Newton/Isaac Lab
Python runtime sources matched across the two hosts.

A matched 512-environment comparison on one NVIDIA L40 gave:

| Mode | Inference transitions/s | PPO transitions/s | GPU peak, MiB | Pre-PPO terminal success |
| --- | ---: | ---: | ---: | ---: |
| MPM, `pic27`, Jacobi budget 24 | 1,797 | 1,753 | 7,517 | 95.9% |
| Particles, 2 iterations | 5,650 | 5,272 | 2,665 | 74.4% |

Particle training was 3.01x faster at the same environment count. The inference window recorded
263 spill terminations with particles and four with MPM. Its success rates describe the frozen
input policy before PPO and divide successful episodes by completed episodes within that window.
Evaluate the continued checkpoints separately to assess learned quality.
Both continuations saved changed actor weights, finite actor/critic/optimizer tensors, five finite
loss records per tag, and 100 additional Adam updates in every optimizer state.

The same particle mode and PPO protocol scaled as follows. Peaks are sampled whole-GPU usage at
one-second intervals, including the scene, rollout storage, policy, and optimizer allocations.

| GPU | Environments | PPO transitions/s | GPU peak, MiB |
| --- | ---: | ---: | ---: |
| L40, 48 GB | 4,096 | 12,093 | 10,565 |
| L40, 48 GB | 8,192 | 10,501 | 19,339 |
| L40, 48 GB | 21,504 | 6,825 | 47,915 |
| RTX PRO 6000 Blackwell, 96 GB | 4,096 | 20,616 | 11,108 |
| RTX PRO 6000 Blackwell, 96 GB | 8,192 | 19,155 | 19,884 |
| RTX PRO 6000 Blackwell, 96 GB | 16,384 | 16,653 | 37,402 |
| RTX PRO 6000 Blackwell, 96 GB | 40,960 | 11,646 | 90,841 |
| RTX PRO 6000 Blackwell, 96 GB | 45,056 | 10,858 | 96,466 |

Among these completed primary cases, 4,096 environments gave the highest throughput on both
hosts. The L40 also completed five PPO iterations with 21,504 environments, or 10,063,872 particles.
Its next tested full-protocol batch, 22,528 environments, exhausted CUDA memory while collecting
after the first PPO iteration (20 Adam updates). A shorter 16-step inference/one-PPO-iteration probe
at 22,528 had passed, while a 24,576-environment probe failed during optimization of its first PPO
iteration. The full-protocol
L40 bracket therefore has 1,024-environment resolution; a short capacity probe alone does not confirm
later training capacity. The host-memory guard retained an 8 GiB reserve and did not trigger in
these GPU-memory failures.

The RTX PRO 6000 completed five PPO iterations with 45,056 environments, or 21,086,208 particles.
The next tested count, 49,152, completed its first PPO rollout and exhausted CUDA memory in
`loss.backward()` during optimization, requesting another 288 MiB with 254.75 MiB free. No complete
PPO iteration was logged, and its internal minibatch index was not instrumented. Its sampled peak
was 97,064 MiB. This is a tested 4,096-environment bracket: the successful lower point uses the
full protocol, while the failing upper point uses the shorter one-iteration capacity protocol.
No host-memory guard, timeout, or foreign GPU computation caused that failure. Earlier interrupted
attempts remain excluded from capacity and throughput conclusions.

The largest completed batch is 47.3% slower than the 4,096-environment batch on the RTX system.
These counts bound the measured scene and fixed PPO configuration; they do not establish an
exact particle-solver or hardware maximum.

Only completed, isolated five-iteration cases enter the throughput table. Startup and cold
compilation are excluded, and the hosts have different CPUs as well as GPUs. Changing the number
of environments also changes PPO minibatch size and the reset populations. Measure time to the
desired task performance separately from these collection and optimization rates.

For example, run the same training protocol using a cached official checkpoint:

```bash
uv run python scripts/benchmarks/benchmark_franka_pour_fidelity.py \
  --mode particles2 --num_envs 8192 \
  --checkpoint /path/to/cached-official-pour.pt \
  --warmup_steps 16 --steps 96 --train_iterations 5 \
  --seed 42 --cpu_threads 1 \
  --output logs/mpm_fidelity/particles2-8192-ppo5.json
```

### Historical cached-checkpoint measurements

On October 16, 2026, a single NVIDIA L40 with Warp 1.18.0 and PyTorch 2.11.0 ran the saved
iteration-6997 policy with seed 42 and 468 particles per environment. Each rollout used 16
warmup policy steps followed by 96 timed steps, or 3.2 simulated seconds at the task's 30 Hz
policy rate. Timing includes the robot, coupling, inference, observations, rewards, and resets.
Both modes used the same corrected convex-mesh adapter and control/substep settings. This was
an earlier locally cached checkpoint; the official published default is iteration 7450 above.

| Environments | MPM transitions/s | Particles, 2 iterations | Speedup | Terminal success, MPM / particles |
| --- | ---: | ---: | ---: | ---: |
| 16 | 156 | 431 | 2.77x | 77.5% / 54.5% |
| 64 | 502 | 1,229 | 2.45x | 68.9% / 38.0% |
| 256 | 1,309 | 3,477 | 2.66x | 59.1% / 43.4% |
| 512 | 1,813 | 5,873 | 3.24x | 53.1% / 44.9% |

Terminal success divides successful episodes by completed episodes within each short window.
The particle mode produced more spill terminations, so the speedup does not establish physical
equivalence or equal policy quality. These are single-seed measurements. They also do not establish
the maximum supported environment count. At 512 environments, sampled whole-GPU peak memory
was 7,269 MiB for MPM and 2,417 MiB for particles, including shared scene and policy allocations.

Five actual PPO continuation iterations from the same checkpoint gave the following rates,
including optimization, logging, and final checkpoint saving. Each iteration collected 96
transitions per environment; the policy dimensions and optimizer were retained.

| Environments | MPM training transitions/s | Particle training transitions/s | Speedup |
| --- | ---: | ---: | ---: |
| 128 | 820 | 1,983 | 2.42x |
| 512 | 1,794 | 5,134 | 2.86x |

The rollout matrix used Newton `c4ff41bf`; PPO used `d183ffaa`. The latter adds a fix for
positive-cohesion pair friction; all these runs used zero particle cohesion. Each comparison
used identical source fingerprints within its group. Five PPO iterations demonstrate executable
continuation and throughput, not training convergence or recovered transfer success.

The separate S2 experiment kept the same 64 worlds and used Gauss-Seidel with the following
upper iteration budgets. Its purpose was to test retaining contact impulses across grid rebuilds.

| S2 contact cache | Iteration budget | Transitions/s | Terminal success |
| --- | ---: | ---: | ---: |
| Disabled | 20 | 355 | 65.0% |
| Enabled | 5 | 410 | 62.1% |
| Disabled | 5 | 401 | 31.3% |

The cache restored much of the low-budget task behavior in this short run. The cached five-iteration
budget was about 15% faster than the 20-iteration cold budget. This does not make S2 faster than the
task's default `pic27`/Jacobi configuration, or establish equivalent material behavior.

Separate 48-step capacity probes, after eight warmup policy steps, completed at the following
sampled whole-GPU peaks:

| Environments | Full MPM peak, MiB | Particle peak, MiB |
| --- | ---: | ---: |
| 1,024 | 12,771 | 3,275 |
| 2,048 | 24,277 | 5,021 |
| 4,096 | 47,089 | 8,507 |
| 8,192 | Allocation failed | 15,375 |

At 8,192 worlds, MPM exhausted GPU memory during initial sparse-volume allocation and exited
after reaching 48,467 MiB. Particle mode completed with 3,833,856 particles. These are tested
allocation and finite-motion outcomes, not exact capacity limits or training-quality guarantees.
MPM's environment-workspace guard activated frequently at the larger counts; its root cause was
not established. Reset-row coverage and grid phase also change across environment counts.
Capacity timing is excluded because some probes overlapped video or functional work on the
other GPU. All primary rollout and PPO timing comparisons above ran without that overlap.

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

The 16-world continuation experiment also exercised these APIs inside PPO. A 50% success gate
did not trigger within five iterations. A separate, explicitly forced zero-threshold mechanism
trial switched from particles to MPM at policy step 265 during training steps 20 through 500.
Two complete PPO iterations used particles before the handoff, and two complete iterations used MPM
afterward; the intervening rollout included both modes. The handoff changed 15 mm voxels to
12 mm and reduced the radius by 2%, retaining all 7,488 particles and their 2.204999 kg total mass.
The radius change increased density by approximately 6.25% at fixed mass.

A second experiment started in full cached-S2 MPM and rebuilt its already-active grid from
15 mm to 12 mm at policy step 355 during PPO. It retained particle radius, count, and total mass.
Both final checkpoints contained changed actor weights and 100 additional Adam updates from
the saved iteration-6997 checkpoint, with no optimizer restart. These short mechanism trials
demonstrate executable handoffs and grid rebuilding; they do not establish a learned curriculum.

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
adaptive competence frontier. The reset sampler has no runtime stages or region probabilities.

The sampler reports its effective pool size, concentration, and cyclic-replay fraction. Offline
reset regions remain in the artifact as generation provenance, but runtime does not maintain a
second set of per-region counters. Evaluate frozen policies separately rather than inferring
performance from training-time samples.

Each worker applies completed outcomes immediately to its own sampler. In distributed training,
the curriculum therefore adapts independently on each rank while PPO still learns from all rank
rollouts. Actor and critic empirical observation normalization is disabled because task
observations are physically scaled at their source.
