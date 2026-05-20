# RBY1 Waterhose Task Profiling and Setup Review

Date: 2026-05-19

Branch inspected: `feat/newton-implicit-mpm`

Task: `Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0`

## Executive Summary

The current RBY1 waterhose task is a working manager-based Isaac Lab task built around a Newton proxy-coupled solver stack: MJWarp handles the robot rigid bodies, VBD handles the two deformable hose/cable curves, and a lagged proxy coupling connects the RBY1 gripper bodies to the hose. The implementation follows several Isaac Lab patterns correctly, including a manager-based environment config, an `ActionTerm` for task-space IK, observation/reward/termination managers, and Kit-specific visual sync callbacks.

The main limitation is compute cost. In no-Kit scripted teleop-style stepping, one environment takes about 24.0 ms per environment step, while the configured simulation time step is 10 ms. This is about 41.7 Hz wall-clock for a 100 Hz simulated control step, so the task is not real-time even before Kit rendering. With Kit visualization active, `env.step()` takes about 62.8 ms because Isaac Lab renders every step when Kit is active. This gives about 15.9 Hz active teleop update rate. A separate extra `env.sim.render()` call costs another 15.4 ms, but the active teleop script does not call this extra render after `env.step()`.

The biggest runtime cost is the solver configuration and scene size: two 100-segment cables, 202 to 205 VBD bodies, 40 MJWarp robot bodies, 10 Newton substeps, 10 VBD iterations, 20 MJWarp iterations, 10 MJWarp line-search iterations, and five VBD collision substeps. The biggest startup reliability risk is the Newton USD importer for the static scene. One profiling attempt remained inside `newton/_src/utils/import_usd.py:parse_usd` for over 60 seconds while adding the static scene USD from `waterhose_core.py:1236`.

## Profile Methodology

I profiled from the normal Isaac Lab task entry path using `parse_env_cfg()` and `gym.make()`, with deterministic teleop-like 7D task-space actions. This exercises the same `NewtonTaskSpaceIKAction` path used by teleop, but it does not include physical SpaceMouse hardware polling. SpaceMouse polling itself should be negligible compared with the measured simulation and render costs.

The no-Kit run used `--viz none`. The Kit run used `--visualizer kit` with `DISPLAY=:1` and `CUDA_VISIBLE_DEVICES=0`. GPU/CPU synchronization was forced around timings with `torch.cuda.synchronize()` and `warp.synchronize_device()` to avoid undercounting async GPU work.

## Measured Results

### No-Kit scripted teleop-style stepping

Configuration:

- `sim.dt = 0.01`
- `decimation = 1`
- `render_interval = 1`
- `sim_substeps = 10`
- `rigid_substeps = 1`
- `proxy_iterations = 1`
- `vbd_iterations = 10`
- `vbd_collide_substeps = 5`
- `cable_num_segments = 100`
- `mujoco_iterations = 20`
- `mujoco_ls_iterations = 10`
- `cuda_graph = True`

Scene size:

- Builder bodies: 245
- Builder joints: 243
- Builder shapes: 524
- Cable curves: 2
- Cable bodies: 100 + 100
- Cable heads: 2
- Proxy bodies: 8
- Proxy shapes: 12
- VBD bodies: 205
- MJWarp robot bodies: 40
- Static scene shapes: 251

Timing:

- Startup plus reset: 5.431 s
- `env.step()` mean: 24.008 ms
- p50: 23.952 ms
- p90: 24.193 ms
- Effective rate: 41.65 Hz

Interpretation: the physics/control loop is already slower than the configured 100 Hz simulated step. This is the baseline compute bottleneck independent of Kit rendering.

### Kit scripted teleop-style stepping

Scene size differs because the Kit path uses procedural/static Kit visuals instead of importing all static scene collision/visual shapes into Newton:

- Builder bodies: 242
- Builder joints: 240
- Builder shapes: 275
- Cable curves: 2
- Cable bodies: 100 + 100
- Cable heads: 2
- Proxy bodies: 8
- Proxy shapes: 12
- VBD bodies: 202
- MJWarp robot bodies: 40
- Static scene shapes: 2

Timing:

- Startup plus reset: 6.205 s
- Active `env.step()` mean: 62.807 ms
- p50: 62.633 ms
- p90: 63.798 ms
- Effective active teleop rate: 15.92 Hz
- Extra standalone `env.sim.render()` mean: 15.429 ms
- Step plus extra render mean: 78.236 ms

Interpretation: in active Kit teleop, `env.step()` is the relevant loop cost because Isaac Lab renders inside `ManagerBasedRLEnv.step()` whenever Kit rendering is active and `render_interval` is reached. The teleop script calls `env.step(actions)` for active commands, not a second explicit render.

## Major Computing Limitations

1. The solver is expensive relative to the requested 100 Hz step.

The task uses 10 Newton substeps, 10 VBD iterations, 20 MJWarp iterations, and 10 MJWarp line-search iterations every 10 ms environment step. This produces stable deformation/contact behavior, but the measured no-Kit loop is about 24 ms per step, so it cannot run in real time at the current fidelity.

2. The cable resolution is high for interactive teleop.

Each hose curve uses 100 cable bodies, so the two-cable setup drives roughly 200 deformable bodies plus two cable-end rigid bodies. This directly increases VBD solve cost, contact work, and Kit curve point updates.

3. Kit rendering roughly doubles to triples the active loop cost.

With Kit active, `env.step()` rises from about 24 ms to about 63 ms. This includes the GUI/viewport app update, Fabric/Newton transform sync, and the dynamic cable BasisCurves update. At the current settings, Kit teleop should be expected to feel closer to 16 Hz than 60 Hz.

4. USD importer startup can be inconsistent.

One no-Kit ablation run remained for over 60 seconds in Newton's USD importer:

`newton/_src/utils/import_usd.py:parse_usd -> builder.add_usd -> waterhose_core.py:1236 _add_static_table_and_socket`

This is a real startup risk for repeated profiling, demos, and automated tests. The Kit path avoids this specific heavy static-scene import by using procedural static table/socket setup when Kit is requested.

5. The current machine/profile mode is not ideal for clean Kit timing.

Kit reported that `CUDA_VISIBLE_DEVICES` is set and warned that CUDA and Omniverse device enumeration differ. It also reported CPU powersave mode, PCIe link width 8 instead of 16 for the active GPU, and IOMMU enabled. These are not task-code bugs, but they can depress measured performance and should be controlled for official benchmark numbers.

## Task Setup Review

### What is set up well

- The task is registered as a manager-based Isaac Lab environment and uses standard action, observation, reward, termination, and scene config classes. See `waterhose_env_cfg.py:34-121`.
- The Newton solver parameters are exposed as config fields rather than hidden constants. See `waterhose_env_cfg.py:134-188`.
- The environment builds the Newton scene before `ManagerBasedRLEnv` initialization, then injects the runtime-built solver config into `cfg.sim.physics`. See `waterhose_env.py:31-45`.
- Teleop uses the normal manager action path. The active loop repeats the 7D device command across environments and calls `env.step(actions)`. See `teleop_se3_agent.py:540-570`.
- The task-space action is implemented as a custom `ActionTerm`, with command scaling, analytic IK, gripper target handling, and Newton control-buffer writes isolated in `actions.py:23-121`.
- The Kit cable visualization now follows the right general approach: spawn Kit-only BasisCurves and update their points from Newton body poses before render. See `waterhose_core.py:2051-2097` and `waterhose_core.py:2204-2238`.
- The launch helper explicitly handles the known Newton/Kit import-order issue. See `launch.py:147-196`.

### Risks and deviations from best practice

- Reset handling is incomplete for training-style use. `_reset_idx()` calls the parent reset and reapplies cable transforms, but the comment states that full partial Newton state reset is not implemented until coupled solver per-world reset hooks exist. See `waterhose_env.py:159-164`. This is acceptable for demo collection, but it is not best practice for scalable RL training or robust multi-env evaluation.
- The task is effectively single-environment focused. The config has `replicate_physics=False`, labels sometimes fall back for `num_envs == 1`, and Kit cable visuals are rooted under Env_0 for multi-env. This is acceptable for teleop/demo, but not yet a scalable vectorized training setup.
- The no-Kit path imports the static USD scene through Newton. This can produce many shapes and the observed importer stall. For repeatable startup, the procedural static table/socket approach used for Kit should also be considered for no-Kit unless full USD fidelity is required.
- Kit BasisCurves are updated through USD attribute authoring each render. This is the most practical path for visibility, but it still emits Fabric warnings about optional curve attributes such as `connections` and `timeVaryingAttributes`. The code intentionally avoids mirroring topology into USDRT every frame, which is good, but the remaining warnings should be cleaned up if this becomes a supported demo.
- The action path copies Newton/Warp state to NumPy and Torch in several places, including `state.body_q.numpy()`, `joint_q.numpy()`, and `detach().cpu().numpy()` in `actions.py:93-121`. For one env this is fine; for scaling, these host round trips will become a bottleneck and should be moved into GPU-native buffers or batched kernels.

## Recommendations

1. Define two official presets:

- `demo_quality`: current settings, intended for stable visual demos and cable behavior.
- `teleop_fast`: lower solver/render fidelity for interactive Kit control, likely by reducing `sim_substeps`, `vbd_iterations`, `mujoco_iterations`, and possibly `cable_num_segments`.

2. Avoid importing the full static scene USD into Newton for no-Kit runs unless it is needed for contact fidelity.

The Kit path already uses a procedural static table/socket. A similar no-Kit procedural collision representation would remove the importer startup failure mode and reduce shape count.

3. Reduce Kit render cadence for operator control if physics remains the priority.

Since the active Kit loop renders inside `env.step()` at `render_interval = 1`, setting `render_interval = 2` or `3` for teleop would reduce viewport cost. This trades visual smoothness for control throughput.

4. Add a small benchmark script for this task.

A task-local benchmark should measure startup, `env.step()` no-Kit, `env.step()` with Kit, and optional render-only cost. It should print solver settings and scene counts. This would make future changes to cable visualization and solver parameters measurable instead of subjective.

5. Plan a proper reset path before treating this as RL-ready.

For demonstration collection, the current reset behavior is probably sufficient. For policy training, evaluation, or vectorized rollouts, partial Newton state reset for robot, cable bodies, cable heads, and coupled solver state should be implemented and tested.

6. Clean up runtime warnings before supervisor demos.

The warnings about `CUDA_VISIBLE_DEVICES`, CPU powersave mode, PCIe link width, IOMMU, and Fabric curve attributes should be separated into environment issues versus task-code issues. The environment warnings may affect benchmark credibility; the Fabric warnings may make users suspect the cable visualization even when it is visually functioning.

## Bottom Line

The task is architecturally reasonable for a demonstration-grade Newton waterhose manipulation environment and is now using the correct high-level Kit cable visualization strategy. However, it is not currently configured for real-time Kit teleoperation or scalable RL. The major bottleneck is the high-fidelity coupled Newton/VBD/MJWarp solve, followed by Kit rendering/Fabric sync. The main reliability issue is the no-Kit Newton USD static-scene importer path, which can stall during startup. The next engineering step should be a fast teleop preset plus a procedural no-Kit static scene path, followed by a small repeatable benchmark script.
