# Waterhose Robot Demo

Last verified: 2026-05-29.

This task is a client-facing Isaac Lab demo for inserting a flexible water hose into a socket with an RBY1DF robot.

The stable default is:

- a manager-style Isaac Lab task;
- self-contained under `waterhose_robot_demo`;
- one-way Newton coupling between a MuJoCo robot runtime and a VBD hose runtime;
- standard Isaac Lab launch, visualization, teleop, and Mimic surfaces.

The coupled-manager tasks remain available for solver development, but they are not the default demo path.

## Quick Setup

For a fresh machine, put `waterhose.sh` and `waterhose_demo_assets.tar.gz` in a clean folder, then run:

```bash
chmod +x ./waterhose.sh && ./waterhose.sh setup --accept-eula
```

The script creates:

```text
waterhose-demo/
  IsaacLab-waterhose/    Isaac Lab fork checkout, task code, venv, assets
  IsaacSim/              source-built Isaac Sim checkout and build
```

The Newton version is controlled by the Isaac Lab package metadata, not by `waterhose.sh`. The relevant pins are in:

```text
source/isaaclab_newton/setup.py
source/isaaclab_physx/setup.py
source/isaaclab_visualizers/setup.py
```

At the time of this document, those packages install:

```text
newton[sim] @ git+https://github.com/newton-physics/newton.git@refs/pull/2848/head
```

For an existing workspace after pulling dependency changes:

```bash
cd waterhose-demo/IsaacLab-waterhose
git pull --ff-only
source .venv/bin/activate
./isaaclab.sh -i all
```

## Assets

The demo assets are distributed as `waterhose_demo_assets.tar.gz`. The archive contains a top-level `WaterhoseDemo/` directory with the RBY1DF URDF, robot meshes, fridge scene USD, cable curve USD, and plug meshes.

When using `waterhose.sh setup`, the archive is unpacked automatically. For manual setup, extract it into the Isaac Lab asset data directory:

```bash
tar -xzf /path/to/waterhose_demo_assets.tar.gz -C source/isaaclab_assets/data
```

After extraction these files should exist:

```text
source/isaaclab_assets/data/WaterhoseDemo/RBY1DF/urdf/robot_edited.urdf
source/isaaclab_assets/data/WaterhoseDemo/Waterhose/Cable008/Cable008_Body.usda
source/isaaclab_assets/data/WaterhoseDemo/Waterhose/Cable008/curve/cable_SRA_curve03.usda
```

## Registered Tasks

Task registrations live in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose_robot_demo/config/rby1df/__init__.py
```

Current task IDs:

| Task ID | Purpose | Recommended use |
| --- | --- | --- |
| `Isaac-Waterhose-Robot-Demo-v0` | Stable default one-way split runtime | Client demos, scripted runs, teleop, data collection |
| `Isaac-Waterhose-Robot-Demo-Mimic-v0` | Stable default runtime with Mimic APIs | Mimic/data-generation experiments |
| `Isaac-Waterhose-Robot-Demo-Coupled-v0` | Coupled-manager one-way proxy coupling | Solver/API development |
| `Isaac-Waterhose-Robot-Demo-OneWay-Coupled-v0` | Explicit alias for the coupled one-way task | Solver/API development |
| `Isaac-Waterhose-Robot-Demo-TwoWay-Coupled-Experimental-v0` | Coupled-manager two-way proxy coupling | Experimental only |
| `Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0` | Coupled-manager ADMM contact coupling | Experimental only |

Choosing a coupling method is done by selecting the task ID. There is no longer a `--backend reference|oneway|manager` runner switch.

## Default Architecture

`Isaac-Waterhose-Robot-Demo-v0` is implemented as a manager-style Isaac Lab environment:

```text
env.py
env_cfg.py
actions.py
mdp.py
manager.py
simulation.py
kit_display.py
teleop.py
recorders.py
```

`WaterhoseRobotDemoEnvCfg` uses:

```text
SimulationCfg(
  physics=NewtonCfg(
    solver_cfg=WaterhoseNewtonSolverCfg(...)
  )
)
```

The runtime is intentionally split per environment:

```text
MuJoCo robot runtime
VBD cable/fridge/plug runtime
VBD proxy gripper bodies
scripted or teleop controller
```

The coupling is one-way:

```text
MuJoCo robot state
  -> copied finger poses and velocities
  -> VBD proxy gripper bodies
  -> VBD cable/plug/fridge contact solve
```

There is no force path from VBD back into MuJoCo. This is deliberate. The original Newton success demo used carefully filtered normal-force feedback, while generic full-force feedback introduced tangential/friction forces that destabilized the robot. The default demo avoids that risk: the robot is authoritative, and the cable/plug respond to the robot through proxy gripper bodies.

## Multi-Env Behavior

`--num-envs` / `--num_envs` is supported for the default task. The current implementation creates N independent split Newton runtimes and arranges them on an Isaac Lab-style grid. It also builds a combined visualization model so Newton Viewer and Kit can show multiple environments.

This is not a true batched Newton model. The drawbacks are:

- setup cost and memory scale roughly with the number of independent runtimes;
- stepping is orchestrated by the waterhose manager rather than by one vectorized Newton model;
- Kit display is manually authored because the real physics is split between MuJoCo and VBD;
- cable, plug, and socket state are not normal Isaac Lab `InteractiveScene` rigid objects;
- Mimic object APIs are task-specific overrides.

This is acceptable for demo, teleop, and initial data tooling. It is not the final high-throughput RL architecture.

## Visualization

The demo uses the standard Isaac Lab launcher path. The waterhose runner supports a local `--vis` alias:

```text
--vis none
--vis newton
--vis kit
--vis kit,newton
```

When Kit is selected, the runner automatically adds the local source-built Isaac Sim extension folders from `_isaac_sim`, so direct runner launches and `waterhose.sh` launches use the same Kit extension resolution.

On machines where the display is on `:1`, prefix visual runs with:

```bash
DISPLAY=:1
```

## Run Commands

Using the standalone helper:

```bash
./waterhose.sh demo --vis kit
./waterhose.sh demo --vis newton
./waterhose.sh demo --vis kit,newton
./waterhose.sh demo --vis kit --headless --profile --max-steps 200
```

Multiple default environments:

```bash
./waterhose.sh demo --vis kit --num-envs 2 --profile --max-steps 1000
./waterhose.sh demo --vis newton --num-envs 2 --profile --max-steps 1000
```

Manual Isaac Lab runner examples:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit --max_steps 2000
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis newton --max_steps 2000
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit,newton --max_steps 2000
```

## Teleop

The task supports two teleop entry points:

- `waterhose.sh demo --teleop`, which uses the waterhose runner's built-in SpaceMouse loop;
- `waterhose.sh teleop`, which uses Isaac Lab's standard `scripts/environments/teleoperation/teleop_se3_agent.py`.

Recommended SpaceMouse teleop:

```bash
./waterhose.sh teleop --teleop-device spacemouse --vis kit
```

Built-in runner teleop:

```bash
./waterhose.sh demo --teleop --vis kit
```

XR / Apple Vision Pro path through Isaac Teleop:

```bash
./waterhose.sh teleop --xr --cloudxr-env avp --vis kit
```

Teleop uses the waterhose-specific SpaceMouse mapping from the old simple demo: XYZ translation plus gripper yaw/spin. Roll and pitch are suppressed. Translation and yaw can be separated with:

```bash
--spacemouse_simple_yaw_translation_lock
```

Useful tuning flags for the built-in runner teleop path:

```bash
--sensitivity 0.75
--spacemouse_pos_sensitivity 0.04
--spacemouse_rot_sensitivity 0.12
--spacemouse_simple_x_sign -1
--spacemouse_simple_y_sign -1
--spacemouse_simple_z_sign 1
--spacemouse_simple_yaw_sign -1
--debug_teleop
```

## Mimic

The Mimic task is:

```text
Isaac-Waterhose-Robot-Demo-Mimic-v0
```

It uses `WaterhoseRobotDemoMimicEnv`, which adapts the stable default task to `ManagerBasedRLMimicEnv`. The object poses are supplied directly from the Newton manager:

```text
hose_plug
hose_tip
socket
```

The robot end-effector name is:

```text
rby1df_right
```

The subtask termination signals are:

```text
approach
grasp
align
insert
```

Mimic smoke run:

```bash
./waterhose.sh mimic-smoke --num-envs 2
```

Manual Mimic smoke run:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Robot-Demo-Mimic-v0 \
  --vis kit --headless --num_envs 2 --max_steps 1 --profile
```

## Coupled-Manager Variants

The coupled-manager path is separate from the default demo. It uses:

```text
coupled_builder.py   combined Newton model: robot + VBD cable/scene + optional gripper proxies
coupled_manager.py   NewtonWaterhoseCoupledManager and solver configs
coupled_mdp.py       observation and termination helpers
coupled_env_cfg.py   shared manager-style coupled env config
one_way_env_cfg.py   coupled-manager one-way proxy coupling
two_way_env_cfg.py   coupled-manager two-way proxy coupling
admm_env_cfg.py      ADMM cross-solver coupling
```

The one-way coupled task:

```bash
./waterhose.sh demo \
  --task Isaac-Waterhose-Robot-Demo-Coupled-v0 \
  --vis newton --num-envs 1 --max-steps 1000
```

The explicit one-way alias:

```bash
./waterhose.sh demo \
  --task Isaac-Waterhose-Robot-Demo-OneWay-Coupled-v0 \
  --vis newton --num-envs 1 --max-steps 1000
```

The experimental two-way task:

```bash
./waterhose.sh demo \
  --task Isaac-Waterhose-Robot-Demo-TwoWay-Coupled-Experimental-v0 \
  --vis newton --num-envs 1 --max-steps 1000
```

The experimental ADMM task:

```bash
./waterhose.sh demo \
  --task Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0 \
  --vis newton --num-envs 1 --max-steps 1000
```

The ADMM path steps the robot and cable as separate solvers and reconciles contact with linearized ADMM each step. It is architecturally interesting, but it is not the recommended demo path because it has been slower and less stable around stiff gripper/cable contact.

## Current Caveats

- The default task is one-way. The cable does not apply forces back to the robot.
- Default multi-env is N independent Newton runtimes, not one true vectorized Newton model.
- Kit display is task-authored because the real simulation is split between MuJoCo and VBD.
- The Mimic task has task-local object pose overrides rather than default scene object APIs.
- The coupled-manager tasks are for solver development and are less validated than the default task.
- The ADMM task is experimental and should be run single-env.
- XR/Apple Vision Pro support is wired through Isaac Teleop but has not had the same validation as scripted and SpaceMouse runs.
