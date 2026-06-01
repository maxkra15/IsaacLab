# Waterhose Robot Demo

Last verified: 2026-06-01.

The waterhose tasks live in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose/
```

The layout follows the standard manager-based manipulation tasks such as
`reach`:

```text
waterhose/
  __init__.py
  waterhose_env_cfg.py
  mdp/
  config/
    rby1df/
      __init__.py
      coupled_env_cfg.py
      teleop_env_cfg.py
      agents/
```

The package exposes two public task variants:

| Task ID | Purpose |
| --- | --- |
| `Isaac-Waterhose-Coupled-v0` | Client-facing RBY1DF waterhose demo using Newton proxy coupling and absolute Newton IK actions for scripted demo and XR. |
| `Isaac-Waterhose-Coupled-Teleop-v0` | Same coupled scene with relative Newton IK actions and `env_cfg.teleop_devices` for native keyboard/SpaceMouse teleop. |

The coupled task is the primary demo path. It uses normal IsaacLab scene configuration:

- `ArticulationCfg` for the RBY1DF robot USD.
- `RigidObjectCfg` for the plugs and kinematic cable anchors.
- `CableObjectCfg` for the two USD cable curves.
- `AssetBaseCfg` for the fridge/static visual scene.
- `CoupledNewtonCfg` plus `isaaclab_newton.physics.CoupledSolverCfg`.

`CoupledSolverCfg.class_type` resolves to `isaaclab_newton.physics.coupled_manager:NewtonCoupledManager`. That manager partitions one Newton model into an MJWarp source view for the robot and a VBD destination view for the cable/plug bodies, then builds Newton's `SolverCoupledProxy` from `newton.solvers.experimental.coupled`.

## Running

Scripted demo with Kit:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --vis kit
```

Scripted demo with the Newton viewer:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --vis newton
```

Headless profile:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --vis none \
  --max_steps 200 \
  --profile
```

SpaceMouse teleop:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --teleop_device spacemouse \
  --visualizer kit
```

Keyboard teleop:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --teleop_device keyboard \
  --visualizer kit
```

XR / IsaacTeleop:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --visualizer kit \
  --xr
```

Apple Vision Pro CloudXR profile:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --visualizer kit \
  --xr \
  --cloudxr_env avp
```

The scripted/XR task uses an absolute 8D Newton IK action space configured through `env_cfg.isaac_teleop`.
The desktop teleop task uses a relative 7D action space configured through `env_cfg.teleop_devices`, matching
IsaacLab's native keyboard and SpaceMouse devices.

The SpaceMouse entry uses a waterhose-specific mapping:

- Cap translation moves the gripper in XYZ with the same signs as the original stable waterhose demo.
- Cap twist rotates the gripper around the insertion/yaw axis.
- Roll and pitch are suppressed to keep the plug aligned.
- Translation suppresses accidental twist noise; use a pure twist motion for yaw.

Multi-env smoke test:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 4 \
  --vis none \
  --max_steps 100 \
  --profile
```

## Assets

By default the task uses packaged assets next to the waterhose package:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose/assets/
```

Required files:

```text
fridge/fridge.usda
fridge/cable/cable001.usda
fridge/cable/cable002.usda
fridge/cable/plug.usda
rby1df/rby1df.usda
```

To use another asset directory:

```bash
WATERHOSE_ASSETS_DIR=/path/to/waterhose/assets
```
