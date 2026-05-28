# Waterhose Robot Demo

Last verified: 2026-05-28.

This task is a client-facing Isaac Lab demo for inserting a flexible water hose into a socket with an RBY1DF robot. The stable default path is a one-way Newton setup wrapped as a manager-style Isaac Lab task. The experimental ADMM coupled-solver path remains available for solver development, but it is not the recommended demo path.

## Repositories

Use the Isaac Lab fork branch that contains the task:

```bash
git clone git@github.com:maxkra15/IsaacLab.git IsaacLab-waterhose-demo
cd IsaacLab-waterhose-demo
git checkout waterhose-demo
```

The Newton dependency is PR 2848. This branch was verified with:

```text
branch: pr-2848-coupled-solver-framework-latest
tracks: origin/pr-2848-head
commit: c2f21df3acc0f06d207812810b2e27ca7c4da08c
```

For a separate Newton checkout:

```bash
git clone https://github.com/newton-physics/newton.git /path/to/newton
cd /path/to/newton
git fetch origin pull/2848/head:pr-2848-head
git checkout pr-2848-head
```

The Isaac Lab packages in this fork pin `newton[sim]` to the verified PR commit above. The demo runner also accepts `--newton_root /path/to/newton` for workflows that use a separate Newton checkout.

## Assets

The demo assets are distributed as `waterhose_demo_assets.tar.gz`. The archive contains a top-level `WaterhoseDemo/` directory with the RBY1DF URDF, robot meshes, fridge scene USD, cable curve USD, and plug meshes. Extract it into the Isaac Lab asset data directory:

```bash
cd /path/to/IsaacLab-waterhose-demo
tar -xzf /path/to/waterhose_demo_assets.tar.gz -C source/isaaclab_assets/data
```

After extraction these files should exist:

```text
source/isaaclab_assets/data/WaterhoseDemo/RBY1DF/urdf/robot_edited.urdf
source/isaaclab_assets/data/WaterhoseDemo/Waterhose/Cable008/Cable008_Body.usda
source/isaaclab_assets/data/WaterhoseDemo/Waterhose/Cable008/curve/cable_SRA_curve03.usda
```

## Install

Install this fork with the normal Isaac Lab workflow:

```bash
cd /path/to/IsaacLab-waterhose-demo
./isaaclab.sh --install
```

If a machine uses a separate Newton checkout during development, pass it explicitly when running:

```bash
--newton_root /path/to/newton
```

If the assets are outside the default location, pass the extracted `WaterhoseDemo` directory:

```bash
--asset_root /path/to/WaterhoseDemo
```

The task package sets `PXR_WORK_THREAD_LIMIT=1` by default before the USD-backed cable and fridge assets are imported. This avoids a USD/Warp threading failure seen during standalone Newton setup.

## Registered Tasks

The task registrations live under:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose_robot_demo/config/rby1df/__init__.py
```

Available task IDs:

```text
Isaac-Waterhose-Robot-Demo-v0
Isaac-Waterhose-Robot-Demo-Mimic-v0
Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0
```

`Isaac-Waterhose-Robot-Demo-v0` is the stable default. `Isaac-Waterhose-Robot-Demo-Mimic-v0` exposes the same simulation through Isaac Lab Mimic APIs. `Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0` is retained for coupled-solver experiments only.

## Stable One-Way Architecture

The stable task is implemented as a normal Isaac Lab manager-style environment:

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

`WaterhoseRobotDemoEnvCfg` uses `SimulationCfg(physics=NewtonCfg(...))` with `WaterhoseNewtonSolverCfg`. `WaterhoseRobotDemoEnv` synchronizes the Isaac Lab scene settings into the solver config before the manager is constructed. The action, observation, termination, event, recorder, and teleop surfaces are Isaac Lab managers.

Each simulated environment owns a split Newton runtime:

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

There is no force path from VBD back into MuJoCo. This is deliberate. The original Newton success demo used carefully filtered normal-force feedback, while generic coupled-manager feedback introduced tangential/friction forces that destabilized the robot. The default demo avoids that risk: the robot is authoritative, and the cable/plug respond to the robot through proxy gripper bodies.

The proxy bodies duplicate the four gripper finger bodies and their attached shapes. They are high-friction VBD collision bodies and are driven kinematically from the MuJoCo finger states. The VBD solver then handles cable, plug, fridge, and gripper-proxy contacts.

## Multi-Env Behavior

`--num_envs` is supported for the stable one-way task. The current implementation creates N independent split Newton runtimes and arranges them on an Isaac Lab-style grid. It also builds a combined visualization model so Newton Viewer and Kit can show multiple environments.

This is not a true batched Newton model. The drawbacks are:

- setup cost and memory scale roughly with the number of independent runtimes;
- stepping is orchestrated by the task manager rather than by one vectorized Newton model;
- Kit display is manually authored because the real physics is split between MuJoCo and VBD;
- cable, plug, and socket state are not normal Isaac Lab `InteractiveScene` rigid objects;
- Mimic object APIs must be provided by task-specific overrides.

This is acceptable for demo, teleop, and initial data tooling. It is not the final high-throughput RL or imitation-learning architecture.

## Kit And Newton Visualization

The demo uses the standard Isaac Lab launcher path. The runner supports a local `--vis` alias:

```text
--vis none
--vis newton
--vis kit
--vis kit,newton
```

When Kit is selected, the dynamic robot/cable/plug display comes from the combined Newton visualization model. The static fridge scene is authored once per environment under a Kit display root because the real VBD static scene is internal to the Newton runtime.

On machines where the display is on `:1`, prefix visual runs with:

```bash
DISPLAY=:1
```

## Run Commands

Scripted demo without visualization:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis none --max_steps 2000
```

Scripted demo in Newton Viewer:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis newton --max_steps 2000
```

Scripted demo in Kit:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit --max_steps 2000
```

Scripted demo with both viewers:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit,newton --max_steps 2000
```

Two environments without visualization:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis none --num_envs 2 --max_steps 1000 --profile
```

Two environments in Newton Viewer:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis newton --num_envs 2 --max_steps 1000 --profile
```

Two environments in Kit:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit --num_envs 2 --max_steps 1000 --profile
```

Teleop with SpaceMouse in Kit:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --mode teleop --teleop_device spacemouse --vis kit
```

Teleop with SpaceMouse in Newton Viewer:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --mode teleop --teleop_device spacemouse --vis newton
```

Teleop uses the waterhose-specific SpaceMouse mapping from the old simple demo: XYZ translation plus gripper yaw/spin. Roll and pitch are suppressed, and translation/yaw can be locked apart with `--spacemouse_simple_yaw_translation_lock`. The left/right gripper action comes through the standard Isaac Lab `Se3SpaceMouse` gripper term.

Useful teleop tuning flags:

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

Mimic task smoke run:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --task Isaac-Waterhose-Robot-Demo-Mimic-v0 --vis none --num_envs 2 --max_steps 1 --profile
```

Experimental ADMM task:

```bash
DISPLAY=:1 ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --task Isaac-Waterhose-Robot-Demo-Admm-Experimental-v0 --vis newton --num_envs 1 --max_steps 1000
```

The ADMM task is single-env only in this runner.

## ADMM Coupling Architecture

The ADMM path is implemented in:

```text
admm_builder.py
admm_manager.py
admm_env_cfg.py
admm_mdp.py
```

It builds a task-local combined Newton model and uses the Newton coupled-solver framework through `NewtonCoupledManager`. The model contains the MuJoCo robot, VBD cable/fridge/plug scene, and contact/coupling definitions between the robot gripper side and the VBD cable side. The config uses `WaterhoseAdmmSolverCfg`, `CoupledSolverEntryCfg`, `AdmmCouplingCfg`, and a VBD sub-solver config.

This approach is architecturally attractive because it is closer to a single coupled Newton simulation. In the long term, that is the likely direction for a proper coupled, batched, Isaac Lab-native task.

It was not pursued as the primary demo path because the observed behavior was less stable than the one-way runtime:

- contact transfer could push forces back into the robot and disturb the scripted motion;
- the cable became jumpy under stiff contact/coupling settings;
- friction and tangential forces were much harder to control than the reference one-way proxy behavior;
- ADMM iterations and contact sets made the simulation slower;
- matching the original success demo required careful filtering of contact pairs and solver parameters;
- the path was not validated for multi-env or client demo use.

For now, the ADMM path is retained as a research and solver-development target. It should not be used as the default waterhose demo.

## Mimic Integration

The Mimic variant is:

```text
Isaac-Waterhose-Robot-Demo-Mimic-v0
```

It uses `WaterhoseRobotDemoMimicEnv`, which adapts the stable one-way task to `ManagerBasedRLMimicEnv`. The object poses are supplied directly from the Newton manager:

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

This is task-specific because the cable, plug, and socket are not regular Isaac Lab scene assets. The overrides provide the object pose and subtask signal surface that Mimic expects, while keeping the stable one-way Newton runtime.

## Current Caveats

- The stable task is one-way. The cable does not apply forces back to the robot.
- Multi-env is N independent Newton runtimes, not one true vectorized Newton model.
- Kit display is manually authored for this task because the real simulation is split between MuJoCo and VBD.
- The Mimic task has task-local object pose overrides rather than default scene object APIs.
- The ADMM task is experimental and single-env.
- XR/Apple Vision Pro support should be layered through the standard Isaac Lab teleop/XR stack, but it has not been validated in this task yet.
