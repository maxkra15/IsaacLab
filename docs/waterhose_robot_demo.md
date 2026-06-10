# Waterhose Robot Demo

Last verified: 2026-06-10.

This customer-facing branch is `max/waterhose-coupled-experimental`. It supersedes the earlier
`waterhose-demo` branch used for the first package handoff.

The waterhose tasks live in:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/
```

The layout follows IsaacLab's contributed-task structure while keeping the
manager-based environment style:

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

The coupled task is the default and primary demo path. The previous local
`Isaac-Waterhose-Kinematic-v0` workaround has been removed so the demo exercises the real two-way
Newton proxy coupling path. It uses normal IsaacLab scene configuration:

- `ArticulationCfg` for the RBY1DF robot USD.
- `RigidObjectCfg` for the plugs and kinematic cable anchors.
- `CableObjectCfg` for the two USD cable curves.
- `AssetBaseCfg` for the fridge/static visual scene.
- `CoupledNewtonCfg` plus `isaaclab_newton.physics.CoupledSolverCfg`.

`CoupledSolverCfg.class_type` resolves to `isaaclab_newton.physics.coupled_manager:NewtonCoupledManager`. That manager partitions one Newton model into an MJWarp source view for the robot and a VBD destination view for the cable/plug bodies, then builds Newton's `SolverCoupledProxy` from `newton.solvers.experimental.coupled`.

## Standalone Setup

Use the bundled wrapper for a fresh machine:

```bash
./waterhose.sh init --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz
```

`init` is an alias for `setup`. It clones this branch by default, builds Isaac Sim, creates `.venv`,
installs the full Isaac Lab workspace with `isaaclab.sh -i all`, installs Newton from upstream PR
2848 commit `31f56815a35d3a57b64f3894d574c4814c3c7c1a`, unpacks the demo assets, and runs the
headless smoke check unless `--skip-smoke` is passed.

The setup does not depend on `/home/maximiliank/Work/newton-coupled` or any other local Newton edits.

## Running

Scripted demo with Kit:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --visualizer kit
```

Scripted demo with the Newton viewer:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --visualizer newton
```

Headless profile:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --visualizer none \
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

Apple Vision Pro XR teleoperation is verified on this branch with the IsaacTeleop/OpenXR path. Use the
wrapper command:

```bash
./waterhose.sh teleop --xr --cloudxr-env avp --vis kit --num-envs 1 --debug-teleop
```

Open the XR panel in Isaac Sim, select `OpenXR` and `System OpenXR Runtime`, then click `Start XR`.
The AVP and workstation must be IP-reachable on the same wireless network or routed VLAN. If the
workstation also has Ethernet connected, make sure the IP entered on the AVP is the workstation's
Wi-Fi IP and that firewall rules allow CloudXR on that interface.

Minimum AVP firewall rules for the native CloudXR profile:

```bash
sudo ufw allow 48010/tcp
sudo ufw allow 47998/udp
sudo ufw allow 48005/udp
sudo ufw allow 48008/udp
sudo ufw allow 48012/udp
sudo ufw allow 47999/udp
sudo ufw allow 48000/udp
sudo ufw allow 48002/udp
```

Meta Quest / Pico via CloudXR.js use the web profile:

```bash
./waterhose.sh teleop --xr --cloudxr-env cloudxrjs --vis kit --num-envs 1 --debug-teleop
```

Open `https://nvidia.github.io/IsaacTeleop/client` in the headset browser, enter the workstation IP,
accept the self-signed certificate at `https://<workstation-ip>:48322/`, then connect.

Minimum CloudXR.js firewall rules:

```bash
sudo ufw allow 49100/tcp
sudo ufw allow 48322/tcp
sudo ufw allow 47998/udp
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
  --visualizer none \
  --max_steps 100 \
  --profile
```

## Assets

By default the task uses packaged assets next to the waterhose package:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/
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

## Customer Verification Notes

These notes address the initial customer feedback from the first RB-Y1 refrigerator water hose handoff.

1. External endpoints: the packaged waterhose assets are local after `waterhose_demo_assets.tar.gz` is
   unpacked, but setup and first-run dependency resolution still need network access unless everything is
   pre-cached. Allow at least these endpoints for this repo's setup/runtime path; Ubuntu/apt mirror endpoints
   depend on the user's OS configuration and are separate from the demo:

   ```text
   api.github.com
   github.com
   github-releases.githubusercontent.com
   objects.githubusercontent.com
   raw.githubusercontent.com
   pypi.org
   files.pythonhosted.org
   pypi.nvidia.com
   download.pytorch.org
   astral.sh
   omniverse-content-staging.s3-us-west-2.amazonaws.com
   ovextensionsprod.blob.core.windows.net
   nvidia.github.io
   ```

   The AWS S3 asset root appears in the Isaac Lab Kit app files as
   `https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`. The Azure Blob endpoint
   is used by Omniverse extension registry/cache resolution. If a corporate firewall blocks those endpoints,
   the demo can fail while resolving Kit extensions even when the local waterhose USD assets are present.

2. `isaacsim.core.experimental.primdata`: if this extension cannot be resolved, use this branch instead of
   commenting out Kit app dependencies by hand. The setup should use a compatible Isaac Sim source build and
   install the full workspace. If a registry sync still starts during launch, it usually means the local
   Isaac Sim build or extension cache is incomplete.

3. Apple Vision Pro: AVP teleoperation works on this branch using:

   ```bash
   ./waterhose.sh teleop --xr --cloudxr-env avp --vis kit --num-envs 1
   ```

   SpaceMouse is optional and only needed for desktop teleop. A missing SpaceMouse should not block AVP.

4. Reference normal output: a healthy smoke run should reach the following milestones, mixed with harmless
   Kit/RTX warnings:

   ```text
   [ISAACLAB] AppLauncher initialization complete
   [INFO]: Parsing configuration from: ...Waterhose...
   [INFO]: Completed setting up the environment
   [INFO]: Time taken for scene creation ...
   [INFO]: Time taken for simulation start ...
   [PROFILE] steps=... sim_time=... rollout_time=... cuda_graph=captured
   ```

   A healthy AVP XR run should additionally show:

   ```text
   [INFO]: Using IsaacTeleop stack for teleoperation
   [INFO]: CloudXR runtime auto-launched
   [INFO]: XR enabled changed to: True
   [INFO]: Acquired OpenXR handles from Kit XR bridge
   [INFO]: IsaacTeleop session started: WaterhoseTeleop
   ```

   `XR_ERROR_FORM_FACTOR_UNSUPPORTED` and some `XR_ERROR_FEATURE_UNSUPPORTED` hand-tracker probe messages
   can appear during OpenXR device probing and are not by themselves fatal. A fatal dependency-resolution
   block is different: it ends with `Failed to resolve extension dependencies` and should be treated as an
   installation/cache/firewall issue.
