# Waterhose Robot Demo

Last verified: 2026-06-15.

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
  geometry.py
  scripted_state_machine.py
  teleop.py
  teleop_pipelines.py
  teleop_pipelines_legacy.py
  waterhose_env_cfg.py
  assets/
  mdp/
  config/
    rby1df/
      __init__.py
      coupled_env_cfg.py
      teleop_env_cfg.py
      admm_env_cfg.py
      agents/
```

The package exposes two supported task variants:

| Task ID | Purpose |
| --- | --- |
| `Isaac-Waterhose-Coupled-v0` | Client-facing RBY1DF waterhose demo using Newton proxy coupling and absolute Newton IK actions for scripted demo and XR. |
| `Isaac-Waterhose-Coupled-Teleop-v0` | Same coupled scene with relative Newton IK actions and `env_cfg.teleop_devices` for native keyboard/SpaceMouse teleop. |

A third task, `Isaac-Waterhose-Admm-v0` (`admm_env_cfg.py`), is also registered but is an
**experimental** ADMM-coupling variant used for internal solver experiments; it is not part of
the supported demo and may be unstable.

The coupled task is the default and primary demo path. The previous local
`Isaac-Waterhose-Kinematic-v0` workaround has been removed so the demo exercises the real two-way
Newton proxy coupling path. It uses normal IsaacLab scene configuration:

- `ArticulationCfg` for the RBY1DF robot USD.
- `RigidObjectCfg` for the plugs and kinematic cable anchors.
- `CableObjectCfg` for the two USD cable curves.
- `AssetBaseCfg` for the fridge/static visual scene.
- wrapper USD layers (`fridge_waterhose.usda`, `rby1df_waterhose.usda`) for task-specific collision
  overrides, including the socket SDF and right gripper finger SDF colliders.
- `CoupledNewtonCfg` plus `isaaclab_newton.physics.CoupledSolverCfg`.

`CoupledSolverCfg.class_type` resolves to `isaaclab_newton.physics.coupled_manager:NewtonCoupledManager`. That manager partitions one Newton model into an MJWarp source view for the robot and a VBD destination view for the cable/plug bodies, then builds Newton's `SolverCoupledProxy` from `newton.solvers.experimental.coupled`.

## Standalone Setup

For a clean handoff, send these files together:

```text
waterhose-setup.sh
waterhose_demo_assets.tar.gz
docs/waterhose_robot_demo.md
docs/newton_local_setup.md
```

On a fresh Linux workstation:

```bash
mkdir -p ~/waterhose-handoff
cd ~/waterhose-handoff

chmod +x ./waterhose-setup.sh
./waterhose-setup.sh setup --accept-eula --assets-tar ./waterhose_demo_assets.tar.gz

cd waterhose-demo/IsaacLab-waterhose
```

`init` is an alias for `setup`. The setup script only performs setup: it clones this branch, builds
Isaac Sim, creates `.venv`, installs the full Isaac Lab workspace with `isaaclab.sh -i all`, unpacks
the demo assets, and runs a short headless smoke check unless `--skip-smoke` is passed.

Newton is pinned in the source tree to upstream Newton PR 2848 (coupled-solver) commit
`6409c9f454a8222ca5ab7119eb5102148aab0af5`, resolved on 2026-06-15. A fresh handoff install
resolves this commit from GitHub via the pyproject pin and does not require any local Newton
checkout. (A developer machine may additionally `pip install -e` a local Newton checkout that
carries an extra, not-yet-upstreamed "immovable proxy" patch, but the waterhose demo does not
use it, so the pinned commit alone is sufficient.)

The setup script intentionally does not wrap runtime commands. Run demo, profile, and teleop commands
directly from the `waterhose-demo/IsaacLab-waterhose` checkout so the task, device, CloudXR profile,
and visualizer choices are explicit.

## Running

Set the display explicitly for visible Kit/Newton viewer sessions when launching from a shell that does
not already export `DISPLAY`:

```bash
export DISPLAY=:1
```

Scripted demo with Kit:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit
```

Scripted demo with the Newton viewer:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer newton
```

Headless profile. Prefer `--visualizer none`; do not add the legacy `--headless` flag:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer none \
  --max_steps 200 \
  --profile
```

Multi-env scaling profile:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 128 \
  --visualizer none \
  --max_steps 100 \
  --profile
```

SpaceMouse desktop teleop uses the relative-action teleop task:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --num_envs 1 \
  --teleop_device spacemouse \
  --visualizer kit
```

Keyboard desktop teleop uses the same relative-action task:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --visualizer kit
```

The scripted/XR task uses an absolute 8D Newton IK action space configured through `env_cfg.isaac_teleop`.
The desktop teleop task uses a relative 7D action space configured through `env_cfg.teleop_devices`, matching
IsaacLab's native keyboard and SpaceMouse devices.

The SpaceMouse entry uses a waterhose-specific mapping:

- Cap translation moves the gripper in XYZ with the same signs as the original stable waterhose demo.
- Cap twist rotates the gripper around the insertion/yaw axis.
- Roll and pitch are suppressed to keep the plug aligned.
- Translation suppresses accidental twist noise; use a pure twist motion for yaw.

## Apple Vision Pro

Apple Vision Pro XR teleoperation is verified on this branch with the IsaacTeleop/OpenXR path. Launch from
the Isaac Lab checkout:

```bash
export DISPLAY=:1

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit \
  --xr \
  --cloudxr_env avp \
  --debug_teleop
```

Open the XR panel in Isaac Sim, select `OpenXR` and `System OpenXR Runtime`, then click `Start XR`.
The AVP and workstation must be IP-reachable on the same wireless network or routed VLAN. If the
workstation also has Ethernet connected, enter the workstation's Wi-Fi IP in the AVP app and allow the
CloudXR ports on that interface.

The AVP does not need to use the same iCloud account as the Mac at runtime. For Xcode installation,
use an Apple developer account/provisioning setup that can sign and trust the Isaac XR Teleop sample app
on the headset; using the same Apple ID on the Mac and AVP is the simplest path.

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

Before wearing the headset, test the signaling port from the Mac:

```bash
nc -vz <workstation-wifi-ip> 48010
```

If this succeeds from the Mac but times out from the AVP, the usual causes are headset local-network
permission, the AVP being on a different SSID/VLAN, client isolation on the Wi-Fi network, or the AVP app
still pointing at the Ethernet IP instead of the Wi-Fi IP.

## Meta Quest / Pico

Meta Quest and Pico use the CloudXR.js web profile:

```bash
export DISPLAY=:1

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit \
  --xr \
  --cloudxr_env cloudxrjs \
  --debug_teleop
```

Open `https://nvidia.github.io/IsaacTeleop/client` in the headset browser, enter the workstation Wi-Fi IP,
accept the self-signed certificate at `https://<workstation-wifi-ip>:48322/`, return to the CloudXR.js
client page, then connect.

Minimum CloudXR.js firewall rules:

```bash
sudo ufw allow 49100/tcp
sudo ufw allow 48322/tcp
sudo ufw allow 47998/udp
```

## Batching Status

The task uses normal IsaacLab cloned scene setup (`replicate_physics=True`, regex prim paths, per-env
cable anchors, and batched Torch actions/state). On 2026-06-10, the coupled task completed 100-step
headless non-teleop profile runs with CUDA graph capture at `--num_envs 1`, `8`, and `128`. The wall-step
rates on the local workstation were 25.7, 22.5, and 17.2 manager steps/s, which corresponds to about 25.7,
180, and 2202 effective env-steps/s. The current Newton coupled solver path is functionally batched in
play/demo mode; teleop should still be kept at one env because XR input and visualization are
single-operator workflows.

## Assets

By default the task uses packaged assets next to the waterhose package:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/
```

Required files:

```text
fridge/fridge.usda
fridge/fridge_waterhose.usda
fridge/cable/cable001.usda
fridge/cable/plug.usda
rby1df/rby1df.usda
rby1df/rby1df_waterhose.usda
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
   export DISPLAY=:1

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
     --task Isaac-Waterhose-Coupled-v0 \
     --num_envs 1 \
     --visualizer kit \
     --xr \
     --cloudxr_env avp
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
