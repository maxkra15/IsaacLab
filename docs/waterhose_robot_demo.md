# Waterhose Robot Demo

Last verified: 2026-06-25.

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
| `Isaac-Waterhose-Coupled-v0` | Client-facing RBY1DF waterhose demo using Newton proxy coupling and absolute differential-IK end-effector actions for the scripted demo and XR. |
| `Isaac-Waterhose-Coupled-Teleop-v0` | Same coupled scene with relative differential-IK actions and `env_cfg.teleop_devices` for native keyboard/SpaceMouse teleop. |

A third task, `Isaac-Waterhose-Admm-v0` (`admm_env_cfg.py`), is also registered as an ADMM-coupling
variant for solver comparison. It shares the same scene and reaches the same grasp-insert result, but is
slower than the proxy default and is not the supported demo path (see *Performance and batchability*).

The coupled task is the default and primary demo path. The previous local
`Isaac-Waterhose-Kinematic-v0` workaround has been removed so the demo exercises the real two-way
Newton proxy coupling path. It uses normal IsaacLab scene configuration:

- `ArticulationCfg` for the RBY1DF robot USD.
- `RigidObjectCfg` for the plug and the kinematic cable-tail anchor.
- `CableObjectCfg` for the USD cable curve.
- `AssetBaseCfg` for the fridge/static visual scene.
- wrapper USD layers (`fridge_waterhose.usda`, `rby1df_waterhose.usda`) for task-specific collision
  overrides, including the socket SDF and right gripper finger SDF colliders.
- `CoupledNewtonCfg` plus `isaaclab_newton.physics.CoupledSolverCfg`.

`CoupledSolverCfg.class_type` resolves to `isaaclab_newton.physics.coupled_manager:NewtonCoupledManager`. That manager partitions one Newton model into an MJWarp source view for the robot and a VBD destination view for the cable/plug bodies, then builds Newton's `SolverCoupledProxy` from `newton.solvers.experimental.coupled`.

## Solvers, coupling, and collision

- **Two solvers, one model.** MuJoCo-Warp (`SolverMuJoCo`) integrates the articulated RBY1DF robot;
  VBD/AVBD (`SolverVBD`) integrates the deformable Cosserat-rod cable plus the welded rigid plug and the
  kinematic tail anchor. `NewtonCoupledManager` partitions one Newton model into per-solver **entries**
  (`"mjc"` = robot, `"vbd"` = cable/plug/anchor) selected by `shape_label_patterns`.
- **Default coupling is proxy** (`Isaac-Waterhose-Coupled-v0` → `SolverCoupledProxy`): the right-gripper
  finger bodies are mirrored into the VBD solver as driven *proxy bodies*, so the cable/plug collide
  against the gripper (the grasp), with the contact reaction harvested back to the robot (two-way).
  `Isaac-Waterhose-Admm-v0` is an ADMM variant (`SolverCoupledADMM`, force consensus) — stable but
  slower; it shares the same scene and entries.
- **The fridge housing is a world-static collider shared by both solvers.** The connector housing is a
  **single welded, decimated concave mesh** (`Cable008_BodyCollision`, a few hundred triangles so the
  contact batches) and the socket bore is `Cable008_SocketCollision` — both authored as world-static
  shapes in one asset. The **robot** collides the housing directly in MJWarp (`use_mujoco_contacts=False`)
  **and** the **cable** collides it in VBD — the same single mesh, no duplication. This works because a
  world-static shape is auto-included into any entry whose shape list is empty (Newton's
  `_entry_visible_shapes` makes every `shape_body < 0` shape visible there, as *shared visibility*, not
  ownership — the "owned by >1 entry" error only fires on explicit listings). The VBD/cable entry keeps
  an empty list (so it auto-includes both the housing **and** the socket → cable↔housing and
  plug↔socket). The MJWarp/robot entry lists the housing **explicitly** so it collides the housing but
  *not* the socket — the gripper must not fight the bore during insertion; only the plug seats into it.
  Robot collision is further restricted to the right gripper by a `MODEL_INIT` hook (the rest of the arm
  does not generate contacts). `WATERHOSE_FRIDGE_COLLISION=0` clears the housing mesh's collide flags for
  both sides (≈1.5× faster at scale) while leaving the socket, so the plug still inserts.

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

Newton is pinned in the source tree to the upstream Newton PR 2848 (coupled-solver) head commit
`526b36396777c18b82af8f30c4693b7c8bb4d89d`, resolved on 2026-06-22. Every Newton direct URL in
`source/*/pyproject.toml` (and the wheel builder's package list) resolves to this one commit, so a
fresh handoff install pulls a single, consistent Newton from GitHub and does not require any local
Newton checkout. See `docs/newton_local_setup.md` for how to refresh the pin to a newer PR 2848 head.

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

The scripted/XR task uses an absolute 8D differential-IK action space (right end-effector pose plus a
normalized gripper command) configured through `env_cfg.isaac_teleop`. The desktop teleop task uses a
relative 7D action space configured through `env_cfg.teleop_devices`, matching IsaacLab's native keyboard
and SpaceMouse devices.

The SpaceMouse entry uses a waterhose-specific mapping:

- Cap translation moves the gripper in XYZ.
- Cap twist rolls the gripper about its own approach axis (the relative IK applies it in the
  end-effector frame), spinning the held plug to line its keying up with the bore.
- Wrist pitch and yaw are suppressed to keep the plug aligned.
- Translation and twist are independent, so the operator can move and roll at the same time; a small
  `twist_deadzone` rejects the twist cross-talk the cap reports during a translation push. Flip
  `twist_sign` on `WaterhoseSpaceMouseCfg` if the roll direction feels inverted.

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

## Performance and batchability

The task uses normal IsaacLab cloned-scene setup (`replicate_physics=True`, regex prim paths, per-env
cable anchors, and batched Torch actions/state) and is fully CUDA-graph captured, so it batches headless.
The workload is **throughput-bound, not memory-bound** (the hose is a thin Cosserat rod with no particle
field), and the per-step cost is dominated by the VBD/AVBD cable+contact solve (~40%); the CUDA graph is
worth ~6×.

Measured on a 32 GiB RTX 5090 (proxy coupling, the default):

- **Proxy is the faster backend at every world count and scales to 6144 worlds** —
  **32,544 env-steps/s at 6144** (the peak), climbing monotonically
  (10.9k→17.7k→24.7k→29.2k→31.2k→32.5k from 512 to 6144). 8192 crashes (NaN at ~26 GiB — the
  deterministic-contact-matching cap, not OOM).
- **ADMM peaks at 1024 (~12.3k env-steps/s)** and is ~2.6× slower than proxy at scale (and heavier:
  15.5 GiB at 2048 vs proxy's 12.2). Use proxy.
- **Startup grows super-linearly** (~24 s at 512 to ~14 min at 6144), so the throughput/iteration
  **sweet spot is 2048–4096**. Tune the per-step cost with `WATERHOSE_SUBSTEPS`/`WATERHOSE_VBD_ITERS`
  (see below).
- Keep teleop and XR runs at **one environment** (single-operator workflows).

Full numbers, plots, and the per-step breakdown: `_scratch/reports/waterhose_scaling/`
(`waterhose_env_scaling_report.tex` + the `bench_sweep.py`/`bench_child.py` drivers).

## Environment-variable flags

- `WATERHOSE_FRIDGE_COLLISION` (default **on**): connector-housing collision. The housing is a shared
  world-static mesh — the robot (MJWarp) **and** the cable (VBD) both collide it (robot↔housing **and**
  cable↔housing). `0` clears the housing mesh's collide flags for both sides (≈1.5× faster at scale,
  removing the hose↔body soft contacts) while leaving the socket. The plug↔socket insertion contact is
  independent of this flag (only the cable collides the socket — the robot is scoped to the housing so
  the gripper does not fight the bore).
- `WATERHOSE_SUBSTEPS` (default **8**) / `WATERHOSE_VBD_ITERS` (default **16**): the coupled-solver
  substep count and VBD iteration count — the throughput/stability knob. The 8/16 default keeps the
  scripted-demo arc unchanged; `WATERHOSE_SUBSTEPS=6 WATERHOSE_VBD_ITERS=12` is ~1.44× faster per step
  (good for training; the plug seats a little slower).
- `WATERHOSE_SOCKET_SDF` (default **off**): the socket bore is a plain triangle-mesh (BVH) collider; `1`
  upgrades it to a texture-SDF (smoother insertion gradient, at the cost of a per-env SDF build at
  startup). The SDF is applied in code (`spawn_fridge_with_socket_sdf`), not baked in the USD, so the
  flag is the single source of truth.
- `WATERHOSE_ASSETS_DIR`: override the packaged asset root (also exposed as `--asset_root`). Point this
  at the unpacked `waterhose_demo_assets.tar.gz` (`…/WaterhoseDemo`) on a site where git-LFS is blocked
  and the in-tree assets did not pull — see *Offline operation and firewall*.
- `WATERHOSE_ALLOW_NETWORK` (default **off**): the runner disables Kit's extension registry and
  telemetry by default so a firewalled site needs no external access. Set `1` to restore Kit's
  registry/telemetry network calls (e.g. a connected dev box that wants the extension registry).

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
skies/kloofendal_43d_clear_puresky_4k.hdr          # sky dome (bundled, was Isaac S3)
ground/default_environment.usd                     # grid ground (bundled, was Isaac S3)
ground/Materials/Textures/*.png                    # grid textures (relative refs)
```

Every asset the demo loads is in-tree, so the runtime makes **no external (S3/Nucleus) asset fetch**.
The sky dome and grid ground default upstream to the Isaac cloud asset root (AWS S3); we ship local
copies under `skies/` and `ground/` instead and point the scene config at them. The ground USD uses
relative texture paths and the core `OmniPBR.mdl` (resolved locally), so it is self-contained offline.

To use another asset directory:

```bash
WATERHOSE_ASSETS_DIR=/path/to/waterhose/assets   # or: --asset_root /path/to/waterhose/assets
```

`waterhose_demo_assets.tar.gz` packages this exact tree under `WaterhoseDemo/` (untracked in git, copied
alongside the setup script). On a site where git-LFS is blocked, unpack it and run with
`--asset_root <unpacked>/WaterhoseDemo`.

## Offline operation and firewall

The demo runs **fully offline at runtime** — no external endpoints are contacted once it is installed.
This is the default, verified by running the scripted demo to completion with all network blocked.

What makes it offline (already done in this repo):

- **Assets are local.** Every USD plus the sky dome HDR and grid ground are bundled in-tree / in the
  asset tar (see *Assets*). Nothing resolves to the Isaac cloud asset root
  (`omniverse-content-production.s3-us-west-2.amazonaws.com`).
- **Kit registry + telemetry are off.** The runner appends
  `--/app/extensions/registryEnabled=0 --/telemetry/enableAnonymousData=0 --/telemetry/enableAnonymousAppName=0`
  (`_apply_offline_kit_args`), so Kit never contacts the extension registry
  (`ovextensionsprod.blob.core.windows.net`) or NVIDIA telemetry. A source-built Isaac Sim already has
  every extension on disk, so the registry is unused at runtime. Set `WATERHOSE_ALLOW_NETWORK=1` to
  re-enable both. For non-runner launches (teleop / Apple Vision Pro), pass the same three `--/…` kit
  args, or set them once in `apps/isaaclab.python*.kit` (`registryEnabled`, `[settings.telemetry]`).

**Build vs. runtime.** The above is *runtime*. The setup script (`waterhose-setup.sh`) builds Isaac Sim
from source and runs `isaaclab.sh -i all`, which *does* need network. On a locked-down site, build on a
connected machine and transfer the workspace, or allowlist the build endpoints for the install only. If
git-LFS is blocked, the in-tree assets will not pull — use the asset tar
(`--asset_root <unpacked>/WaterhoseDemo`), which carries the full runtime asset tree.

### Firewall allowlist (backup)

If you prefer to grant access rather than run offline, these are the only endpoints the demo would
otherwise reach at **runtime** (all HTTPS / 443):

| Host | Purpose | Still needed? |
| --- | --- | --- |
| `omniverse-content-production.s3-us-west-2.amazonaws.com` | Isaac asset root `/Assets/Isaac/6.0` — sky HDR, grid ground | No — assets bundled locally |
| `ovextensionsprod.blob.core.windows.net` | Kit extension registry (Azure) | No — disabled by default; source build has all exts |
| `*.nvidia.com` (e.g. `telemetry.omniverse.nvidia.com`) | Anonymous Kit telemetry | No — disabled by default |

**Build / install only** (the source build), in addition to the above: `github.com`, `astral.sh`,
`pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`, and `*.nvidia.com` + NVIDIA packman CDNs.
Capture a full connection log during your first install to confirm the exact NGC/packman hosts for your
release.

## Architecture map

- **`waterhose_env_cfg.py`** — scene, MDP, the `CoupledNewtonCfg` solver (MJWarp + VBD entries, proxy by
  default with an ADMM variant), the action cfgs, and the env variants. Physics tuning constants live at
  the top (`_VBD_*`, `_GRIPPER_*`). Three `MODEL_INIT` builder hooks run before `finalize()`:
  `_restrict_rby1df_collision_to_right_gripper`, `_disable_anchor_collision`,
  `_merge_plug_shape_into_cable_head` (re-parents the plug onto the cable head so the connector is rigidly
  part of the rod), plus `_hide_fridge_collider_visuals` (viewer-only).
- **`scripted_state_machine.py`** — the demo policy (REST→…→DONE). Reads the live plug pose from the
  Newton solver state (so the pick is agnostic to the cable's resting pose) and emits the multi-body IK
  action + gripper command.
- **`mdp/actions.py`** — the gripper and Newton-IK action terms (scripted/XR absolute IK; teleop relative
  IK). **`mdp/terminations.py`** — the `plug_inserted_in_socket` success predicate.
- **`geometry.py`** — shared poses, offsets, and collider label patterns. **All quaternions are
  `(x, y, z, w)`** (USD-authored `(w,x,y,z)` is converted here).
- **`teleop.py` / `teleop_pipelines.py`** — SpaceMouse device + IsaacTeleop XR pipelines.

## Handover gotchas

- **Quaternions are `(x, y, z, w)` everywhere** (Isaac Lab math + Newton IK).
- **Do not delete `teleop_pipelines_legacy.py`.** It is a deliberate byte-for-byte fallback of the last
  known-good XR pipeline; switch the import in `waterhose_env_cfg.py` back to it if a pipeline refactor
  regresses the live session.
- **The cable tail anchor must be a per-env body**, not the shared world body (`-1`) — a fixed joint to
  the world body NaNs the multi-env coupled solve at step 0.
- **The plug's `RigidObject` asset view is stale for a coupled body**; read its pose from
  `NewtonManager.get_state_0().body_q` (the state machine and the success term already do this).
- **Contact stiffness comes from the model-wide `NewtonModelCfg.shape_material_*` fill**, which overwrites
  per-shape USD/material contact properties — tune it there, not on the USD.
- **Viewer geometry (Visuals vs Collisions).** Kit renders by USD `purpose`; the **Newton** viewer keys on
  `ShapeFlags`. `_hide_fridge_collider_visuals` clears `VISIBLE` on the fridge colliders so they show only
  under the viewer's *Collisions* toggle, and the Newton cfg sets `show_static=False` so the static
  fridge/ground obey the toggles.
- **Newton pin:** PR-2848 head `526b3639` plus a local *immovable one-way kinematic proxy* commit on top
  (see `docs/newton_local_setup.md` to refresh).

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
