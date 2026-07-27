# Waterhose robot demo

The waterhose demo has two supported entry points:

- `run_robot_demo.py` runs the scripted grasp-and-insert sequence and finishes in a seated hold.
- `teleop_se3_agent.py` runs bimanual Apple Vision Pro control.

Both use the same Newton proxy-coupled scene. The scripted runner intentionally does not launch the
human-driven teleop task.

## Install

Use a clean Python 3.12 environment associated with this worktree and a runnable Isaac Sim release
layout. The `_isaac_sim` target must contain `setup_python_env.sh` (or `setup_conda_env.sh`) and
`apps/`; for an Isaac Sim source checkout, point it at the built release directory.

```bash
ln -s /path/to/isaac-sim-release _isaac_sim
uv venv --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -i all
```

The internal waterhose assets are distributed separately as ``waterhose_demo_assets.tar.gz`` and are
not tracked by Git. From the Isaac Lab repository root, verify the archive and unpack it:

```bash
ASSET_TARBALL=/path/to/waterhose_demo_assets.tar.gz
echo "4c40dca88b4f5db17ef8fabd073f13fde1031a159e2e1da1867ef57d22ea5248  $ASSET_TARBALL" \
  | sha256sum --check -
tar -xzf "$ASSET_TARBALL" -C .
```

The archive already contains the repository-relative
``source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets`` path. Keep both the extracted assets
and the archive local; the repository ignores them.

Verify that the tested Newton pin, Isaac Teleop, and WebSockets resolve from that environment:

```bash
.venv/bin/python -c \
  'import importlib.metadata as m, newton, websockets; \
   print(newton.__file__); print(m.version("newton")); \
   print(m.version("isaacteleop")); print(websockets.__version__)'
uv pip freeze --python .venv/bin/python | grep '^newton'
```

## Visible runs

```bash
export DISPLAY=:1
unset CUDA_VISIBLE_DEVICES
unset NV_GPU_INDEX NV_CXR_GPU_INDEX_VULKAN
export NV_CXR_GPU_INDEX_CUDA=0
unset __NV_PRIME_RENDER_OFFLOAD __NV_PRIME_RENDER_OFFLOAD_PROVIDER
unset __GLX_VENDOR_LIBRARY_NAME __VK_LAYER_NV_optimus
```

Do not derive a CloudXR Vulkan device index from the CUDA ordinal. On hybrid-GPU hosts the two APIs
enumerate different physical devices. The supported ``NV_CXR_GPU_INDEX_CUDA=0`` selector and
``--device cuda:0`` keep CloudXR, Kit, and simulation on the same NVIDIA GPU without changing Vulkan
device ordering.

The task uses one scene-fit camera pose for `ViewerCfg`, Kit, and the Newton viewer.

## Scripted demo

Run commands from the repository root. The launcher wrapper exports the Isaac Sim environment before
starting this worktree's Python.

Run the complete demo in Kit:

```bash
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit \
  --device cuda:0 \
  --max_steps 4500
```

Use the same scene through Newton's standalone viewer by changing only the visualizer:

```bash
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer newton \
  --device cuda:0 \
  --max_steps 4500
```

Visible Kit and Newton runs are paced to simulation time by default, so the complete sequence remains
open for at least about 28 seconds rather than flashing by at GPU benchmark speed (and takes longer
when rendering cannot sustain 100 Hz). Use ``--no-realtime --profile`` only when measuring throughput.
The profile footer should report both ``control_graph=captured`` and ``physics_graph=captured`` on a
default CUDA run. When physics capture is deferred (notably with Kit/RTX), the launcher performs one
warmup and resets the complete scene before capturing the controller, so setup does not alter the
demonstrated trajectory. `--debug_script` intentionally selects the eager controller for diagnostics.

For a quick non-rendered startup and physics smoke test:

```bash
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer none \
  --device cuda:0 \
  --max_steps 100 \
  --profile
```

The validated high-fidelity defaults use 10 substeps, 20 VBD iterations, and one fixed-feedback
proxy-coupling pass. For controlled experiments, override them per run with `WATERHOSE_SUBSTEPS`,
`WATERHOSE_VBD_ITERS`, and `WATERHOSE_COUPLING_ITERS`. More coupling passes are not a higher-fidelity
drop-in setting for this nonlinear contact scene: they changed the retained connector behavior in
testing. Re-run the full insertion, seated-hold, and linger acceptance sequence after changing any of
these values.

## Apple Vision Pro teleoperation

CloudXR must be installed and configured for the Apple Vision Pro client. The first runtime launch may
pause for CloudXR EULA acceptance. Apply the visible-run environment above before starting Kit.

```bash
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --num_envs 1 \
  --visualizer kit \
  --xr \
  --device cuda:0 \
  --cloudxr_env avp
```

CloudXR auto-launch is enabled by default. Add `--no-auto_launch_cloudxr` when managing the runtime
separately. The explicit `--device cuda:0` above keeps this checkout's acceptance run on the requested
GPU. Add `--enable_debug_visualization` only when hand-joint and controller-aim markers are useful.

The session starts paused and uses Isaac Teleop's standard AVP controls:

- **Play** starts applying wrist and gripper commands.
- **Stop** pauses physics stepping while keeping the XR session connected.
- **Reset** restores the robot and cable to their initial state and clears the relative wrist
  calibration. The first valid pose from each hand after reset is clutched to the reset robot pose, so
  the next episode does not inherit the previous episode's extended-arm offset.

Reset preserves whether the session is playing or stopped. For deliberate episode boundaries, use
**Stop**, **Reset**, align your hands comfortably, and then **Play**. The same controls can be activated
with visionOS Voice Control.

The 16-D waterhose pipeline maps complete absolute poses from both tracked wrists to the complete right
and left RBY1 arm chains, followed by independent right- and left-hand pinch commands. Each gripper uses
the standard binary hysteresis: a thumb-index distance below 3 cm closes it, a distance above 5 cm opens
it, and the state is retained between those thresholds. The closed target deliberately retains the
existing nonzero finger gap instead of commanding the joints fully shut. Each first valid (or reacquired)
hand pose is clutched to the robot's current gripper-base wrist pose, so neutral alignment does not depend
on guessed tool-frame Euler offsets. Subsequent translation and spatial rotation deltas stay one-to-one
on all three axes, and wrist tracking loss holds the last target. The shared torso is held stable. The
configured robot-base target frame keeps hand axes aligned with the task; if that prim cannot be read, the
runtime logs a reason and emits no action until the frame resolves rather than sending world-frame poses
to IK.

The bimanual task intentionally does not configure keyboard or SpaceMouse devices: those devices emit a
single 7-D arm command and do not satisfy the two-wrist 16-D action contract.

For a local XR startup check without launching CloudXR, replace `--cloudxr_env avp` with
`--cloudxr_env none` and close the application after the scene and teleop session initialize. A real AVP
acceptance pass must still verify:

1. The headset connects and the scene anchor faces the fridge.
2. Both wrists track XYZ and all three rotation axes one-to-one; each hand's pinch controls its gripper.
3. Robot and hose visuals update in the headset while physics is stepping.
4. Play and Stop gate motion, and Reset restores the scene and re-clutches both wrists.
5. No mixed-WebSockets or target-frame warnings remain after startup.

## Mimic annotation and generation

AVP collection remains a 16-D operator-input task. Its recorder also stores a 20-D
`processed_actions` stream containing the robot-side targets actually sent downstream:

```text
right wrist pose (7), left wrist pose (7),
right gripper joint targets (3), left gripper joint targets (3)
```

`Isaac-Waterhose-Coupled-Mimic-v0` consumes that 20-D stream directly. It does not run the AVP
clutch again, and annotated output is written back under the standard `actions` key expected by
Mimic generation. The initial configuration treats each arm's complete trajectory as one final
subtask, so automatic annotation does not require intermediate subtask signals.

```bash
VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p \
  scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Waterhose-Coupled-Mimic-v0 \
  --input_file /path/to/waterhose_teleop.hdf5 \
  --output_file /path/to/waterhose_annotated.hdf5 \
  --auto --headless --visualizer none --device cuda:0

VIRTUAL_ENV="$PWD/.venv" ./isaaclab.sh -p \
  scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Waterhose-Coupled-Mimic-v0 \
  --input_file /path/to/waterhose_annotated.hdf5 \
  --output_file /path/to/waterhose_generated.hdf5 \
  --num_envs 1 --headless --visualizer none --device cuda:0
```

## Newton contact scope

The coupled scene has an explicit ownership boundary:

- MJWarp owns the articulated RBY1 and its contact with
  `FridgeRobotCollision/Housing`. In addition to the four left/right finger SDFs, runtime stage
  authoring enables the asset's dedicated gripper-base, camera-bracket, tool-flange, and wrist-pitch
  collision meshes on both arms. All twelve robot contact shapes use a raw, critically damped MuJoCo
  `solref=(0.005, 1.0)`, and the outer collision pipeline refreshes on every 1 ms solver substep.
- VBD owns the articulated cable rods, the connector, the socket SDF, and the cable housing proxy.
  Cable-to-fridge contact is therefore solved directly inside VBD and does not use proxy feedback.
  The housing remains collidable everywhere except for a 15 mm local insertion corridor, where the
  dedicated socket SDF supplies the physical insertion contact.
  The connector mesh and inertia are lumped into cable segment 0, so it is physically part of the
  cable rather than a detached visual or welded rigid body. VBD preserves authored joint modes: the
  rod's stretch/bend joints remain compliant, while the tail attachment uses a finite penalty
  constraint by authoring `vbd:joint_is_hard=0` on that joint alone. Other non-cable structural
  joints retain Newton's hard augmented-Lagrangian default.
- Proxy coupling mirrors all four finger bodies into VBD. Those proxies collide with both the connector
  and the cable rods. Finger-to-fridge and finger-to-socket proxy pairs are filtered because MJWarp
  already owns robot-to-fridge contact; solving that pair in both entries pushes the grasp away from the
  bore. The shared outer pipeline is likewise reduced to pairs consumed by MJWarp; cable/socket/gripper
  pairs are detected only by the proxy-local pipeline. The proxies retain the imported model-view
  finger inertia (`use_solver_effective_mass=False`), use `mass_scale=1.0`, and run one staggered,
  fixed-feedback coupling pass without Aitken relaxation. The task contact material uses
  `ke=1.0e4 N/m` and `kd=0.1 N s/m`; the finger proxies use friction `20.0`, a 1 mm physical margin,
  and a 10 mm broad-phase gap, while connector/socket/cable shapes retain the lower task friction.
  AVP gripper commands preserve the published client task's 150 mm per-step limit. That exceeds the
  complete 76 mm driver stroke, so open/close transitions complete in one simulation step while all
  three gripper joints remain synchronized. The scripted demo keeps its direct target path.

Insertion and retention are pure contact physics. There is no connector latch, snap constraint,
adhesion, or post-insertion kinematic hold. The success check only observes connector alignment and
depth; it does not apply forces. The plug remains seated through its mesh contact with the socket,
contact friction, and the cable's elastic response.

This branch pins the tested Newton commit `81cdcfc2dd89f8b7285e32b5e3853092a97fa6f9`. Isaac Lab seeds the
configured rigid-contact capacity before solver construction so VBD contact history exists before CUDA
graph capture. The default model-wide history capacity is 131,072. The default proxy-local capacity
scales as 256 contacts per environment with a 30,000-contact floor. If a larger batch exceeds that envelope, increase
`WATERHOSE_RIGID_CONTACT_MAX` and `WATERHOSE_PROXY_RIGID_CONTACT_MAX` together; the former must be at least
the latter. These allocation limits do not change stiffness, friction, margins, substeps, or iterations.

## Troubleshooting

- `CloudXR requires websockets >= 14`: reinstall this worktree and verify that WebSockets resolves from its
  virtual environment. The launcher preloads one consistent WebSockets package before Kit starts.
- A traceback that names another worktree's ``.venv`` indicates a mixed environment. Run the command with
  `VIRTUAL_ENV="$PWD/.venv"` and this checkout's ``./isaaclab.sh -p`` wrapper; do not invoke a sibling
  checkout's Python directly.
- No Kit window: check `DISPLAY`, then run with `--visualizer kit`.
- Hand directions are rotated or inverted: inspect the target-frame warning and confirm
  `/World/envs/env_0/Robot/Geometry/origin` exists in the live stage.
- Slow bring-up: use the bounded `--max_steps` smoke commands first, then restore high-fidelity settings for
  acceptance.
- CloudXR startup: inspect the runtime logs under `~/.cloudxr` and confirm any EULA prompt was accepted.
