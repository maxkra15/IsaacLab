# Waterhose robot demo

The waterhose demo has two supported entry points:

- `run_robot_demo.py` runs the scripted grasp, insertion, release, and retreat sequence.
- `teleop_se3_agent.py` runs bimanual Apple Vision Pro control.

Both use the same Newton proxy-coupled scene. The scripted runner intentionally does not launch the
human-driven teleop task.

## Install

Use a clean Python 3.12 environment associated with this worktree and a local Isaac Sim checkout:

```bash
ln -s /path/to/isaac-sim _isaac_sim
uv venv --python 3.12 .venv
./isaaclab.sh -i all
```

The internal waterhose assets are distributed separately as ``waterhose_demo_assets.tar.gz`` and are
not tracked by Git. Extract the archive from the Isaac Lab repository root; its paths are already rooted
at the correct task directory:

```bash
tar -xzf /path/to/waterhose_demo_assets.tar.gz -C .
test -f source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/rby1df/rby1df_waterhose.usda
```

Verify that the declared Newton build, Isaac Teleop, and WebSockets resolve from that environment:

```bash
./isaaclab.sh -p -c \
  'import importlib.metadata as m, newton, websockets; \
   print(newton.__file__); print(m.version("newton")); \
   print(m.version("isaacteleop")); print(websockets.__version__)'
```

## Visible runs on a hybrid-GPU Linux desktop

Point visible runs at the active X server. If that display is driven through an integrated GPU, use
PRIME render offload so both Kit's Vulkan renderer and the Newton OpenGL viewer select NVIDIA. Confirm
the provider name with `xrandr --listproviders` and replace `NVIDIA-G0` below if needed.

```bash
export DISPLAY=:1
export __NV_PRIME_RENDER_OFFLOAD=1
export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
```

The task uses one scene-fit camera pose for `ViewerCfg`, Kit, and the Newton viewer. Do not press the
Newton viewer's `F` shortcut for this scene: its dynamic-body framing omits the static fridge and can
place the camera inside it.

## Scripted demo

Run commands from this worktree so its launcher selects the matching environment. The launcher ignores
an activated sibling-worktree venv when this checkout has its own ``.venv``.

```bash
cd /home/maximiliank/Work/IsaacLab-rby1-waterhose-demo
```

Run the complete demo in Kit:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit \
  --device cuda:0 \
  --max_steps 4500
```

Use the same scene through Newton's standalone viewer by changing only the visualizer:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
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
CUDA run. Kit performs one RTX-safe physics warmup and then resets the complete scene before capturing
the controller, so that setup step does not alter the demonstrated trajectory.

For a quick non-rendered startup and physics smoke test:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer none \
  --device cuda:0 \
  --max_steps 100 \
  --profile
```

The high-fidelity defaults use 10 substeps and 20 VBD iterations. For faster experimentation, override
them per run with `WATERHOSE_SUBSTEPS` and `WATERHOSE_VBD_ITERS`.

## Apple Vision Pro teleoperation

CloudXR must be installed and configured for the Apple Vision Pro client. The first runtime launch may
pause for CloudXR EULA acceptance. Apply the visible-run environment above before starting Kit.

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --num_envs 1 \
  --visualizer kit \
  --xr \
  --device cuda:0 \
  --cloudxr_env avp \
  --debug_teleop
```

CloudXR auto-launch is enabled by default. Add `--no-auto_launch_cloudxr` when managing the runtime
separately. The explicit `--device cuda:0` above keeps this checkout's acceptance run on the requested
GPU. Waterhose XR defaults Kit to one renderer GPU
to avoid hybrid-GPU semaphore stalls during shutdown. Set `WATERHOSE_KIT_MULTI_GPU=1` only when multi-GPU
rendering is intentional and validated on the host.

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

Use `--max_steps 100 --cloudxr_env none` for a bounded XR startup smoke without launching CloudXR. A real
AVP acceptance pass must still verify:

1. The headset connects and the scene anchor faces the fridge.
2. Both wrists track XYZ and all three rotation axes one-to-one; right-hand pinch controls the gripper.
3. Robot and hose visuals update in the headset while physics is stepping.
4. Reset/reconnect works and stopping XR terminates the CloudXR process cleanly.
5. No mixed-WebSockets or target-frame warnings remain after startup.

## Record waterhose demonstrations

Use the Mimic task when the recording needs waterhose subtask metadata and the bimanual 15-D action space:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Waterhose-Coupled-Teleop-Mimic-v0 \
  --dataset_file ./datasets/waterhose.hdf5 \
  --num_demos 5 \
  --visualizer kit \
  --xr \
  --device cuda:0 \
  --cloudxr_env avp
```

`record_demos.py` uses the task's `make_recorder_manager_cfg()` when present, preserving Mimic datagen
terms instead of replacing them with the generic action/state recorder.

## Newton contact scope

The coupled scene has an explicit ownership boundary:

- MJWarp owns the articulated RBY1 and its contact with
  `FridgeRobotCollision/Housing`. The four active left/right finger SDF shapes use a raw MuJoCo
  `solref=(0.005, 1.0)`, and the outer collision pipeline refreshes on every 1 ms solver substep.
- VBD owns the articulated cable rods, the connector, the socket SDF, and the cable-clearanced housing.
  The connector mesh and inertia are lumped into cable segment 0, so it is physically part of the
  cable rather than a detached visual or welded rigid body.
- Proxy coupling mirrors all four finger bodies into VBD. Those proxies collide with both the connector
  and the cable rods. Finger-to-fridge and finger-to-socket proxy pairs are filtered because MJWarp
  already owns robot-to-fridge contact; solving that pair in both entries pushes the grasp away from the
  bore.

This branch pins Newton commit `32b69be8726f89bdb1f9ddf31984d1609c73c1bc`. It contains Newton PR
[#3262](https://github.com/newton-physics/newton/pull/3262), which adds water-tight full-surface
rigid-soft SDF contacts.

That feature is intentionally **disabled** for this scene. The current hose is built with `add_rod_graph`:
it is a flexible chain of rigid capsule bodies and cable joints, not a particle triangle surface. In
addition, Newton's current `SolverCoupledProxy` cannot harvest edge/face full-surface reactions and rejects
that mode explicitly. Enabling the flag would therefore add no hose contacts and would stop the coupled
solve. It becomes relevant if the hose is redesigned as an actual soft triangle surface or Newton adds
full-surface proxy-force harvesting.

The latest Newton pin is still useful for subsequent VBD and coupled-contact fixes. Isaac Lab seeds the
configured rigid-contact capacity before solver construction so VBD contact history is allocated before
CUDA graph capture.

## Troubleshooting

- `CloudXR requires websockets >= 14`: reinstall this worktree and verify that WebSockets resolves from its
  virtual environment. The launcher preloads one consistent WebSockets package before Kit starts.
- A traceback that names another worktree's ``.venv`` indicates a mixed environment. Run the command through
  this checkout's ``./isaaclab.sh``; do not invoke a sibling checkout's Python directly.
- No Kit window: check `DISPLAY`, then run with `--visualizer kit`.
- Hand directions are rotated or inverted: inspect the target-frame warning and confirm
  `/World/envs/env_0/Robot/Geometry/origin` exists in the live stage.
- Slow bring-up: use the bounded `--max_steps` smoke commands first, then restore high-fidelity settings for
  acceptance.
- CloudXR startup: inspect the runtime logs under `~/.cloudxr` and confirm any EULA prompt was accepted.
