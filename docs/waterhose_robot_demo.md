# Waterhose robot demo

The waterhose demo has two supported entry points:

- `run_robot_demo.py` runs the scripted grasp, insertion, release, and retreat sequence.
- `teleop_se3_agent.py` runs interactive keyboard, SpaceMouse, or Apple Vision Pro control.

Both use the same Newton proxy-coupled scene. The scripted runner intentionally does not launch the
human-driven teleop task.

## Install

Use a clean Python 3.12 environment associated with this worktree and a local Isaac Sim checkout:

```bash
ln -s /path/to/isaac-sim _isaac_sim
uv venv --python 3.12 .venv
./isaaclab.sh -i all
```

Verify that the declared Newton build, Isaac Teleop, and WebSockets resolve from that environment:

```bash
./isaaclab.sh -p -c \
  'import importlib.metadata as m, newton, websockets; \
   print(newton.__file__); print(m.version("newton")); \
   print(m.version("isaacteleop")); print(websockets.__version__)'
```

## Scripted demo

Run the complete demo in Kit:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 \
  --num_envs 1 \
  --visualizer kit \
  --device cuda:0 \
  --max_steps 4500 \
  --profile
```

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

## Desktop teleoperation

Keyboard:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-Coupled-Teleop-v0 \
  --num_envs 1 \
  --visualizer kit \
  --device cuda:0 \
  --teleop_device keyboard \
  --cloudxr_env none
```

Use `--teleop_device spacemouse` for a configured SpaceMouse. Passing `--cloudxr_env none` prevents a
CloudXR runtime from being started for desktop control.

## Apple Vision Pro teleoperation

CloudXR must be installed and configured for the Apple Vision Pro client. The first runtime launch may
pause for CloudXR EULA acceptance. If the shell has no display configured, point it at the active local X
server before starting Kit.

```bash
export DISPLAY=:1
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
separately. The launcher also selects `cuda:0` for an implicit waterhose XR device, but the explicit
`--device cuda:0` above makes the acceptance run unambiguous.

The waterhose pipeline maps right-wrist translation to end-effector translation, wrist roll to connector
twist, and pinch to the gripper. Torso and left-arm targets remain pinned. The configured robot-base target
frame keeps hand axes aligned with the task; if that prim cannot be read, the runtime logs a reason and
temporarily leaves poses in world frame.

Use `--max_steps 100 --cloudxr_env none` for a bounded XR startup smoke without launching CloudXR. A real
AVP acceptance pass must still verify:

1. The headset connects and the scene anchor faces the fridge.
2. Right-wrist XYZ, wrist roll, and pinch all move the intended controls.
3. Robot and hose visuals update in the headset while physics is stepping.
4. Reset/reconnect works and stopping XR terminates the CloudXR process cleanly.
5. No mixed-WebSockets or target-frame warnings remain after startup.

## Record waterhose demonstrations

Use the Mimic task when the recording needs waterhose subtask metadata and the 7-D relative action space:

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

This branch pins Newton commit `0cfa498b3b1b8c66dfd5853d2455939b370479c9`. It contains Newton PR
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
- No Kit window: check `DISPLAY`, then run with `--visualizer kit`.
- Hand directions are rotated or inverted: inspect the target-frame warning and confirm
  `/World/envs/env_0/Robot/Geometry/origin` exists in the live stage.
- Slow bring-up: use the bounded `--max_steps` smoke commands first, then restore high-fidelity settings for
  acceptance.
- CloudXR startup: inspect the runtime logs under `~/.cloudxr` and confirm any EULA prompt was accepted.
