# GB300 Ethernet cable insertion

`IsaacContrib-Rizon-Sharpa-GB300-Cable-Install` is a Newton manager-based RL
task with a fixed-base Rizon 4s and a Sharpa right hand. A short, stiff cable
starts grasped, with its other connector already seated in the upper switch
port. The objective is to insert, latch, and release the loose connector in the
lower port. Gravity and physical cable attachment remain active.

The task uses Isaac Lab's `CableObject`, native Newton IK, observation/event/
reward/termination managers, coupling configurations, and RSL-RL PPO runner.
The custom builder hook authors the two sockets, connector housings, and spring
clip. The active cabinet uses a simple shell; the ports use detailed SDF contact
geometry. Optional scenery references eight instances of the original GB300
CAD, without adding background bodies or collisions.

## Training

The default preset has 28 normalized actions: six incremental palm commands and
22 reset-relative finger targets. Its state-based observations include robot
joints, connector and goal poses, clip state, cable positions, and previous
actions. Success uses measured seating, latch capture, and hand withdrawal;
it does not require the scripted finger pose. The native VBD/proxy configuration
uses the repository's standard Newton dependency.

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Rizon-Sharpa-GB300-Cable-Install \
  --num_envs 32 --viz none
```

The runner is a baseline, not a trained policy. Increasing environment count
requires checking solver/contact capacity and GPU memory. Camera observations,
domain randomization, and a removal reward are not part of this initial task.

## Continuous physical example

The example inserts, opens and withdraws the hand, regrasps the housing,
presses the clip with the thumb, and extracts the connector in one episode.
Controllers produce normal robot actions only: no grasp attachment, body-pose
writes, hidden reset at the handoff, or connector gravity cancellation.

It selects `presets=scripted`: 29 absolute actions (palm position, XYZW
quaternion, and finger targets), hydroelastic hand/housing/clip contact, and
ADMM coupling to the VBD cable. These actions and physics differ from the
default training preset; checkpoints are not interchangeable.

```bash
uv run python examples/gb300_sharpa_insertion.py --viz newton_gl
uv run --extra video python examples/gb300_sharpa_insertion.py \
  --viz newton_gl --video --output_dir logs/demos/gb300_sharpa
uv run --extra video python examples/gb300_sharpa_insertion.py \
  --viz none --video --output_dir logs/demos/gb300_sharpa_headless
uv run --extra isaacsim --extra video python examples/gb300_sharpa_insertion.py \
  --viz kit --video --output_dir logs/demos/gb300_sharpa_kit
```

Use `--no_presentation --viz none` for physics-only verification. `--num_steps`
bounds the episode; success requires completing both insertion and extraction
without an environment reset. Recording uses `VideoRecorderCfg` and unique
filenames. Promo compositing and HUD design are separate from the task.

### Newton hydroelastic dependency

The full scripted clip demo currently needs two corrections not included in
the Newton version pinned by upstream Isaac Lab:

- Convert per-contact hydroelastic stiffness/damping through the contact
  inverse effective mass when constructing MJWarp constraints.
- Avoid CPU contact-counter reads during CUDA graph capture in the
  hydroelastic SDF collision kernel.

Use an explicitly corrected Newton checkout via `PYTHONPATH` for this example.
The task neither patches the installed package nor changes the repository's
dependency pin. The native training preset does not need these corrections.
Upstreaming the corrections is a prerequisite for a self-contained public
hydroelastic example.

## Assets and redistribution

Asset files are not committed. Resolution validates the pinned local files,
and reports missing dependencies rather than downloading private assets.

### Rizon 4s / Sharpa (required)

The current robot comes from NVIDIA's access-controlled
[Fabrics-Sim repository](https://gitlab-master.nvidia.com/dex/fabrics-sim.git),
revision `d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`, at:

```text
src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/
  rizon4s_sharpa_no_spheres/rizon4s_sharpa_no_spheres_generated.usd
```

Copy that USD and its sibling `textures/` directory to an authorized local
bundle. Set `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` to its directory. The default
is `~/.cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256/<bundle_sha256>`.
The bundle SHA-256 is
`ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63`.
Public asset access and redistribution permission must be resolved before an
upstream public PR can be considered ready.

### GB300 (presentation only)

The rack is from [NVIDIA SimReady DSX](https://huggingface.co/datasets/nvidia/simready-dsx),
revision `5938869019f0d2afb6b9b808ed1ab1bc6e0e0961`,
`GB300/simready_usd/payloads/external.usd` (CC-BY-4.0).
Set `ISAACLAB_SIMREADY_DSX_GB300_ROOT` to the file or its containing directory.
The default is `~/.cache/isaaclab/simready-dsx/sha256/<file_sha256>/external.usd`.
The SHA-256 is
`5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb`.
Training does not load this 473 MB visual payload.

### RJ45 CAD

The connector uses the existing `newton/examples/assets/rj45_plug.usd` bundled
with Newton, not a duplicate task asset. `ISAACLAB_RJ45_PLUG_USD` may point to
the same canonical USD when using an installation without example assets.
