# GB300 Ethernet cable insertion

`IsaacContrib-Rizon-Sharpa-GB300-Cable-Install` is a Newton manager-based task
with a fixed-base Rizon 4s and Sharpa right hand. A short cable starts grasped;
its other connector is fixed in the upper switch port. The task is to insert,
latch, and release the loose connector in the lower port. The example then
regrasps, depresses the clip, and extracts it in one uninterrupted episode.

## Run

Install the assets and select the corrected Newton checkout described below,
then run:

```bash
uv run isaaclab demo rizon-sharpa-rj45
uv run --extra video isaaclab example gb300-sharpa-insertion \
  --viz newton_gl --video --output_dir logs/demos/gb300_sharpa
uv run --extra isaacsim --extra video isaaclab example gb300-sharpa-insertion \
  --viz kit --video --output_dir logs/demos/gb300_sharpa_kit
```

Use `--no_presentation --viz none` for physics-only checks or `--viz none --video`
for headless recording. `--max_steps` bounds the episode; `--settle_duration_s`
adds an initial grasp hold. `--failure_tail_s 8` continues recording for eight
seconds after a controller fault while holding the last targets. It does not
bypass guards or report success; non-finite state or a reset ends the run.
Measured seating, slip, clip clearance, retention, and withdrawal guards are
always active. The default handoff checks actual thumb/index/middle joint
opening within 0.03 rad of the calibrated open posture; this is not a zero-force
contact test. `--validate_contacts` adds slower CPU contact diagnostics.
The final JSON reports completion, failure, reset count, and withdrawal.

## Task and example

The task reuses `ManagerBasedRLEnv`, `CableObject`, Newton IK, joint-position
actions, standard reset events, observation/reward/termination managers, and
RSL-RL PPO. Task-specific code authors the sockets, connector and spring clip,
and the measured cable boundary. The example returns normal actions: no robot
teleports, grasp attachment, gravity cancellation, or reset at the handoff.

The default preset has 28 normalized actions: six incremental palm commands
and 22 reset-relative finger targets. State observations contain joints,
connector/goal poses, clip state, cable positions, and previous actions. Success
requires measured seating, latch capture, and palm withdrawal. This preset uses
native VBD/proxy coupling without hydroelastic contacts.

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Rizon-Sharpa-GB300-Cable-Install --num_envs 32 --viz none
```

This is a PPO baseline, not a validated learned policy. Check contact capacity
and GPU memory before scaling. Camera observations, randomization, and
extraction rewards are not included. The example selects `presets=scripted`:
29 absolute actions (palm position, XYZW quaternion, 22 finger targets),
hydroelastic contacts, and a one-way cable boundary. Training and scripted
checkpoints are not interchangeable. The experimental bidirectional ADMM
configuration, `GB300SharpaScriptedEnvCfg(one_way_cable=False)`, has not been
validated on the current dependency stack.

## Physics and presentation

The active cabinet has a simple collision shell and two CAD-registered socket
SDFs at 0.05 mm resolution. The loose plug is retained by its physical spring
clip, not a weld. Its compression stop is −0.225 rad. Scripted hydroelastic
pairs are thumb/index/middle–housing and thumb–clip; ring/pinky stay visible and
actuated but collision-disabled. Training retains all five fingers.

The cable is a VBD rod with gravity, self-contact, and cabinet/hand/socket/
pedestal/ground contact. Its lower terminal follows the measured housing pose
and COM twist, without force feedback to the housing or robot in the scripted
preset. The upper terminal is kinematic; its seated plug is visual-only.
Cable contact filtering uses a solver view without changing source-body flags
or mass properties.

The scripted configuration uses 120 Hz physics, 30 Hz control, four outer
substeps, two inner MJWarp steps, and two VBD subdivisions with eight iterations.
Bending/torsion moduli are 1e8/6e7 Pa; these and the contact/actuator settings are
numerical tuning, not calibrated measurements. Physics and IK are CUDA-captured;
the state machine and managers run on the CPU, with one compact GPU feedback
readback per control step.

Optional `presentation.py` adds eight instanced GB300 racks, the surrounding
data center, installed colored cables, and status lights, without background
physics. Video uses `VideoRecorderCfg`; promo/HUD tools remain outside the task.
Indicators cover the measured CAD light-pipe faces. The target starts dark,
blinks after measured seating and clip capture, and goes dark on withdrawal.
Pressing the clip while the plug remains seated does not extinguish the link.

## Dependencies and public availability

The task uses the dependency versions selected by upstream `develop` in the
repository's `pyproject.toml` and `uv.lock`; it does not override the shared
Newton, Warp, or MuJoCo pins. Run `uv sync` after switching branches.

Assets are not committed or downloaded automatically. Local resolution validates
their pinned digests and reports missing dependencies:

- **Robot:** authorized [Fabrics-Sim](https://gitlab-master.nvidia.com/dex/fabrics-sim.git)
  revision `d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`,
  `src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres/rizon4s_sharpa_no_spheres_generated.usd`.
  Copy the USD and sibling `textures/`; set `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`.
  Bundle SHA-256: `ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63`.
- **GB300, optional visual:** [NVIDIA SimReady DSX](https://huggingface.co/datasets/nvidia/simready-dsx),
  revision `5938869019f0d2afb6b9b808ed1ab1bc6e0e0961`,
  `GB300/simready_usd/payloads/external.usd`, CC-BY-4.0.
  Set `ISAACLAB_SIMREADY_DSX_GB300_ROOT` to the file or containing directory.
  File SHA-256: `5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb`.
- **Connector:** `newton/examples/assets/rj45_plug.usd`.
  Set `ISAACLAB_RJ45_PLUG_USD` if the installed package omits example assets.

Public [Rizon](https://github.com/flexivrobotics/flexiv_description) and
[Sharpa](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) components are
Apache-2.0; Isaac Lab also has `FLEXIV_RIZON4S_CFG`. Their combined assembly is
not yet dynamically validated: palm mass, pad meshes, and TCP binding differ.
The access-controlled default therefore remains a public-PR blocker.

Two external Newton corrections are required for the scripted preset:
effective-mass conversion of MJWarp hydroelastic gains, and invalidation of
cached filtered-contact mappings before publishing their new generation.
Prefix commands with `PYTHONPATH=/path/to/corrected-newton`; this task never
patches installed packages. The mapping race was reproduced on CPU, not in
the current GPU fixture. Training does not use hydroelastic gains but still
depends on correct coupling maps. No private allocation or SDF-warning patch
is required by the validated run.

Two full insertion/extraction cycles passed on an A40 with an earlier corrected
Newton stack, without changing grasp targets or relaxing guards. Those runs do
not validate the current upstream dependency stack or establish real-time
performance or broad robustness. Public robot validation, upstream Newton
corrections, and a full-cycle rerun on the declared dependencies remain
prerequisites for a self-contained public example.
