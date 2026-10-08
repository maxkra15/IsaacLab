# GB300 Ethernet cable insertion

`IsaacContrib-Rizon-Sharpa-GB300-Cable-Install` is a Newton manager-based task
with a fixed-base Rizon 4s and Sharpa right hand. A short cable starts grasped;
its other connector is fixed in the upper switch port. The objective is to
insert, latch, and release the loose connector in the lower port. Gravity is
active. The example additionally regrasps, depresses the clip, and extracts the
connector in one uninterrupted episode.

## Implementation

The task reuses `ManagerBasedRLEnv`, `InteractiveScene`, the standard scene-reset
event, observation/reward/termination managers, `CableObject`, Newton IK, joint
position actions, solver/coupler configurations, and RSL-RL PPO. Task-specific
code authors the connector, spring clip, sockets, and measured cable boundary.
The example's state machine produces normal actions, without robot teleports,
grasp attachments, connector gravity cancellation, or a reset at the handoff.

`presentation.py` composes optional, visual-only USD scenery: eight instances of
the original GB300 CAD, the surrounding data center, installed colored leads,
and manager-driven status lights. It adds no background bodies or collisions.
Recording uses `VideoRecorderCfg`; promo compositing and HUD design remain
internal tools, outside the task and controller.

## Run

```bash
uv run isaaclab demo rizon-sharpa-rj45
uv run --extra video isaaclab example gb300-sharpa-insertion \
  --viz newton_gl --video --output_dir logs/demos/gb300_sharpa
uv run --extra video isaaclab example gb300-sharpa-insertion \
  --viz none --video --output_dir logs/demos/gb300_sharpa_headless
uv run --extra isaacsim --extra video isaaclab example gb300-sharpa-insertion \
  --viz kit --video --output_dir logs/demos/gb300_sharpa_kit
```

The scripted example requires the assets and corrected Newton checkout below.
Use `--no_presentation --viz none` for physics-only checks. `--max_steps`
(`--num_steps` is an alias) bounds the episode. `--settle_duration_s 2` adds an
initial grasp hold. Video filenames are unique.

Measured grasp slip, seating, clip clearance, hands-off retention, and withdrawal
guards are always active. `--validate_contacts` additionally enables slower CPU
contact reconstruction. `--failure_tail_s 8` continues simulation and recording
for eight seconds after a controller fault while holding the last targets; it
does not bypass guards or report success. Non-finite state or a reset ends the
run. The final JSON reports completion, failure, reset count, and withdrawal.

## RL action contract

The default preset has 28 normalized actions: six incremental palm commands and
22 reset-relative finger targets. State observations contain joints, connector
and goal poses, clip state, cable positions, and previous actions. Success
requires measured seating, latch capture, and palm withdrawal, not a prescribed
finger posture. It uses native VBD/proxy coupling without hydroelastic contacts.

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Rizon-Sharpa-GB300-Cable-Install \
  --num_envs 32 --viz none
```

This is a PPO baseline, not a validated trained policy. Check contact capacity
and GPU memory before increasing the environment count. Camera observations,
domain randomization, and extraction rewards are not included.

The example selects `presets=scripted`: 29 absolute actions (palm position,
XYZW quaternion, and 22 finger targets), hydroelastic contacts, and a one-way
cable boundary. Its actions and physics differ from training; checkpoints are
not interchangeable. `--one_way_cable` remains accepted for existing commands.
The configuration also retains the bidirectional ADMM variant via
`GB300SharpaScriptedEnvCfg(one_way_cable=False)`.

## Physical configuration

The active cabinet uses a simple collision shell. Both detailed socket SDFs
are registered to the pinned SN2201 CAD, including the upper socket's measured
0.26 mm lateral correction. The socket mesh closes native open surfaces and
retains rear guides and the lower entry support at 0.05 mm SDF resolution.
Electrical contacts and the deeper rear step remain simplified. Contact search
gaps do not move the visual or collision frames.

The loose plug is retained by its physical spring clip, not a weld. Its
CAD-calibrated compression stop is −0.225 rad, retaining approximately 65 µm
of housing clearance at the free press beam and at least 120 µm of keeper
clearance at the nominal seated pose. The supported hinge root is excluded
from the housing-clearance measurement.

In the scripted preset, the robot, housing, and clip share MJWarp. Hydroelastic
interfaces are thumb/index/middle–housing and thumb–clip. Ring and pinky remain
visible and actuated but collision-disabled; training retains all five fingers.
Fixed fingertip pads inherit authored adjacent-link exclusions without disabling
cross-finger contact. Ordinary contacts use cooked fingertip distance fields
where supported; mesh/capsule pairs can still use triangle queries.

The cable remains a VBD rod with gravity, self-contact, and contact against the
hand, cabinet, sockets, pedestal, and ground. The lower terminal follows the
**measured** housing pose and COM velocity, including its offset's angular
velocity contribution. The upper terminal is kinematic; its seated plug is
visual-only, with no SDF or hydroelastic solve. Cable forces do not feed back
to the loose plug or robot in this preset.

Endpoint updates use the existing proxy lifecycle inside CUDA capture. The
rigid pipeline uses its solver-owned pair list. The cable pipeline detects
contacts after rigid pose transfer and excludes immovable–immovable pairs once
at initialization. It has no redundant hydroelastic field. Source body flags
and mass properties are unchanged. Cable diagnostics use
`get_proxy_contacts("rigid", "cable")`, not the main rigid contact buffer.

Release holds the seated palm pose while the thumb opens upward and the index
and middle fingers open downward. The extraction grasp retargets once to the
measured seated plug, without an axial retreat. Supporting fingers settle
before the thumb presses the clip; a firmer pinch follows measured clip release.
Lost clearance pauses withdrawal at the measured pose for bounded recovery.

The scripted solver uses 120 Hz physics and 30 Hz control, four outer substeps,
two inner MJWarp steps, and two VBD subdivisions with eight iterations each.
Rigid geometry refreshes at 480 Hz, preserving 1/960 s clip and cable integration.
Cable bending/torsion moduli are 1e8/6e7 Pa. Finger armature is 2e-4 kg·m²;
ordinary housing/keeper stiffness is 1e5/1e6 N/m. These are numerical tuning,
not calibrated material or motor measurements. The normalized preset is unchanged.
Zero substeps do not advance physics; coarser rigid timesteps soften clip contact
through MJWarp's `refsafe` clamp and must be validated through a complete cycle.

Physics, implicit actuator decimation, and Newton IK are CUDA-captured. The
state machine, manager orchestration, and one compact feedback readback remain
on the CPU. Rendering and optional contact diagnostics are outside capture.

## Assets and Newton dependency

Asset payloads are not committed or downloaded automatically. Asset resolution
validates pinned local files and reports missing dependencies.

The manufacturers publish the [Rizon arm sources](https://github.com/flexivrobotics/flexiv_description)
and [Sharpa hand sources](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
under Apache-2.0. The Rizon arm also has an existing Isaac Lab asset configuration,
`FLEXIV_RIZON4S_CFG`. These public components are not a drop-in replacement for
the validated combined assembly below: the hand's USD palm mass and pad meshes
differ, and its palm lacks `r_palm_ctrl`. A public composition can preserve the
mount and TCP using a fixed joint and the existing Newton IK body-offset fields;
its grasp and contact behavior still need validation before replacing the default.

- **Robot:** the access-controlled [Fabrics-Sim repository](https://gitlab-master.nvidia.com/dex/fabrics-sim.git),
  revision `d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`, at
  `src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres/rizon4s_sharpa_no_spheres_generated.usd`.
  Copy the USD and sibling `textures/` into an authorized local bundle and set
  `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`. Its bundle SHA-256 is
  `ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63`.
  The default root is `~/.cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256/<bundle_sha256>`.
- **GB300, presentation only:** [NVIDIA SimReady DSX](https://huggingface.co/datasets/nvidia/simready-dsx),
  revision `5938869019f0d2afb6b9b808ed1ab1bc6e0e0961`,
  `GB300/simready_usd/payloads/external.usd`, CC-BY-4.0. Set
  `ISAACLAB_SIMREADY_DSX_GB300_ROOT` to the file or containing directory.
  Its SHA-256 is `5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb`;
  the default is `~/.cache/isaaclab/simready-dsx/sha256/<file_sha256>/external.usd`.
  Training does not load this 473 MB payload.
- **Connector:** reuse `newton/examples/assets/rj45_plug.usd`, not a task copy.
  `ISAACLAB_RJ45_PLUG_USD` can supply the canonical file when example assets are
  absent from the installed Newton package.

Two contact-coupling defects remain in the pinned public Newton dependency:
MJWarp hydroelastic gains omit effective-mass conversion, and filtered contacts
can publish a generation before invalidating its cached mapping. Focused
regressions fail on the public dependency and pass with these corrections. The
mapping race is scheduling-dependent; it was reproduced on CPU, not in the
current L40 fixture. Prefix example commands with
`PYTHONPATH=/path/to/corrected-newton`; the task does not patch installed packages
or change the dependency pin. Native training does not use hydroelastic gains,
but mapping correctness affects coupled solvers generally.

Subset-aware hydroelastic allocation is not an external prerequisite: the cable
pipeline uses Newton's public solver-view pair override. A graph-safe SDF warning
poll is a separate, conditional capture issue, not an established cause of the
current extraction timeout. IK and implicit-actuator capture are opt-in
performance additions. The Kit camera fix supports headless moving-camera
recording, and the launcher fix enables sibling imports through the public CLI;
neither changes insertion contacts. These framework changes should be reviewed
separately from the task, not described as physics prerequisites.

## Validation and public PR status

On an idle L40, the three-finger physics-only example completed one continuous
cycle at 15.2 control steps/s (0.51 real-time factor), excluding startup and
diagnostic sampling. It had 79 active colliders, 925 rigid candidate pairs, and
1,499 cable pairs, with zero resets, non-finite states, fixed-tail drift, or
reported overflow. The largest sampled cable-joint gap/socket overlap was
0.090/0.174 mm. This is not real-time performance or a robustness claim.

Two subsequent cycles completed insertion, hands-off retention, regrasp,
physical clip depression, extraction, and final hold after making thumb feedback
during the housing pinch as responsive as during pressing and withdrawal.
Measured withdrawal was 59.80/59.74 mm, with maximum sampled cable-joint gaps
of 0.050/0.055 mm and no non-finite state or fixed-tail drift. Grasp targets,
physical clip stops, and release guards were unchanged. Two passes are not a
broad robustness or public-asset validation claim. A dynamically validated public
robot composition and upstream contact corrections remain outstanding for a
self-contained public example.
