<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Rizon4s Sharpa hanging-cable teleoperation

`IsaacContrib-Rizon-Sharpa-Hanging-RJ45-XR-Teleop` is a standalone,
single-world Apple Vision Pro teleoperation task. Its physical task contains a
fixed-base Rizon4s with its 22-DoF right Sharpa hand, a compact pedestal, and
one native 0.5 m Newton cable. The cable has one dynamic RJ45 plug at its free
end and one fixed attachment above it. A small exact-length lateral preload
makes the cable swing visibly when simulation starts. The connector mesh is
loaded from the pinned Newton package and verified by content hash.
The plug and the first 40 mm cable span are geometry on the same rigid body:
there is no plug-to-cable joint and no relative connector motion. The first
deformable bend/twist joint starts behind the housing and strain relief rather
than through the middle of the plug. A cyan floating socket in front of the
first GB300 provides an optional insertion target; both plug and socket use the
canonical narrow-band SDF geometry for contact, while a separate invisible box
remains dedicated to robust hand grasping.

Kit presents that clean physical task in a polished data-center showroom:
eight gapless, front-facing SimReady GB300 cabinets, a glossy white floor, a
white backwall, and neutral studio lighting. The cabinets, wall, and extra
lights are authored only after Newton finalizes the coupled model and IK
prototype. They are render-only, so the large CAD payload never changes or
slows the physics topology. The pedestal remains the one visible physical
cuboid that supports the robot.

The robot is solved by MJWarp and the cable by VBD. One directed staggered
proxy mirrors the complete physical right hand—palm and all finger collision
shapes—into the cable entry. Its proxy mass/inertia is scaled by 1000 so the
hand is effectively rigid to the cable. Proxy feedback relaxation is zero:
the hand moves and collides with the cable, while cable forces cannot perturb
the arm or finger joints. The task does
not contain a Franka, table, decorative cables, or duplicated showroom
collision geometry. The interactive profile uses a 60 Hz simulation clock,
one physics substep, and four warm-started Newton IK iterations so tracked hand
targets are applied in real time without changing the ownership boundary.

Right-palm position follows the calibrated XR anchor absolutely, so the robot
palm and tracked-hand markers share one world position. Tracking loss holds
the last target, and reacquisition resumes the current absolute pose. Orientation
is absolute after composing OpenXR's wrist convention, NVIDIA's official
OpenXR-to-Sharpa mapping, and Fabrics-Sim's measured native-palm-to-canonical-
palm fixed joint. Newton IK
targets Fabrics-Sim's authored ``r_palm_ctrl`` frame rather than Sharpa's raw
``right_hand_C_MC`` frame. That upstream control frame corrects Sharpa's native
X/Z palm-axis swap and uses the shared convention X toward the knuckles and Z
out of the palm. Consequently a flat physical palm commands a flat simulated
palm rather than inheriting the robot's arbitrary orientation on acquisition.
Subsequent translation deltas remain one-for-one while analytic IK drives the
seven arm joints.

The XR stage X/Y origin is calibrated so a neutral right hand held about 45 cm
in front of the operator maps onto the simulated hand's home workspace.
CloudXR already reports floor-relative height, so the Z anchor remains at the
studio floor instead of applying an additional downward offset. This keeps the
rendered hand-debug markers and camera at the operator's physical height.

All 22 right-hand joints are independently retargeted from OpenXR's 26 tracked
hand joints with NVIDIA IsaacTeleop's Sharpa ``DexHandRetargeter`` and its
DexPilot configuration. A joint-limit-safe, flexion-only thumb gain restores
full closure without distorting thumb abduction, and a dedicated higher-authority
thumb actuator tracks that heavier five-joint chain. As in NVIDIA's mature Isaac Lab dexterous-hand
pipelines, fingers consume the raw tracked hand—not the world-transformed wrist
stream—and DexPilot owns the exact OpenXR-to-robot basis conversion. The
packaged configuration and official standalone Sharpa URDF are content-pinned;
joint outputs are reordered to the articulation's exact 22-joint contract and
written directly with the authored joint limits.

The session starts paused. Connect the Isaac XR Teleop Sample Client on Apple
Vision Pro, then use the explicit Start, Stop, and Reset controls:

```bash
PYTHONPATH=source/isaaclab_teleop \
uv run --frozen --no-sync isaaclab teleop run \
  --task IsaacContrib-Rizon-Sharpa-Hanging-RJ45-XR-Teleop \
  --num_envs 1 --device cuda:0 --visualizer kit \
  --xr --cloudxr_env avp \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"
```

### External asset downloads

Download the render-only rack from NVIDIA's public
[`GB300 external.usd`](https://huggingface.co/datasets/nvidia/simready-dsx/resolve/5938869019f0d2afb6b9b808ed1ab1bc6e0e0961/GB300/simready_usd/payloads/external.usd?download=true)
at pinned SimReady-DSX revision
`5938869019f0d2afb6b9b808ed1ab1bc6e0e0961`. Place the 473,434,496-byte file
at:

```text
~/.cache/isaaclab/simready-dsx/sha256/5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb/external.usd
```

Download the robot USD and its `textures/` directory from the SSO-gated
[`rizon4s_sharpa_no_spheres` source directory](https://gitlab-master.nvidia.com/dex/fabrics-sim/-/tree/d0dbd1ddaefc4996db546949a7dfb37e39afcbeb/src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres)
at pinned Fabrics-Sim revision
`d0dbd1ddaefc4996db546949a7dfb37e39afcbeb`. Preserve this layout:

```text
~/.cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256/ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63/
|-- rizon4s_sharpa_no_spheres_generated.usd
`-- textures/
```

Alternatively, set `ISAACLAB_SIMREADY_DSX_GB300_ROOT` to the downloaded GB300
file (or its directory) and `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` to the
downloaded robot bundle directory. Startup verifies the pinned sizes and
SHA-256 digests before loading either asset.

The finger-retargeting reference is NVIDIA's public
[`IsaacTeleop`](https://github.com/NVIDIA/IsaacTeleop) repository at commit
`c5fe6624cc4dff456485d2e786922c8e41100f83`. The standalone hand model is from
[`sharpa-urdf-usd-xml`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
at commit `3e953f588ba9954cebaa720aaa4cee06a43a068e`. Both packaged files retain their
upstream Apache-2.0 provenance and are verified before use.
