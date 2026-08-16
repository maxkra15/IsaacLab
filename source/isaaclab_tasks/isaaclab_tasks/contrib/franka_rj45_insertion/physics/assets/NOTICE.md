# Newton RJ45 asset attribution

`rj45_plug.usd` is an unmodified copy of
`newton/examples/assets/rj45_plug.usd` from the
[Newton Physics](https://github.com/newton-physics/newton) repository at
commit `7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a`.

- Copyright (c) 2026 The Newton Developers.
- SPDX license identifier: `Apache-2.0`.
- Source SHA-256: `50c95bcfb63544777f9148d548aac6f16b62f65cacbaaa9316453d579de4b4fa`.
- A copy of the Apache License 2.0 distributed with Newton is retained in this
  repository at `docs/licenses/dependencies/newton-license.txt`.

The task-local Python implementation and its connector/cable kernels were
adapted from Newton's Apache-2.0 RJ45 example to batched Isaac Lab worlds; the
USD asset itself was not modified. The official socket, plug, latch, and cable
remain the insertion/contact geometry, and the example's infinite support plane
remains at assembly-local z=0 so its cable boundary follows each translated
environment. Isaac Lab adds one invisible, zero-density box on the plug's rear
housing to make reset-generated grasps reproducible. Coupled Franka hand/finger
proxies are filtered from the official connector, cable, and support geometry,
and the hand is filtered from the box, leaving a single intentional task
contact interface between the two fingers and that grasp box.
