Fixed
^^^^^

* Kept Newton waterhose XR launches on CUDA unless the user explicitly selects a
  device, aligned the separate CloudXR Vulkan runtime with that same physical GPU,
  defaulted Kit rendering to one GPU to avoid hybrid-GPU interop/shutdown failures,
  and preserved the early CloudXR WebSockets preload. Set
  ``WATERHOSE_KIT_MULTI_GPU=1`` to opt into multi-GPU rendering.
* Added bounded/debug teleop runs, target-frame recovery diagnostics, and task-
  specific recorder configuration for waterhose Apple Vision Pro workflows.
* Added bimanual Apple Vision Pro tracking with absolute one-to-one wrist
  rotations and complete RBY1 arm-chain control.
* Suppressed actions while a configured target frame is unavailable so an IK
  controller never receives silently misframed world-space wrist poses.
