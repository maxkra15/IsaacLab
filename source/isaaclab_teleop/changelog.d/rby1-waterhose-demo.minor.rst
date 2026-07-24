Added
^^^^^

* Added :func:`~isaaclab_teleop.preload_cloudxr_websockets` to load one
  compatible WebSockets installation before Isaac Sim starts.

Fixed
^^^^^

* Removed automatic CloudXR GPU-index mapping because CUDA ordinals do not
  identify the same physical devices as Vulkan ordinals on hybrid-GPU hosts.
* Fixed teleoperation actions being emitted in the world frame while a
  configured target frame is temporarily unavailable; actions now resume only
  after the frame resolves.
