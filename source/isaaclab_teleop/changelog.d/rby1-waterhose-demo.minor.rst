Added
^^^^^

* Added :func:`~isaaclab_teleop.preload_cloudxr_websockets` to load one
  compatible WebSockets installation before Isaac Sim starts.
* Added CloudXR GPU selection that preserves Kit's primary Vulkan device and
  supports numeric ``CUDA_VISIBLE_DEVICES`` remapping.

Fixed
^^^^^

* Fixed teleoperation actions being emitted in the world frame while a
  configured target frame is temporarily unavailable; actions now resume only
  after the frame resolves.
