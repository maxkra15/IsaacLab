* Added opt-in CUDA graph capture for analytic Newton IK solves, preserving live targets and input seeds
  while reducing per-iteration launch overhead.
* Added opt-in ``NewtonCfg.capture_implicit_actuators`` to apply implicit-only action targets once per control step
  during physics-owned decimation, preserving solver drives and effort telemetry.
