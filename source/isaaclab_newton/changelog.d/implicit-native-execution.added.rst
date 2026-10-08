* Added opt-in ``NewtonCfg.capture_implicit_actuators`` for physics-owned decimation of ordinary implicit-only
  actuator collections, preserving solver drives and effort telemetry. The default preserved existing execution;
  enabling the option applied manager-based action targets once per control step instead of once per physics step.
