Added
^^^^^

* Added ``IsaacContrib-Franka-RJ45-Pick-Insert``, a distinct six-phase Franka task with
  randomized pickup and socket poses, an extended Newton cable, and strictly validated reset data.
* Added a pick-only persistent, reset-seeded absolute-target controller with robot-scoped MJWarp
  joint gravity compensation, bounded target tracking, and a 7D target-error observation (135D
  actor and 138D asymmetric-critic inputs).
* Added drive-free guarded pickup from the true 0.04 m open posture, preload-preserving staged
  transport, and a canonical proof of at least 60 simulated seconds under task contract 6,
  pick-insert semantics 6, reset-dataset schema 3, and validation-report schema 5. Schema 5 binds
  the immutable one-seed validator, exact source and external-asset snapshots, independently
  recomputed diversity evidence, and atomic full-replay checkpoint/resume. The production
  grasp uses a 0.0 m close target and a pick-only raw proxy friction of 4.5 (effective finger/proxy
  friction 3.0). Its post-settle training gates allow a 0.10 plug velocity norm, 0.10 rad relative-latch
  angle, and 0.012 m whole-task drift while retaining the exact insertion geometry, cable-speed,
  contact, collision, and controller gates. The legacy task retains its original material and success
  limits. Pick reset recovery uses phase-aware incremental Cartesian routes with a gradual,
  step-bounded actuator-preload transition from each stored start target to the exact canonical
  target. Each initial route stops at the canonical goal before any bounded, error-triggered
  compensation instead of crossing the seated pose toward proactive overtravel; legacy scripted
  recovery remains unchanged. Pick reset artifacts persist both ordered
  VBD pose-history buffers. The pick task stages the latest reset,
  queues both histories immediately before stepping, and applies each exactly once through the
  public named-coupler API after normal input/proxy rebaselining and before the first VBD solve;
  legacy insertion reset behavior remains unchanged.
* Added pick-only direct-VBD cable input preservation and post-solver, pre-collision capsule
  alignment so authored anchor and alignment poses are not interpreted as external teleports;
  the pinned terminal cable body is excluded and legacy insertion remains unchanged.
* Added a content-addressed, byte-verified 19-file Franka and Seattle-table closure. Production
  pickup startup now requires ``ISAACLAB_FRANKA_RJ45_ASSET_CLOSURE_ROOT`` and binds only verified
  local entrypoints, while artifacts retain normalized logical URIs and ``external_assets``
  contract v1 instead of host-local cache paths.
