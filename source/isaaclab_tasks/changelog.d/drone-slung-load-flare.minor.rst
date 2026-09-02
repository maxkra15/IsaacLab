Added
^^^^^

* Added ``IsaacContrib-DroneSlungLoad-Waypoint-FLARE``, a manager-based Newton
  task for a 305 g quadrotor carrying a 70 g payload on a 0.5 m AVBD cable.
  The actor observations, action scaling, reward coefficients, 100 Hz control
  cadence, and waypoint lookahead follow FLARE (arXiv:2508.09797v1).

* Modeled the cable as a thin, nearly inextensible string with zero bending and
  twisting stiffness, hard ball attachments, eight Newton substeps, and thirty-two
  VBD iterations. Its 0.5 m rest length connects the modeled drone and point-payload
  centers of mass exactly, and fold-safe capsule-endpoint diagnostics guard every
  attachment and internal joint. The actor receives only deployable FLARE
  observations while an asymmetric critic may consume cable state during training.
  Replicated worlds share a precision-centered numerical origin, with the ground
  shifted instead of changing the physical 1.5 m flight clearance.

* Applied nonnegative thrust and yaw reaction torque at each physical rotor site
  on the drone's sole rigid body. The cable and payload receive no commanded
  wrench and move only through gravity, contact, and attachment constraints. A
  collective-priority bounded allocator preserves the requested thrust while
  desaturating infeasible moments without independent-clip distortion.

* Added a solver-clean single-body drone with explicit mass, inertia, rotor
  locations, and yaw authority aligned with the later FLARE scenario-one asset.
  It uses the installed Newton Crazyflie example as a visual-only mesh when
  available, with a procedural Newton-GL-compatible fallback.

* Added terminal-sample-safe episode metrics and a headless checkpoint evaluator
  that rejects candidates with crashes, illegal states/actions, workspace exits,
  cable separation, joint gaps, non-finite metrics, or excessive peak swing. The
  seeded evaluation preserves the training horizon and initial-swing distribution
  while measuring the paper-inspired three-lap figure-eight benchmark. Interactive
  play renders the same route with progress-colored waypoint and segment markers.

* Added ``IsaacContrib-DroneSlungLoad-Waypoint-FLARE-Enhanced`` for stable,
  all-heading route tracking. It preserves the paper-aligned baseline while
  adding randomized bounded ellipse/figure-eight spline routes, indexed spline projection and
  preview observations, analytic cable-angle-rate actor observations, continuous
  swing/transverse-speed/body-rate/action-acceleration costs, a robust log tracking
  cost, and segment-length-calibrated passive AVBD cable bending damping. A staged
  curriculum first tightens waypoint acceptance, tracking scale, and route
  corridor, then raises a shared target speed only after a fixed precision hold.
  Training and evaluation use the same completable one-lap distribution. Indexed
  knot-plane traversal prevents a small waypoint miss from freezing the route,
  while strict-radius precision hits remain separately measurable. A shared
  curvature- and braking-aware speed profile, explicit signed speed error,
  one-shot early-completion reward, successful completion termination, and a
  separate unsafe terminal cost remove the prior sprint-and-exit optimum.
  Its enhanced-only geometric controller adds bounded 3D path-velocity tracking,
  cross-track convergence, curvature feedforward, suspended-mass-aware thrust
  feedback, and finite tilt compensation while leaving the baseline action
  mapping unchanged. Its policy
  rate action uses a smooth residual envelope before adding that prior and then
  clamps the complete command to FLARE's published body-rate envelope. The
  enhanced PPO profile uses a directly learnable, hover-centered tanh-Gaussian
  policy, low physical body-rate exploration, and the later FLARE release's lower
  entropy pressure. It logs detached per-axis latent, normalized-action, and SI
  command exploration scales plus explicitly labeled base-Normal entropy. The
  published baseline retains its bounded Beta policy. Enhanced checkpoints also
  restore the exact environment control step and apply the matching curriculum
  phase before the first resumed observation or action.

* Added ``IsaacContrib-DroneSlungLoad-Waypoint-FLARE-DirectCTBR``, which keeps
  the enhanced ellipse/figure-eight geometry but makes the learned policy own
  the complete collective-thrust/body-rate command. Its only engineered flight
  layer is a conventional measured-rate PID, rate-priority rotor mixer, and
  motor lag. The actor now observes body rates directly, and a dedicated speed
  curriculum targets precise 3.5 m/s flight without loading residual-controller
  checkpoints under incompatible action semantics. Its long-horizon PPO profile
  collects 500 control steps per rollout with ``gamma=lambda=0.999`` so one
  rollout spans the observed three-to-five-second failure window. Four hundred
  updates cover exactly 200,000 control steps. The curriculum reaches its final
  stage at completed update 360; learning-rate and entropy schedules anchored at
  update 359 apply their first 1/40 decay for training update 361 and their floors
  for training update 400.
