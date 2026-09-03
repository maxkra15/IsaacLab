Added
^^^^^

* Added ``IsaacContrib-Franka-RJ45-Dual-Rack-Insert``, a trainable two-ended
  cable task with two physical RJ45 sockets and plugs, one permanently seated
  cable end, two detailed AS4610 rack presentations, and a collision-enabled
  T-slot workcell with an open robot service corridor.
* Added role-explicit anchored-end observations and termination, a distinct
  152D actor / 155D critic checkpoint interface, and a source-bound reset-bank
  generator that shares the six-phase pick-and-insert curriculum while routing
  one exact-length cable between its movable and fixed ends.
* Added ``IsaacContrib-Franka-RJ45-GB300-Insert`` with the pinned NVIDIA
  SimReady GB300 exterior, eight target sockets registered to actual jacks on
  its modeled 48-port SN2201 switch, one randomly selected hidden exact RJ45
  SDF per reset, one occupied physical cable end, and a recessed cuboid
  collision approximation. The CC-BY-4.0 CAD remains optional and render-only;
  no synthetic panel, replacement port visuals, or inactive-port collision
  geometry is added.
* Added a GB300 studio presentation with eight consistently front-facing
  gapless cabinets that share immutable source-payload caches. Corrected the
  active CAD's native front-to-port registration and preserved the SimReady
  root transform below a separate placement prim. Removed the table visual and
  contact slab, placed the glossy floor at the exact cabinet bottom, and added
  a pinned static Rizon4s Sharpa, robot pedestal, three decorative hanging
  cable drops, and white backwall without expanding the training physics
  topology or policy ABI.
