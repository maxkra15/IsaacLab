Added
^^^^^

* Added the standalone ``IsaacContrib-Rizon-Sharpa-Hanging-RJ45-XR-Teleop``
  task with one fixed-base Rizon4s/Sharpa, one exact 0.5 m top-anchored native
  Newton cable, and one dynamic RJ45 plug. MJWarp owns the robot, VBD owns the
  cable, and one staggered right-digit proxy couples the two.
* Added calibrated Apple Vision Pro retargeting: Newton IK follows the XR
  anchor in absolute position and uses an absolute, geometry-derived
  OpenXR-to-canonical-palm orientation on Fabrics-Sim's
  ``r_palm_ctrl`` frame, correcting Sharpa's upstream-documented X/Z palm-axis
  swap and the former 90-degree wrist offset. NVIDIA IsaacTeleop's default
  Sharpa DexPilot path independently drives all 22 right-hand joints from raw
  OpenXR hand tracking, with flexion-only thumb gain and a dedicated stiffer
  thumb actuator for full, stable closure. This replaced the
  former binary pinch action, changing the teleoperation action vector from 8
  to 29 values.
* Made the hand-to-cable proxy explicitly one way: robot poses and contacts
  still drive the VBD cable, but zero feedback relaxation prevents cable
  impulses from disturbing the arm or independently retargeted fingers.
  Expanded the proxy from digits-only to the complete physical right hand,
  scaled its destination mass/inertia by 1000, raised the hanging cable, and
  calibrated the XR origin so the operator's neutral right palm is co-located
  with the simulated home palm.
* Reused the canonical RJ45 asset's embedded cable-to-plug strain-relief datum
  from the Franka task. Made the plug and a 40 mm strain-relief span one rigid
  body, moving the first deformable bend/twist joint behind the housing so the
  cable cannot hinge through or separate from the plug. Added one floating,
  canonical RJ45 socket in front of the first GB300,
  with an exact plug/socket narrow-band SDF contact pair and a socket-pose
  observation. Raised the hanging cable, socket, and desktop camera together
  while keeping the AVP anchor floor-relative.
* Added a weak proximal posture reference that removes redundant-arm null-space
  drift and explicit paused Start/Stop/Reset operation. Removed the former
  GB300/Franka-derived Sharpa teleoperation task and its rack-specific
  cable-routing scaffolding.
* Restored the polished render-only showroom around the standalone task with
  eight front-facing SimReady GB300 cabinets, a glossy white floor and
  backwall, studio lighting, and a lower half-meter robot pedestal. Reduced
  the eager interactive solver profile
  from roughly six control updates per second to a one-substep, four-iteration
  IK path while retaining the same coupled robot/cable ownership.
