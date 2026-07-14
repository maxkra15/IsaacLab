Fixed
^^^^^

* Updated the waterhose task for current Newton contact-history allocation and
  clarified why full-surface rigid-soft contacts do not apply to its rod model.
* Restored clean authored colors in the scripted waterhose demo.
* Paced visible scripted demos to simulation time by default so the robot motion
  remains observable, with ``--no-realtime`` available for benchmarks.
* Moved the shared Kit/Newton camera to a side view so the fridge no longer
  occludes the robot, and authored cable-curve normals required by Fabric.
* Fixed RBY1 wheel visualization after Newton fixed-joint collapsing and added
  bimanual Apple Vision Pro control for both complete arm chains.
* Calibrated each tracked AVP wrist to the current RBY1 gripper-base wrist,
  preserving one-to-one pose deltas without startup or reacquisition jumps.
* Restored the gripper proxy's authored finger inertia after the coupled-solver
  refresh and normalized task-local VBD contact units while retaining the
  validated connector and cable-rod contact pairs for all four left/right
  gripper fingers.
* Separated robot/fridge MJWarp response from the VBD grasp material, using a
  critically damped raw MuJoCo contact response and refreshing collision
  manifolds on every solver substep to prevent deep, bouncy penetration.
* Required measured connector retention through insertion, release, backoff,
  and the runner's final linger instead of allowing timeout-only success.
* Warmed Kit's RTX-deferred Newton graph once, reset the full scene, and then
  captured the scripted state machine and IK so visible runs use both the
  controller and physics CUDA graphs.
