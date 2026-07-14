Added
^^^^^

* Added a Newton cable asset with attachment support for coupled VBD tasks.

Fixed
^^^^^

* Preserved the option for coupled-solver entries to use their model-view
  mass and inertia instead of solver-specific effective mass.

* Restored VBD rigid-contact capacity/history, AVBD, and joint-constraint
  tuning needed by coupled cable scenes, including penalty-only structural
  joints through Newton's public constraint-mode API.
* Prevented generic forward kinematics from overwriting VBD-owned articulation
  poses in coupled simulations.
* Preserved each builder's configured shape defaults when authoring cable rods,
  instead of silently restoring Newton's standalone contact defaults.
