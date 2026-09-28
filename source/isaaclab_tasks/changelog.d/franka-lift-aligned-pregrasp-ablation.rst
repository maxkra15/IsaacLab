Changed
^^^^^^^

* Retained aligned Franka Lift pre-grasps in the reset bank and applied their finger opening after the
  generic gripper reset, while keeping broad starts and success-based sampling. Re-evaluate existing Lift
  checkpoints because the training reset distribution changed.
* Retained valid held-object Franka Reorient starts without farthest-point thinning. Re-evaluate existing
  Reorient checkpoints because the training reset distribution changed.
