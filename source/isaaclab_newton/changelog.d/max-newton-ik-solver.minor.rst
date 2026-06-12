Added
^^^^^

* Added :class:`~isaaclab_newton.ik.NewtonIKSolver` and
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonInverseKinematicsAction`
  for Newton-backed inverse kinematics. IK problems are configured as an
  ordered list of :class:`~isaaclab_newton.ik.NewtonIKObjectiveCfg` entries
  (pose objectives drive action dimensions, constraint objectives such as
  :class:`~isaaclab_newton.ik.NewtonIKJointLimitObjectiveCfg` add residuals
  only), and the single-env prototype model is resolved through the cloner's
  retained clone-plan builders.
