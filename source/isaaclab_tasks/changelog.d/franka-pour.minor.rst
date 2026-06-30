Added
^^^^^

* Added the ``Isaac-Pour-Franka-v0`` (``-Play-v0`` / ``-Teleop-v0``) contributed task: a Franka
  grasps a small hollow-cube bowl full of granular MPM media and pours it into a fixed target bowl
  using Newton-generated rigid contacts resolved by a proxy-coupled MJWarp and implicit-MPM
  solver, with isolated per-environment sparse grids, CUDA graph replay, capacity scaling, and
  selective reset.

Changed
^^^^^^^

* Changed Franka Pour cups and media to scene-owned assets, kept solver-only collision proxies in
  the per-world Newton builder hook, and added selective public-state reset plus two-world Kit and
  Newton visualization coverage. Registered task users need no migration; custom scripts should
  access the cups through ``env.scene["source_cup"]`` and ``env.scene["target_cup"]``.
* Changed the Franka Pour RSL-RL runner to log to the ``franka-pour-mpm`` W&B project by default;
  set ``WANDB_MODE=offline`` to retain runs locally without syncing.
