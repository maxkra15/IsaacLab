:orphan:

.. _run-scripted-state-machines:

Run Scripted State Machines
===========================

Isaac Lab includes hand-written state-machine examples for inspecting an
environment's observations and action interface without training a policy. The
first three examples below run transitions in parallel as Warp kernels, which
keeps them efficient at larger environment counts. The contributed showcases
use measured Torch state machines instead.

Run these commands from the Isaac Lab repository root. Use ``--num_envs`` to
change the number of parallel environments and ``--viz`` to select a
visualizer.

Pick and lift a rigid cube
--------------------------

This example approaches a cube, closes the gripper, and lifts the cube to its
goal pose:

.. code-block:: bash

   uv run python scripts/environments/state_machine/lift_cube_sm.py \
      --num_envs 32 --viz kit

Lift a deformable object
------------------------

This example uses the Newton backend to grasp and lift a soft object. The
Newton visualizer opens by default:

.. code-block:: bash

   uv run --extra tetrahedralization python scripts/environments/state_machine/lift_franka_soft.py \
      --num_envs 1

Open a cabinet drawer
---------------------

This example approaches the drawer handle, grasps it, pulls the drawer open,
and releases it:

.. code-block:: bash

   uv run python scripts/environments/state_machine/open_cabinet_sm.py \
      --num_envs 32 --viz kit

Each script defines its states, wait times, transition kernel, and action loop
in one file. Start with ``lift_cube_sm.py`` when adapting the pattern to a new
manipulation task.

Coupled-solver showcases
------------------------

Three contributed manager-based PPO tasks also have finite scripted demos. Their
state machines advance from measured robot and object state; the scripts are
playback controllers, not trained policies. The scenes retain the default
ground plane, use a small visual-only USD backdrop, and report physical
progress at the end of each run.

.. code-block:: bash

   # MJWarp + MPM: Franka pours particles between two cups.
   uv run --extra video python scripts/environments/state_machine/kinetic_foundry.py \
      --max_steps 1800 --video

   # MJWarp + VBD: KUKA presses a cloth sheet while Fourier GR1T2 presents its far edge.
   uv run --extra video python scripts/environments/state_machine/textile_atelier.py \
      --max_steps 800 --video

   # MJWarp rigid contact: KUKA-Allegro launches a ball for a GR1T2 open-hand deflection.
   uv run --extra video python scripts/environments/state_machine/relay_juggle.py \
      --max_steps 720 --video

Use ``--viz none`` without ``--video`` to check physics and phase metrics first.
The demos use one environment by default; each registered task retains its
separate PPO action configuration for future training.
