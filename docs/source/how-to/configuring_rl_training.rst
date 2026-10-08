:orphan:

.. _tutorial-configure-rl-training:

Configuring an RL Agent
=======================

.. currentmodule:: isaaclab

In the previous tutorial, we saw how to train an RL agent to solve the cartpole balancing task
using the `Stable-Baselines3`_ library. In this tutorial, we will see how to configure the
training process to use different RL libraries and different training algorithms.

In the directory ``scripts/reinforcement_learning``, you will find the scripts for
different RL libraries. These are organized into subdirectories named after the library name.
Each subdirectory contains the training and playing scripts for the library.

To configure a learning library with a specific task, you need to create a configuration file
for the learning agent. This configuration file is used to create an instance of the learning agent
and is used to configure the training process. Similar to the environment registration shown in
the :ref:`tutorial-register-rl-env-gym` tutorial, you can register the learning agent with the
``gymnasium.register`` method.

The Code
--------

As an example, we will look at the configuration included for the task ``Isaac-Cartpole``
in the ``isaaclab_tasks`` package. This is the same task that we used in the
:ref:`tutorial-run-rl-training` tutorial.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/__init__.py
   :language: python
   :lines: 50-64

The Code Explained
------------------

Under the attribute ``kwargs``, we can see the configuration for the different learning libraries.
The key is the name of the library and the value is the path to the configuration instance.
This configuration instance can be a string, a class, or an instance of the class.
For example, the value of the key ``"rl_games_cfg_entry_point"`` is a string that points to the
configuration YAML file for the RL-Games library. Meanwhile, the value of the key
``"rsl_rl_cfg_entry_point"`` points to the configuration class for the RSL-RL library.

The pattern used for specifying an agent configuration class follows closely to that used for
specifying the environment configuration entry point. This means that while the following
are equivalent:


.. dropdown:: Specifying the configuration entry point as a string
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_manager_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
         },
      )

.. dropdown:: Specifying the configuration entry point as a class
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_manager_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
         },
      )

The first code block is the preferred way to specify the configuration entry point.
The second code block is equivalent to the first one, but it leads to import of the configuration
class which slows down the import time. This is why we recommend using strings for the configuration
entry point.

The reinforcement learning entrypoints are configured by default to read the
``<library_name>_cfg_entry_point`` from the ``kwargs`` dictionary to retrieve the configuration instance.

For instance, the following code block shows how the Stable-Baselines3 training implementation
reads the configuration instance:

.. dropdown:: Code for train_sb3.py with SB3
    :icon: code

    .. literalinclude:: ../../../source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_sb3.py
      :language: python
      :linenos:
      :emphasize-lines: 56-60, 97-98

The argument ``--rl_library`` selects the reinforcement learning library. The ``--agent``
argument selects the library-specific configuration entry point from the ``kwargs``
dictionary, so you can manually specify alternate configuration instances.

The Code Execution
------------------

Since for the cartpole balancing task, RSL-RL library offers two configuration instances,
we can use the ``--agent`` argument to specify the configuration instance to use.

* Training with the standard PPO configuration:

  .. tab-set::

     .. tab-item:: uv (Recommended)

        .. code-block:: bash

          # standard PPO training
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --run_name ppo

* Training with the PPO configuration with symmetry augmentation:

  .. tab-set::

     .. tab-item:: uv (Recommended)

        .. code-block:: bash

          # PPO training with symmetry augmentation
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --agent rsl_rl_with_symmetry_cfg_entry_point \
            --run_name ppo_with_symmetry_data_augmentation

          # you can use hydra to disable symmetry augmentation but enable mirror loss computation
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --agent rsl_rl_with_symmetry_cfg_entry_point \
            --run_name ppo_without_symmetry_data_augmentation \
            agent.algorithm.symmetry_cfg.use_data_augmentation=false

The ``--run_name`` argument is used to specify the name of the run. This is used to
create a directory for the run in the ``logs/rsl_rl/cartpole`` directory.

Comparing PPO learners on G1
----------------------------

The registered ``Isaac-Velocity-Flat-G1`` task supports RoboLearn's Warp-NN PPO through
the same native training entrypoint:

.. code-block:: bash

   uv sync --extra robolearn
   uv run --no-sync isaaclab train --task Isaac-Velocity-Flat-G1 \
     --rl_library robolearn --algorithm warp_ppo physics=newton_mjwarp

The comparison workflow uses the existing Torch MDP frontend for both learners and
Newton's MuJoCo Warp physics backend. Both use the backend's existing physics CUDA graph;
Warp PPO additionally captures its neural learning update. Observation, reward, command,
reset and rollout assembly remain in the existing Isaac Lab workflow.

``scripts/benchmarks/compare_g1.py`` matches separate 256/128/128 Tanh actor and critic networks,
a fixed learning rate of ``3e-4``, five full-batch epochs, a 24-step rollout horizon and action
clipping to ``[-1, 1]``. Its RSL-RL run overrides the stock G1 agent's ELU activation, adaptive
learning rate and four minibatches. The stock agent configuration remains available unchanged.
Network initialization, timeout bootstrapping and gradient clipping still differ between learners.

For a paired comparison, use identical iteration and environment counts. The defaults collect
36,864,000 transitions per learner: 1024 environments times 24 steps times 1500 iterations.
With two GPUs, run each seed's two commands in separate terminals, then swap GPUs for the next seed:

.. code-block:: bash

   uv run --no-sync python scripts/benchmarks/compare_g1.py train \
     --algorithm warp_ppo --seed 0 --device cuda:0 --output logs/g1-warp-seed0
   uv run --no-sync python scripts/benchmarks/compare_g1.py train \
     --algorithm rsl_rl_ppo --seed 0 --device cuda:1 --output logs/g1-rsl-seed0
   uv run --no-sync python scripts/benchmarks/compare_g1.py train \
     --algorithm warp_ppo --seed 1 --device cuda:1 --output logs/g1-warp-seed1
   uv run --no-sync python scripts/benchmarks/compare_g1.py train \
     --algorithm rsl_rl_ppo --seed 1 --device cuda:0 --output logs/g1-rsl-seed1

After training finishes, evaluate the saved checkpoints using deterministic actions and common
reset seeds. Evaluation disables observation corruption and random pushes for both learners and
measures both the native command distribution and a fixed 0.5 m/s forward-walking scenario:

.. code-block:: bash

   for g1_run_dir in logs/g1-warp-seed0 logs/g1-rsl-seed0 logs/g1-warp-seed1 logs/g1-rsl-seed1; do
     uv run --no-sync python scripts/benchmarks/compare_g1.py evaluate \
       --output "$g1_run_dir" --device cuda:0 --checkpoint_iterations 500 1000 1500
   done
   uv run --no-sync python scripts/benchmarks/compare_g1.py merge \
     --output logs/g1-comparison \
     --inputs logs/g1-warp-seed0 logs/g1-rsl-seed0 logs/g1-warp-seed1 logs/g1-rsl-seed1
   uv run --no-sync python scripts/benchmarks/render_g1_comparison.py \
     logs/g1-comparison/comparison.json logs/g1-comparison/comparison.html

The merge phase requires paired seeds, matching MDP fingerprints, source revisions, dependency
versions and actual collection budgets. The offline HTML report separates rollout and learning
time, shows return against time or samples, and includes survival, velocity tracking and forward
walking metrics. Select algorithms, seeds, GPUs, evaluation scenarios and checkpoints interactively.
Use the measured quality results to interpret speed differences; two seeds provide an exploratory
comparison rather than a general algorithm ranking.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
