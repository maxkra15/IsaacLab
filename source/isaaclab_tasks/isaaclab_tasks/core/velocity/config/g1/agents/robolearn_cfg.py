# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboLearn learner recipes for the existing flat G1 locomotion MDP."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.robolearn import RoboLearnRunnerCfg


@configclass
class G1FlatWarpPPORunnerCfg(RoboLearnRunnerCfg):
    """Match the native G1 PPO recipe with captured Warp-NN learner updates."""

    algorithm = "warp_ppo"
    experiment_name = "g1_flat"
    num_steps_per_env = 24
    max_iterations = 1500
    clip_actions = None
    init_at_random_ep_len = True
    algorithm_cfg = {
        "hidden_dims": [256, 128, 128],
        "activation": "elu",
        "std_type": "scalar",
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 1.0,
        "value_loss_scale": 1.0,
        "entropy_coefficient": 0.008,
        "epochs": 5,
        "num_mini_batches": 4,
        "max_grad_norm": 1.0,
        "initial_std": 1.0,
        "normalize_advantages": True,
        "advantage_sample_std": True,
        "timeout_bootstrap": "current",
        "separate_grad_clipping": True,
        "clip_value": True,
        "optimized_linear_backward": False,
    }


@configclass
class G1FlatFlashSACRunnerCfg(RoboLearnRunnerCfg):
    """Use the FlashSAC authors' G1 Isaac Lab collection and learner recipe.

    The upstream ``run_isaaclab.sh`` uses 1024 environments, two replay updates
    per vector step, a ten-million-transition replay buffer and compiled AMP.
    ``ACTION_BOUNDS`` assigns G1 a policy action bound of one. The task retains
    its native PD target scaling. The learning-rate decay budget below is 2050
    iterations of 24 vector steps, or 50,380,800 collected transitions.
    """

    algorithm = "flashsac"
    experiment_name = "g1_flat"
    num_steps_per_env = 24
    max_iterations = 2050
    updates_per_step = 2
    flash_updates_during_rollout = True
    init_at_random_ep_len = True
    algorithm_cfg = {
        "buffer_max_length": 10_000_000,
        "buffer_min_length": 100_000,
        "sample_batch_size": 2048,
        "n_step": 3,
        "asymmetric_observation": False,
        "use_compile": True,
        "compile_mode": "auto",
        "use_amp": True,
        "learning_rate_decay_step": 98_400,
    }
