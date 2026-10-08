# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboLearn Warp-NN PPO configuration for the existing flat G1 locomotion MDP."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.robolearn import RoboLearnRunnerCfg


@configclass
class G1FlatWarpPPORunnerCfg(RoboLearnRunnerCfg):
    """Train separate actor and critic networks with captured full-batch PPO updates."""

    algorithm = "warp_ppo"
    experiment_name = "g1_flat"
    num_steps_per_env = 24
    max_iterations = 1500
    algorithm_cfg = {
        "hidden_dims": [256, 128, 128],
        "learning_rate": 3.0e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        # Warp PPO multiplies squared value error by 0.5 before applying this coefficient.
        "value_coefficient": 2.0,
        "entropy_coefficient": 0.008,
        "epochs": 5,
        "max_grad_norm": 1.0,
        "initial_std": 1.0,
        "normalize_advantages": True,
        "clip_value": True,
    }
