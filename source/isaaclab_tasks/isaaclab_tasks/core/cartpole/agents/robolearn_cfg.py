# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboLearn configurations for the existing Cartpole MDP."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.robolearn import RoboLearnRunnerCfg


@configclass
class CartpoleFlashSACCfg(RoboLearnRunnerCfg):
    experiment_name = "cartpole"
    algorithm = "flashsac"
    updates_per_step = 2


@configclass
class CartpoleWarpPPOCfg(RoboLearnRunnerCfg):
    experiment_name = "cartpole"
    algorithm = "warp_ppo"
    algorithm_cfg = {
        "hidden_dims": [32, 32],
        "learning_rate": 1.0e-3,
        "epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "value_coefficient": 2.0,
        "entropy_coefficient": 0.005,
        "clip_value": True,
    }


@configclass
class CartpoleDirectFlashSACCfg(CartpoleFlashSACCfg):
    experiment_name = "cartpole_direct"


@configclass
class CartpoleDirectWarpPPOCfg(CartpoleWarpPPOCfg):
    experiment_name = "cartpole_direct"


@configclass
class CartpoleWarpFlashSACCfg(CartpoleFlashSACCfg):
    """Experimental FP32 FlashSAC with captured Warp-NN updates."""

    algorithm = "warp_flashsac"


@configclass
class CartpoleDirectWarpFlashSACCfg(CartpoleWarpFlashSACCfg):
    experiment_name = "cartpole_direct"
