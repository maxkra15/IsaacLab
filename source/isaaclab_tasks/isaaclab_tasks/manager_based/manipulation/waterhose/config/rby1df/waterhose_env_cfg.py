# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env_cfg import RBY1DFWaterhoseEnvCfg


@configclass
class RBY1DFWaterhoseEnvCfg_PLAY(RBY1DFWaterhoseEnvCfg):
    """Small play variant for Newton viewer and teleoperation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.fps = 300
        self.sim_substeps = 10
        self.vbd_iterations = 24
        self.sim.dt = 1.0 / self.fps
        self.sim.physics.num_substeps = int(self.sim_substeps)
        self.sim.physics.use_cuda_graph = not bool(self.disable_cuda_graph)
        self.observations.policy.enable_corruption = False
