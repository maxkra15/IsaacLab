# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Captured stepping for the existing Isaac Lab flat G1 scene and Newton MJWarp solver.

The ordinary manager environment constructs the USD scene, articulation, actuators,
contact sensor, and randomized physics model. This adapter replaces its steady-state
Python MDP orchestration with fixed-size Warp operations. It never imports a separate
MuJoCo model or changes the scene's physics stepping cadence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp
from isaaclab_newton.assets.articulation import kernels as articulation_kernels
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from .captured_g1_mdp import G1CapturedMDP

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _reset_joint_history(
    mask: wp.array(dtype=wp.bool),
    velocities: wp.array2d(dtype=wp.float32),
    previous_velocities: wp.array2d(dtype=wp.float32),
    accelerations: wp.array2d(dtype=wp.float32),
):
    env_id, joint_id = wp.tid()
    if mask[env_id]:
        previous_velocities[env_id, joint_id] = velocities[env_id, joint_id]
        accelerations[env_id, joint_id] = 0.0


def _record_arrays(record: Any) -> dict[str, wp.array]:
    """Collect owned arrays and one level of array containers for startup snapshots."""
    result = {}
    for name, value in vars(record).items():
        if isinstance(value, wp.array):
            result[name] = value
        elif isinstance(value, dict):
            for child_name, child in value.items():
                if isinstance(child, wp.array):
                    result[f"{name}.{child_name}"] = child
        elif hasattr(value, "__dict__") and (
            type(value).__module__.startswith("mujoco_warp") or type(value).__name__ == "AttributeNamespace"
        ):
            for child_name, child in vars(value).items():
                if isinstance(child, wp.array):
                    result[f"{name}.{child_name}"] = child
    return result


class CapturedG1Env:
    """Fixed-buffer rollout interface for the stock flat G1 MDP.

    Rendering and manager logging run outside the captured region. The adapter is
    restricted to the G1 flat task with a Newton MJWarp CUDA solver, implicit joint
    drives, and its contact sensor. The owning runner keeps the original environment
    alive and closes it after training.

    Args:
        env: Initialized manager environment with the original G1 scene and startup events.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        from isaaclab.actuators import ImplicitActuatorCfg

        from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

        self.env = env.unwrapped
        self.physics = self.env.sim.physics_manager
        self.device = str(self.env.device)
        self.num_envs = self.env.num_envs
        if not isinstance(self.env.cfg, G1FlatEnvCfg):
            raise ValueError("Captured G1 stepping requires the Isaac-Velocity-Flat-G1 environment configuration.")
        physics_cfg = self.env.cfg.sim.physics
        if (
            not isinstance(physics_cfg, NewtonCfg)
            or not isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg)
            or not wp.get_device(self.device).is_cuda
        ):
            raise ValueError("Captured G1 stepping requires Newton MJWarp physics on a CUDA device.")
        if self.physics._solver.use_mujoco_cpu:
            raise ValueError("Captured G1 stepping cannot use the CPU MuJoCo solver.")
        if not self.physics._use_single_state:
            raise ValueError("Captured G1 stepping requires MJWarp's persistent single-state storage.")
        if self.env.recorder_manager.active_terms or self.env.curriculum_manager.active_terms:
            raise ValueError("Captured G1 stepping requires no recorder or curriculum terms.")
        if set(self.env.scene.articulations) != {"robot"} or set(self.env.scene.sensors) != {"contact_forces"}:
            raise ValueError("Captured G1 stepping requires the single G1 articulation and contact_forces sensor.")
        self.robot = self.env.scene["robot"]
        self.sensor = self.env.scene.sensors["contact_forces"]
        if any(not isinstance(cfg, ImplicitActuatorCfg) for cfg in self.robot.cfg.actuators.values()):
            raise ValueError("Captured G1 stepping requires implicit joint actuators.")
        self.action_dim = self.robot.num_joints
        self.observation_dim = 12 + 3 * self.action_dim
        if self.action_dim != 37 or self.observation_dim != 123:
            raise ValueError("Captured G1 stepping requires the original 37-joint minimal G1 asset.")

        # The enclosing rollout graph owns capture. Internal lazy captures must
        # finish outside it; recording the native solver launches avoids nesting.
        self.env.cfg.sim.physics.use_cuda_graph = False
        self.physics._cfg.use_cuda_graph = False
        self.robot.write_data_to_sim()
        self.physics.step()

        self.steps_per_call = self.env.cfg.decimation if self.env._physics_handles_decimation else 1
        self.scene_dt = self.env.cfg.sim.dt * self.steps_per_call
        self.previous_joint_velocities = wp.clone(self.robot.data.joint_vel.warp)
        self.joint_accelerations = wp.zeros((self.num_envs, self.action_dim), dtype=wp.float32, device=self.device)
        self.mdp = G1CapturedMDP(self.env, joint_acc=self.joint_accelerations)
        self.observations = self.mdp.observations
        self.next_observations = self.mdp.next_observations
        self.rewards = self.mdp.rewards
        self.terminated = self.mdp.terminated
        self.truncated = self.mdp.truncated
        self.episode_totals = self.mdp.episode_totals
        self.reset(self.env.cfg.seed)
        self._snapshot_arrays = self._bind_snapshot_arrays()

    def step(self, actions: wp.array) -> None:
        """Step actions through the actual Isaac Lab solver and reset completed rows on the device."""
        self.mdp.pre_step(actions)
        self.robot.actuators.target_command.set_position_mask(value=self.mdp.joint_targets)
        for _ in range(self.env.cfg.decimation // self.steps_per_call):
            self.robot.write_data_to_sim()
            self.physics.step()
            # Explicit finite differencing replaces ArticulationData's host
            # timestamp guard, with the same cadence as the manager environment.
            wp.launch(
                articulation_kernels.get_joint_acc_from_joint_vel,
                dim=(self.num_envs, self.action_dim),
                inputs=[self.robot.data.joint_vel.warp, self.previous_joint_velocities, self.scene_dt],
                outputs=[self.joint_accelerations],
                device=self.device,
            )
            self.sensor.update(self.scene_dt, force_recompute=not self.env.scene.cfg.lazy_sensor_update)
        if self.env.scene.cfg.lazy_sensor_update:
            # The ordinary task reads contact data first in termination/reward
            # computation, after decimation. Preserve that history/air-time cadence.
            self.sensor._update_outdated_buffers(force_recompute=True)

        self.mdp.post_physics()
        self.mdp.prepare_resets()
        self._reset_state()
        self.mdp.post_reset()
        self._apply_pushes()
        self.mdp.compute_observations()

    def reset(self, seed: int | None = None) -> None:
        """Initialize device RNG and restore all worlds to the task's reset distribution."""
        self.mdp.reset(self.env.cfg.seed if seed is None else seed)
        self.mdp.prepare_resets()
        self._reset_state()
        self.mdp.post_reset(initial=True)
        self.mdp.compute_observations()

    def _reset_state(self) -> None:
        mask = self.mdp.reset_mask
        self.robot.instantaneous_wrench_composer.reset(env_mask=mask)
        self.robot.permanent_wrench_composer.reset(env_mask=mask)
        self.sensor.reset(env_mask=mask)
        self.robot.write_root_pose_to_sim_mask(root_pose=self.mdp.root_poses, env_mask=mask)
        self.robot.write_root_velocity_to_sim_mask(root_velocity=self.mdp.root_velocities, env_mask=mask)
        self.robot.write_joint_position_to_sim_mask(position=self.mdp.joint_positions, env_mask=mask)
        self.robot.write_joint_velocity_to_sim_mask(velocity=self.mdp.joint_velocities, env_mask=mask)
        wp.launch(
            _reset_joint_history,
            dim=(self.num_envs, self.action_dim),
            inputs=[mask, self.mdp.joint_velocities, self.previous_joint_velocities, self.joint_accelerations],
            device=self.device,
        )
        # Native writers flag precisely these worlds. Forward consumes the masks
        # and clears MJWarp warm starts, then republishes public-order state.
        self.physics.forward()
        self.robot.data._refresh_user_order_state()

    def _apply_pushes(self) -> None:
        self.robot.write_root_velocity_to_sim_mask(root_velocity=self.mdp.push_velocities, env_mask=self.mdp.push_mask)
        self.physics.forward()
        self.robot.data._refresh_user_order_state()

    def _bind_snapshot_arrays(self) -> dict[str, wp.array]:
        roots = {
            "mdp": self.mdp,
            "state_0": self.physics.backend.state_0,
            "state_1": self.physics.backend.state_1,
            "control": self.physics.backend.control,
            "mjwarp": self.physics._solver.mjw_data,
            "contacts": self.physics._contacts,
            "contact_view": self.sensor.contact_view,
            "sensor": self.sensor,
            "sensor_data": self.sensor._data,
            "actuators": self.robot.actuators,
            "robot_data": self.robot.data,
            "instantaneous_wrenches": self.robot.instantaneous_wrench_composer,
            "permanent_wrenches": self.robot.permanent_wrench_composer,
        }
        arrays = {
            "previous_joint_velocities": self.previous_joint_velocities,
            "joint_accelerations": self.joint_accelerations,
            "world_reset_mask": self.physics._world_reset_mask,
            "fk_reset_mask": self.physics._fk_reset_mask,
        }
        for prefix, record in roots.items():
            if record is not None:
                arrays.update({f"{prefix}.{name}": array for name, array in _record_arrays(record).items()})
        return arrays

    def state_dict(self) -> dict[str, wp.array]:
        """Clone all device state needed to undo startup warmup or capture execution."""
        return {name: wp.clone(array) for name, array in self._snapshot_arrays.items()}

    def load_state_dict(self, state: dict[str, wp.array]) -> None:
        """Restore a startup snapshot without rebinding captured device pointers."""
        if state.keys() != self._snapshot_arrays.keys():
            raise ValueError("Captured G1 state does not match this initialized scene.")
        for name, array in self._snapshot_arrays.items():
            wp.copy(array, state[name])
