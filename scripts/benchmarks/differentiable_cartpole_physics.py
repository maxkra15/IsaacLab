# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental differentiation of Isaac Lab's native Cartpole MJWarp model.

Requires the unmerged MJWarp adjoint branch pinned to
``357a75d60a56d67d476942a1b6e54b3045ee8e87`` (upstream PR #1535).
The ordinary Isaac Lab environment constructs its USD scene and Newton model.
This adapter advances that same model with distinct MJWarp data per integration
step. It is a standalone experiment, not an installed training backend.

Native effort actions enter ``qfrc_applied``. The experimental upstream adjoint
does not expose this input, so this module adds its exact residual derivative:
``dr/df = -I`` gives ``dL/df = H^{-T} dL/dqacc``. GPU finite differences must
validate this extension before interpreting policy-learning results.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src import adjoint, adjoint_util, forward, support
from mujoco_warp._src.types import ConeType, vec5

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


MJWARP_ADJOINT_REVISION = "357a75d60a56d67d476942a1b6e54b3045ee8e87"


# Contact helper adapted from mujoco_warp._src.support.contact_force_fn:
# Copyright 2025 The Newton Developers; Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Full license: docs/licenses/dependencies/mujoco-warp-license.txt
# https://github.com/etaoxing/mujoco_warp/blob/357a75d60a56d67d476942a1b6e54b3045ee8e87/mujoco_warp/_src/support.py
@wp.func
def _native_contact_force_without_adhesion(
    cone: int,
    frames: wp.array[wp.mat33],
    friction: wp.array[vec5],
    dimensions: wp.array[int],
    addresses: wp.array2d[int],
    constraint_forces: wp.array2d[float],
    njmax: int,
    contact_count: wp.array[int],
    world: int,
    contact: int,
    world_frame: bool,
) -> wp.spatial_vector:
    # The adjoint branch adds an adhesion argument to support.contact_force_fn.
    # Newton 1.6 expects the prior signature; reuse the upstream pyramid decoder
    # and express the same force transform for this adhesion-free Cartpole.
    force = wp.spatial_vector()
    address = addresses[contact, 0]
    dimension = dimensions[contact]
    if contact >= 0 and contact < contact_count[0] and address >= 0:
        if cone == ConeType.PYRAMIDAL:
            force = support._decode_pyramid(njmax, constraint_forces[world], address, friction[contact], dimension)
        else:
            for i in range(dimension):
                index = addresses[contact, i]
                if index >= 0 and index < njmax:
                    force[i] = constraint_forces[world, index]
    if world_frame:
        linear = wp.spatial_top(force) @ frames[contact]
        angular = wp.spatial_bottom(force) @ frames[contact]
        force = wp.spatial_vector(linear, angular)
    return force


def install_native_compatibility() -> None:
    """Adapt Newton's contact helper to the pinned PR before constructing a scene.

    This process-local compatibility hook leaves upstream MJWarp's public helper
    and its adhesion-aware kernels intact. It changes no installed source files.
    The Cartpole adapter rejects adhesion before any policy rollout.
    """
    from newton._src.solvers.mujoco import kernels

    kernels._import_contact_force_fn = lambda: _native_contact_force_without_adhesion


@wp.kernel
def _observe(
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    cart_qpos: int,
    pole_qpos: int,
    cart_qvel: int,
    pole_qvel: int,
    observations: wp.array2d[wp.float32],
):
    i = wp.tid()
    observations[i, 0] = qpos[i, cart_qpos]
    observations[i, 1] = qpos[i, pole_qpos]
    observations[i, 2] = qvel[i, cart_qvel]
    observations[i, 3] = qvel[i, pole_qvel]


@wp.kernel
def _apply_effort(
    actions: wp.array2d[wp.float32],
    alive: wp.array[wp.int32],
    cart_dof: int,
    action_scale: float,
    forces: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    force = 0.0
    if j == cart_dof and alive[i] != 0:
        force = wp.clamp(action_scale * actions[i, 0], -400.0, 400.0)
    forces[i, j] = force


@wp.kernel
def _reward(
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    previous_alive: wp.array[wp.int32],
    cart_qpos: int,
    pole_qpos: int,
    cart_qvel: int,
    pole_qvel: int,
    step_dt: float,
    rewards: wp.array[wp.float32],
    alive: wp.array[wp.int32],
):
    i = wp.tid()
    angle = qpos[i, pole_qpos]
    angle = angle - 2.0 * wp.pi * wp.floor((angle + wp.pi) / (2.0 * wp.pi))
    failed = wp.abs(qpos[i, cart_qpos]) > 3.0
    value = 0.0
    if previous_alive[i] != 0:
        running = 1.0
        if failed:
            running = -2.0
        value = (
            running - angle * angle - 0.01 * wp.abs(qvel[i, cart_qvel]) - 0.005 * wp.abs(qvel[i, pole_qvel])
        ) * step_dt
    rewards[i] = value
    alive[i] = int(previous_alive[i] != 0 and not failed)


@wp.kernel(enable_backward=False)
def _accumulate_force_adjoint(
    multiplier: wp.array2d[wp.float32],
    force_gradient: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    force_gradient[i, j] += multiplier[i, j]


@wp.kernel(enable_backward=False)
def _reset_completed(
    alive: wp.array[wp.int32],
    horizon: int,
    episode_limit: int,
    cart_qpos: int,
    pole_qpos: int,
    cart_qvel: int,
    pole_qvel: int,
    cart_asset_joint: int,
    pole_asset_joint: int,
    position_limits: wp.array2d[wp.vec2],
    velocity_limits: wp.array2d[wp.float32],
    rng: wp.array[wp.uint32],
    episode_steps: wp.array[wp.int32],
    reset_mask: wp.array[wp.int32],
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
):
    i = wp.tid()
    steps = episode_steps[i] + horizon
    reset = alive[i] == 0 or steps >= episode_limit
    reset_mask[i] = int(reset)
    if reset:
        state = rng[i]
        qpos[i, cart_qpos] = wp.clamp(
            wp.randf(state, -1.0, 1.0),
            position_limits[i, cart_asset_joint][0],
            position_limits[i, cart_asset_joint][1],
        )
        qpos[i, pole_qpos] = wp.clamp(
            wp.randf(state, -0.25 * wp.pi, 0.25 * wp.pi),
            position_limits[i, pole_asset_joint][0],
            position_limits[i, pole_asset_joint][1],
        )
        qvel[i, cart_qvel] = wp.clamp(
            wp.randf(state, -0.5, 0.5), -velocity_limits[i, cart_asset_joint], velocity_limits[i, cart_asset_joint]
        )
        qvel[i, pole_qvel] = wp.clamp(
            wp.randf(state, -0.25 * wp.pi, 0.25 * wp.pi),
            -velocity_limits[i, pole_asset_joint],
            velocity_limits[i, pole_asset_joint],
        )
        rng[i] = state
        steps = 0
    episode_steps[i] = steps


@wp.kernel(enable_backward=False)
def _clear_reset_rows(mask: wp.array[wp.int32], array: wp.array2d[wp.float32]):
    i, j = wp.tid()
    if mask[i] != 0:
        array[i, j] = 0.0


@wp.kernel(enable_backward=False)
def _initialize_rng(seed: int, rng: wp.array[wp.uint32]):
    i = wp.tid()
    rng[i] = wp.rand_init(seed, i)


class NativeDifferentiableCartpole:
    """Short differentiable rollouts of the existing manager Cartpole scene.

    Observations are ``[cart position, pole angle, cart velocity, pole velocity]``
    with units ``[m, rad, m/s, rad/s]``. Actions are effort / 100 N, with the
    native 400 N actuator limit. Rewards preserve native post-step wrapped angle
    shaping and control-dt scaling. Only cart position outside +/-3 m terminates.

    Each physics integration and observation has separate differentiable buffers.
    Failure masks stop subsequent rewards and bootstrap gradients in the current
    rollout. Resets occur after the update, outside the tape. A horizon dividing
    the 300-step time limit ensures timeouts occur at a rollout boundary and retain
    the terminal critic bootstrap. Masked worlds can still advance with zero
    input after failure; their subsequent samples are excluded from learning.

    Args:
        env: Initialized ``Isaac-Cartpole`` manager environment with Newton MJWarp.
        seed: Seed for device reset RNG. The initial state comes from native reset.
    """

    obs_dim = 4
    action_dim = 1
    observation_dim = 4

    def __init__(self, env: ManagerBasedRLEnv, seed: int = 0):
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg

        self.env = env.unwrapped
        if not isinstance(self.env.cfg, CartpoleEnvCfg):
            raise ValueError("Use the native manager Isaac-Cartpole environment for this experiment.")
        physics_cfg = self.env.cfg.sim.physics
        if not isinstance(physics_cfg, NewtonCfg) or not isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            raise ValueError("Differentiable Cartpole requires native Newton MJWarp physics.")
        if not callable(getattr(mjw, "enable_grad", None)):
            raise RuntimeError("Install the pinned experimental MJWarp adjoint checkout.")
        self.physics = self.env.sim.physics_manager
        self.solver = self.physics._solver
        if self.solver.use_mujoco_cpu:
            raise ValueError("Use the CUDA MJWarp solver.")
        self.model = self.solver.mjw_model
        if self.model.flg_adhesion:
            raise ValueError("The native contact compatibility helper requires adhesion-free physics.")
        self.cpu_model = self.solver.mj_model
        self.num_envs = self.env.num_envs
        self.device = wp.get_device(str(self.env.device))
        self.step_dt = self.env.step_dt
        self.sim_dt = self.env.cfg.sim.dt
        self.substeps = self.env.cfg.decimation * physics_cfg.num_substeps
        self.action_scale = float(self.env.cfg.actions.joint_effort.scale)
        self.episode_limit = self.env.max_episode_length
        if self.cpu_model.nq != 2 or self.cpu_model.nv != 2 or self.solver.mjw_data.nworld != self.num_envs:
            raise ValueError("Expected one two-joint Cartpole per MJWarp world.")
        self.cart_joint = self._joint_id("slider_to_cart")
        self.pole_joint = self._joint_id("cart_to_pole")
        self.cart_qpos = int(self.cpu_model.jnt_qposadr[self.cart_joint])
        self.pole_qpos = int(self.cpu_model.jnt_qposadr[self.pole_joint])
        self.cart_qvel = int(self.cpu_model.jnt_dofadr[self.cart_joint])
        self.pole_qvel = int(self.cpu_model.jnt_dofadr[self.pole_joint])
        robot = self.env.scene["robot"]
        self.cart_asset_joint = robot.find_joints("slider_to_cart")[0][0]
        self.pole_asset_joint = robot.find_joints("cart_to_pole")[0][0]
        self.position_limits = robot.data.soft_joint_pos_limits.warp
        self.velocity_limits = robot.data.soft_joint_vel_limits.warp
        if not math.isclose(self.step_dt, 1 / 60) or self.substeps != 2 or self.episode_limit != 300:
            raise ValueError("Use the native 1/120 s physics, decimation=2, five-second Cartpole configuration.")
        self.horizon = 0
        self.seed = seed
        self.states = []
        self.observations = []
        self.rewards = []
        self.alive = []
        self._gradient_arrays = []
        self.rng = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        self.episode_steps = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self.reset_mask = wp.zeros_like(self.episode_steps)
        wp.launch(_initialize_rng, dim=self.num_envs, inputs=[seed, self.rng], device=self.device)
        mjw.enable_grad()

    def _joint_id(self, name: str) -> int:
        matches = [
            i
            for i in range(self.cpu_model.njnt)
            if (mujoco.mj_id2name(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith(name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one native {name} joint, found {matches}.")
        return matches[0]

    def prepare(self, horizon: int) -> None:
        """Preallocate a fixed differentiable rollout before capture or training."""
        if horizon < 1 or self.episode_limit % horizon:
            raise ValueError("The rollout horizon must be a positive divisor of the native 300-step time limit.")
        if self.horizon:
            if self.horizon != horizon:
                raise ValueError("Create a new adapter to change the allocated rollout horizon.")
            return
        self.horizon = horizon
        with wp.ScopedDevice(self.device):
            # Native reset updates Newton joint coordinates and forward
            # kinematics. With update_data_interval=1, SolverMuJoCo defers its
            # qpos/qvel conversion until the next solver step. Publish that
            # same coordinate conversion now, without advancing time, before
            # detaching the native reset into the experimental rollout.
            self.env.scene.write_data_to_sim()
            self.env.sim.forward()
            self.solver._update_mjc_data(self.solver.mjw_data, self.solver.model, self.physics.get_state_0())
            # SolverMuJoCo.step normally fills this option from its dt argument.
            # The out-of-place experiment calls MJWarp directly instead.
            self.model.opt.timestep.fill_(self.step_dt / self.substeps)
            self.states = [adjoint_util._clone_nograd(self.solver.mjw_data) for _ in range(horizon * self.substeps + 1)]
            for data in self.states:
                data.qfrc_applied.requires_grad = True
                data.ctrl.requires_grad = True
                self._gradient_arrays.extend(adjoint._prepare_data(data))
                self._gradient_arrays.extend((data.qfrc_applied, data.ctrl))
            self.backward_workspace = mjw.create_backward_context(self.model, self.states[0])
            self.observations = [wp.zeros((self.num_envs, 4), requires_grad=True) for _ in range(horizon + 1)]
            self.rewards = [wp.zeros(self.num_envs, requires_grad=True) for _ in range(horizon)]
            self.alive = [wp.ones(self.num_envs, dtype=wp.int32) for _ in range(horizon + 1)]
            self._gradient_arrays.extend((*self.observations, *self.rewards))

    def begin_rollout(self) -> None:
        """Clear adjoints at the detached rollout boundary, outside the actor tape."""
        self.alive[0].fill_(1)
        for array in self._gradient_arrays:
            array.grad.zero_()

    def observe(self, step: int) -> wp.array:
        """Read a unique policy observation from the selected control boundary."""
        data = self.states[step * self.substeps]
        wp.launch(
            _observe,
            dim=self.num_envs,
            inputs=[data.qpos, data.qvel, self.cart_qpos, self.pole_qpos, self.cart_qvel, self.pole_qvel],
            outputs=[self.observations[step]],
            device=self.device,
        )
        return self.observations[step]

    def _backward_step(self, data, next_data) -> None:
        adjoint.step_backward(self.model, data, next_data, bc=self.backward_workspace)
        wp.launch(
            _accumulate_force_adjoint,
            dim=data.qfrc_applied.shape,
            inputs=[self.backward_workspace.solver_ctx.search],
            outputs=[data.qfrc_applied.grad],
            device=self.device,
        )

    def _physics_step(self, data, next_data) -> None:
        # Preserve the native forward call and register one adjoint, including
        # the generalized-force input omitted by the experimental upstream hook.
        runtime = wp._src.context.runtime
        tape = runtime.tape
        runtime.tape = None
        try:
            mjw.step(self.model, data, next_data)
        finally:
            runtime.tape = tape
        if tape is not None:
            arrays = [data.qpos, data.qvel, data.qfrc_applied, *adjoint.step_backward_arrays(data, next_data)]
            tape.record_func(lambda: self._backward_step(data, next_data), arrays=arrays)

    def step(self, step: int, actions: wp.array) -> tuple[wp.array, wp.array]:
        """Advance native physics and return reward plus failure continuation."""
        for substep in range(self.substeps):
            index = step * self.substeps + substep
            data, next_data = self.states[index : index + 2]
            wp.launch(
                _apply_effort,
                dim=data.qfrc_applied.shape,
                inputs=[actions, self.alive[step], self.cart_qvel, self.action_scale],
                outputs=[data.qfrc_applied],
                device=self.device,
            )
            self._physics_step(data, next_data)
        data = self.states[(step + 1) * self.substeps]
        wp.launch(
            _reward,
            dim=self.num_envs,
            inputs=[
                data.qpos,
                data.qvel,
                self.alive[step],
                self.cart_qpos,
                self.pole_qpos,
                self.cart_qvel,
                self.pole_qvel,
                self.step_dt,
            ],
            outputs=[self.rewards[step], self.alive[step + 1]],
            device=self.device,
        )
        return self.rewards[step], self.alive[step + 1]

    def finish_rollout(self) -> wp.array:
        """Read the pre-reset terminal observation for critic bootstrapping."""
        return self.observe(self.horizon)

    def after_update(self) -> None:
        """Detach the advanced state and reset finished worlds outside the tape."""
        forward._copy_state(self.states[-1], self.states[0])
        data = self.states[0]
        wp.launch(
            _reset_completed,
            dim=self.num_envs,
            inputs=[
                self.alive[-1],
                self.horizon,
                self.episode_limit,
                self.cart_qpos,
                self.pole_qpos,
                self.cart_qvel,
                self.pole_qvel,
                self.cart_asset_joint,
                self.pole_asset_joint,
                self.position_limits,
                self.velocity_limits,
                self.rng,
                self.episode_steps,
                self.reset_mask,
                data.qpos,
                data.qvel,
            ],
            device=self.device,
        )
        for array in (data.qacc_warmstart, data.qfrc_applied, data.ctrl, data.act):
            if array.size:
                wp.launch(_clear_reset_rows, dim=array.shape, inputs=[self.reset_mask, array], device=self.device)

    def metadata(self) -> dict:
        """Describe the native model, training reward, and experimental boundaries."""
        return {
            "physics": "native_isaaclab_newton_mjwarp",
            "mjwarp_adjoint_revision": MJWARP_ADJOINT_REVISION,
            "force_adjoint_extension": "qfrc_applied residual derivative -I; adjoint adds IFT multiplier",
            "observation_order": ["cart_position", "pole_angle", "cart_velocity", "pole_angular_velocity"],
            "num_envs": self.num_envs,
            "horizon": self.horizon,
            "physics_steps_per_control": self.substeps,
            "control_dt": self.step_dt,
            "physics_dt": self.sim_dt,
            "action_scale_N": self.action_scale,
            "action_effort_limit_N": 400.0,
            "episode_steps": self.episode_limit,
            "reward": "native dt-scaled alive/failure, wrapped angle squared and absolute velocity penalties",
            "resets": "native sampled ranges and soft-limit clamps; Warp RNG; post-update reset outside tape",
            "initial_state": "native reset converted Newton-to-MJWarp before cloning, with no physics step",
            "model_counts": {
                "nq": self.cpu_model.nq,
                "nv": self.cpu_model.nv,
                "nu": self.cpu_model.nu,
                "ngeom": self.cpu_model.ngeom,
            },
        }
