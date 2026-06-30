# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.cloner import queue_newton_physics_replication
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_newton.sim.spawners.mpm import create_mpm_particle_visualization, emit_mpm_particles

from .kernels import (
    compute_particle_state_w,
    gather_particles_vec3f,
    scatter_particles_state_vec6f_index,
    scatter_particles_state_vec6f_mask,
    scatter_particles_vec3f_index,
    scatter_particles_vec3f_mask,
    vec6f,
)
from .mpm_object_data import MPMObjectData

if TYPE_CHECKING:
    from .mpm_object_cfg import MPMObjectCfg

logger = logging.getLogger(__name__)


@dataclass
class MPMObjectRegistryEntry:
    """Particle object registration consumed by Newton builder replication."""

    cfg: MPMObjectCfg
    particle_offsets: list[int] = field(default_factory=list)
    particles_per_object: int = 0


def add_mpm_entry_to_builder(
    builder,
    entry: MPMObjectRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Emit one registered MPM object into one Newton builder world."""
    if env_idx == 0:
        entry.particle_offsets.clear()
        entry.particles_per_object = 0

    before_count = int(getattr(builder, "particle_count", 0))
    position, orientation = _compose_env_asset_pose(entry.cfg, env_position, env_rotation)
    emit_mpm_particles(builder, entry.cfg.spawn, position=position, orientation=orientation)
    after_count = int(getattr(builder, "particle_count", 0))
    delta = after_count - before_count

    entry.particle_offsets.append(before_count)
    if env_idx == 0:
        entry.particles_per_object = delta
    elif entry.particles_per_object != delta:
        raise RuntimeError(
            f"MPM object '{entry.cfg.prim_path}' produced {delta} particles in env {env_idx}, "
            f"but env 0 produced {entry.particles_per_object}."
        )


def add_registered_mpm_objects_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Emit all registered MPM objects into one Newton builder world."""
    for entry in getattr(SimulationManager, "_mpm_object_registry", []):
        add_mpm_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)


def clear_registered_mpm_objects() -> None:
    """Clear registered MPM object state."""
    SimulationManager._mpm_object_registry = []


class MPMObject(BaseDeformableObject):
    """Newton MPM particle object asset.

    The object is presented through Isaac Lab's deformable-object interface so it
    can participate in existing scene reset/update/state workflows while exposing
    particle-specific aliases on :attr:`data`.
    """

    cfg: MPMObjectCfg
    __backend_name__: str = "newton"

    def __init__(self, cfg: MPMObjectCfg):
        super().__init__(cfg)
        queue_newton_physics_replication(cfg)
        self._registry_entry = MPMObjectRegistryEntry(self.cfg)
        if not hasattr(SimulationManager, "_mpm_object_registry"):
            SimulationManager._mpm_object_registry = []
        SimulationManager._mpm_object_registry.append(self._registry_entry)
        self._kit_points = None
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}

    @property
    def data(self) -> MPMObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        return 1

    @property
    def max_sim_vertices_per_body(self) -> int:
        return self._particles_per_object

    @property
    def particles_per_object(self) -> int:
        """Number of particles generated for each environment instance."""
        return self._particles_per_object

    @property
    def particle_offsets(self) -> wp.array(dtype=wp.int32):
        """Starting model-particle index for each environment instance."""
        return self._particle_offsets

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset selected particle instances to their default particle state."""
        if env_mask is not None:
            self.write_nodal_state_to_sim_mask(self.data.default_nodal_state_w.warp, env_mask=env_mask)
        else:
            self.write_nodal_state_to_sim_index(
                self.data.default_nodal_state_w.warp,
                env_ids=self._resolve_env_ids(env_ids),
                full_data=True,
            )

    def write_data_to_sim(self):
        """No-op; MPM particle writes are applied immediately by write methods."""

    def update(self, dt: float):
        self._data.update(dt)

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_state, (self.num_instances, self._particles_per_object), vec6f, "nodal_state"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_state, (env_ids.shape[0], self._particles_per_object), vec6f, "nodal_state"
            )
        if isinstance(nodal_state, torch.Tensor):
            nodal_state = wp.from_torch(nodal_state.contiguous(), dtype=vec6f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_state_vec6f_index,
                dim=(env_ids.shape[0], self._particles_per_object),
                inputs=[nodal_state, env_ids, self._particle_offsets, full_data],
                outputs=[state.particle_q, state.particle_qd],
                device=self.device,
            )
        self._invalidate_particle_state_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self._particles_per_object), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_pos, (env_ids.shape[0], self._particles_per_object), wp.vec3f, "nodal_pos"
            )
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_index,
                dim=(env_ids.shape[0], self._particles_per_object),
                inputs=[nodal_pos, env_ids, self._particle_offsets, full_data],
                outputs=[state.particle_q],
                device=self.device,
            )
        self._invalidate_particle_pos_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self._particles_per_object), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_vel, (env_ids.shape[0], self._particles_per_object), wp.vec3f, "nodal_vel"
            )
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_index,
                dim=(env_ids.shape[0], self._particles_per_object),
                inputs=[nodal_vel, env_ids, self._particle_offsets, full_data],
                outputs=[state.particle_qd],
                device=self.device,
            )
        self._invalidate_particle_vel_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        raise NotImplementedError("MPMObject does not support deformable kinematic targets.")

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        env_mask = self._resolve_mask(env_mask)
        if isinstance(nodal_state, ProxyArray):
            nodal_state = nodal_state.warp
        self.assert_shape_and_dtype(nodal_state, (env_mask.shape[0], self._particles_per_object), vec6f, "nodal_state")
        if isinstance(nodal_state, torch.Tensor):
            nodal_state = wp.from_torch(nodal_state.contiguous(), dtype=vec6f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_state_vec6f_mask,
                dim=(env_mask.shape[0], self._particles_per_object),
                inputs=[nodal_state, env_mask, self._particle_offsets],
                outputs=[state.particle_q, state.particle_qd],
                device=self.device,
            )
        self._invalidate_particle_state_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        env_mask = self._resolve_mask(env_mask)
        if isinstance(nodal_pos, ProxyArray):
            nodal_pos = nodal_pos.warp
        self.assert_shape_and_dtype(nodal_pos, (env_mask.shape[0], self._particles_per_object), wp.vec3f, "nodal_pos")
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_mask,
                dim=(env_mask.shape[0], self._particles_per_object),
                inputs=[nodal_pos, env_mask, self._particle_offsets],
                outputs=[state.particle_q],
                device=self.device,
            )
        self._invalidate_particle_pos_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        env_mask = self._resolve_mask(env_mask)
        if isinstance(nodal_vel, ProxyArray):
            nodal_vel = nodal_vel.warp
        self.assert_shape_and_dtype(nodal_vel, (env_mask.shape[0], self._particles_per_object), wp.vec3f, "nodal_vel")
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)

        for state in self._iter_particle_states():
            wp.launch(
                scatter_particles_vec3f_mask,
                dim=(env_mask.shape[0], self._particles_per_object),
                inputs=[nodal_vel, env_mask, self._particle_offsets],
                outputs=[state.particle_qd],
                device=self.device,
            )
        self._invalidate_particle_vel_cache()
        SimulationManager._mark_particles_dirty()

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array | ProxyArray,
        env_mask: wp.array | torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError("MPMObject does not support deformable kinematic targets.")

    write_particle_state_to_sim_index = write_nodal_state_to_sim_index
    write_particle_pos_to_sim_index = write_nodal_pos_to_sim_index
    write_particle_velocity_to_sim_index = write_nodal_velocity_to_sim_index
    write_particle_state_to_sim_mask = write_nodal_state_to_sim_mask
    write_particle_pos_to_sim_mask = write_nodal_pos_to_sim_mask
    write_particle_velocity_to_sim_mask = write_nodal_velocity_to_sim_mask

    def _initialize_impl(self):
        entry = self._registry_entry
        self._num_instances = len(entry.particle_offsets)
        self._particles_per_object = entry.particles_per_object
        self._recorded_particle_offsets = entry.particle_offsets

        if self._num_instances == 0 or self._particles_per_object == 0:
            raise RuntimeError(
                f"No MPM particle instances found for '{self.cfg.prim_path}'. "
                "Ensure Newton replication processed the MPM object registry."
            )

        logger.info("Newton MPM object initialized at: %s", self.cfg.prim_path)
        logger.info("Number of instances: %d", self._num_instances)
        logger.info("Particles per object: %d", self._particles_per_object)

        self._particle_offsets = wp.array(self._recorded_particle_offsets, dtype=wp.int32, device=self.device)
        self._data = MPMObjectData(
            particle_offsets=self._particle_offsets,
            particles_per_object=self._particles_per_object,
            num_instances=self._num_instances,
            device=self.device,
        )
        self._create_buffers()
        self.update(0.0)

        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._data._create_simulation_bindings(),
            PhysicsEvent.PHYSICS_READY,
            name=f"mpm_object_rebind_{self.cfg.prim_path}",
        )

    def _create_buffers(self):
        self._ALL_INDICES = wp.array(np.arange(self._num_instances, dtype=np.int32), device=self.device)
        self._ALL_ENV_MASK = wp.ones((self._num_instances,), dtype=wp.bool, device=self.device)

        state = SimulationManager.get_state_0()
        if state is None or state.particle_q is None or state.particle_qd is None:
            raise RuntimeError("Cannot initialize MPMObject buffers before Newton particle state exists.")

        default_pos = wp.zeros((self._num_instances, self._particles_per_object), dtype=wp.vec3f, device=self.device)
        default_vel = wp.zeros((self._num_instances, self._particles_per_object), dtype=wp.vec3f, device=self.device)
        default_state = wp.zeros((self._num_instances, self._particles_per_object), dtype=vec6f, device=self.device)
        wp.launch(
            gather_particles_vec3f,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[state.particle_q, self._particle_offsets, self._particles_per_object],
            outputs=[default_pos],
            device=self.device,
        )
        wp.launch(
            gather_particles_vec3f,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[state.particle_qd, self._particle_offsets, self._particles_per_object],
            outputs=[default_vel],
            device=self.device,
        )
        wp.launch(
            compute_particle_state_w,
            dim=(self._num_instances, self._particles_per_object),
            inputs=[default_pos, default_vel],
            outputs=[default_state],
            device=self.device,
        )
        self._data.default_nodal_state_w = ProxyArray(default_state)
        self._data.default_particle_state_w = self._data.default_nodal_state_w
        self._create_kit_points()

    def _create_kit_points(self) -> None:
        """Create a Kit-visible point cloud for MPM particles when the Kit visualizer is active."""
        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        sim = SimulationContext.instance()
        if sim is None or "kit" not in sim.resolve_visualizer_types() or not self.cfg.spawn.visible:
            return

        self._kit_points = create_mpm_particle_visualization(
            prim_path=_create_kit_visualization_path(self.cfg.prim_path),
            positions=self.data.particle_pos_w.torch,
            particle_offsets=self._recorded_particle_offsets,
            widths=_particle_visual_widths_per_object(self.cfg.spawn, self._particles_per_object),
            color=self.cfg.spawn.visual_color,
            sync_frequency=self.cfg.spawn.visual_update_frequency,
        )
        for prim_path in self._kit_points.prim_paths:
            SimulationManager.register_particle_visual_prim(prim_path)
        logger.info("Kit MPM particle visualization initialized at: %s", self._kit_points.base_path)

    def _resolve_env_ids(self, env_ids):
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(device=self.device, dtype=torch.int32), dtype=wp.int32)
        if isinstance(env_ids, Sequence):
            return wp.array(list(env_ids), dtype=wp.int32, device=self.device)
        return env_ids

    def _resolve_mask(self, mask: wp.array | torch.Tensor | None) -> wp.array:
        if mask is None:
            return self._ALL_ENV_MASK
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            return wp.from_torch(mask.to(device=self.device).contiguous(), dtype=wp.bool)
        return mask

    def _iter_particle_states(self):
        seen: set[int] = set()
        for state in (SimulationManager.get_state_0(), SimulationManager.get_state_1()):
            if state is None or id(state) in seen:
                continue
            seen.add(id(state))
            yield state

    def _invalidate_particle_pos_cache(self) -> None:
        self._data._particle_pos_w.timestamp = -1.0
        self._data._particle_state_w.timestamp = -1.0
        self._data._root_pos_w.timestamp = -1.0

    def _invalidate_particle_vel_cache(self) -> None:
        self._data._particle_vel_w.timestamp = -1.0
        self._data._particle_state_w.timestamp = -1.0
        self._data._root_vel_w.timestamp = -1.0

    def _invalidate_particle_state_cache(self) -> None:
        self._invalidate_particle_pos_cache()
        self._invalidate_particle_vel_cache()

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("Debug visualization is not implemented for MPMObject.")

    def _debug_vis_callback(self, event):
        raise NotImplementedError("Debug visualization is not implemented for MPMObject.")

    def _clear_callbacks(self) -> None:
        super()._clear_callbacks()
        self._kit_points = None
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
        registry = getattr(SimulationManager, "_mpm_object_registry", None)
        if registry is not None and hasattr(self, "_registry_entry") and self._registry_entry in registry:
            registry.remove(self._registry_entry)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)


def _compose_env_asset_pose(
    cfg: MPMObjectCfg,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]),
        float(env_rotation[1]),
        float(env_rotation[2]),
        float(env_rotation[3]),
    )
    init_pos = wp.vec3(float(cfg.init_state.pos[0]), float(cfg.init_state.pos[1]), float(cfg.init_state.pos[2]))
    init_rot = wp.quat(
        float(cfg.init_state.rot[0]),
        float(cfg.init_state.rot[1]),
        float(cfg.init_state.rot[2]),
        float(cfg.init_state.rot[3]),
    )
    asset_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    asset_rot = env_rot * init_rot
    return (
        (float(asset_pos[0]), float(asset_pos[1]), float(asset_pos[2])),
        (float(asset_rot[0]), float(asset_rot[1]), float(asset_rot[2]), float(asset_rot[3])),
    )


def _create_kit_visualization_path(prim_path: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in prim_path.strip("/"))
    return f"/World/Visuals/MPMParticles/{sanitized or 'Object'}"


def _particle_visual_widths_per_object(spawn_cfg, particles_per_object: int) -> list[float]:
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMPointsCfg

    if isinstance(spawn_cfg, MPMGridCfg):
        radius = spawn_cfg.radius
        if radius is None:
            lower = np.asarray(spawn_cfg.lower, dtype=np.float32)
            upper = np.asarray(spawn_cfg.upper, dtype=np.float32)
            extent = upper - lower
            resolution = np.maximum(np.ceil(spawn_cfg.particles_per_cell * extent / spawn_cfg.voxel_size), 1)
            cell_size = extent / resolution
            radius = 0.5 * float(np.max(cell_size))
        return [2.0 * float(radius)] * particles_per_object

    if isinstance(spawn_cfg, MPMPointsCfg):
        radius = spawn_cfg.radius
        if isinstance(radius, (int, float)):
            return [2.0 * float(radius)] * particles_per_object
        if len(radius) != particles_per_object:
            raise ValueError(
                f"MPMPointsCfg radius must be scalar or have one value per particle. Got {len(radius)} values."
            )
        return [2.0 * float(value) for value in radius]

    return [0.02] * particles_per_object
