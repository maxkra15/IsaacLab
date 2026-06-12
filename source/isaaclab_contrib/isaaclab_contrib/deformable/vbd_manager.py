# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import JointType, Model, ModelBuilder, eval_fk
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverVBD

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage

from isaaclab_contrib.cable.cable_object import install_cable_builder_hooks

from .deformable_object import install_deformable_builder_hooks, setup_registered_deformable_fabric_sync
from .newton_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


def _rotate_vector_by_quat_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by an ``xyzw`` quaternion."""

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return vec
    quat = quat / norm
    q_vec = quat[:3]
    q_w = quat[3]
    t = 2.0 * np.cross(q_vec, vec)
    return vec + q_w * t + np.cross(q_vec, t)


@wp.kernel(enable_backward=False)
def _write_free_joint_q_from_body_q(
    free_joint_ids: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformf),
    joint_q: wp.array(dtype=float),
):
    """Copy each FREE joint's child ``body_q`` transform into its ``joint_q`` slot.

    FREE joints store their pose as 7 floats matching the ``wp.transformf``
    layout, so this is a straight component copy.

    NOTE: Can be removed once VBD supports maximal coordinates with full state updates.
    """
    j = free_joint_ids[wp.tid()]
    child = joint_child[j]
    t = body_q[child]
    p = wp.transform_get_translation(t)
    r = wp.transform_get_rotation(t)
    q0 = joint_q_start[j]
    joint_q[q0 + 0] = p[0]
    joint_q[q0 + 1] = p[1]
    joint_q[q0 + 2] = p[2]
    joint_q[q0 + 3] = r[0]
    joint_q[q0 + 4] = r[1]
    joint_q[q0 + 5] = r[2]
    joint_q[q0 + 6] = r[3]


@wp.kernel(enable_backward=False)
def _sync_cable_curve_points(
    fabric_points: wp.fabricarrayarray(dtype=wp.vec3f),
    fabric_world_matrices: wp.fabricarray(dtype=wp.mat44d),
    body_offsets: wp.fabricarray(dtype=wp.uint32),
    body_counts: wp.fabricarray(dtype=wp.uint32),
    last_edge_lengths: wp.fabricarray(dtype=wp.float32),
    body_q: wp.array(ndim=1, dtype=wp.transformf),
):
    """Reconstruct ``UsdGeomBasisCurves`` control points from cable body transforms."""
    i = wp.tid()
    offset = int(body_offsets[i])
    count = int(body_counts[i])
    inv_world = wp.inverse(wp.transpose(wp.mat44f(fabric_world_matrices[i])))

    for j in range(count):
        node_world = wp.transform_get_translation(body_q[offset + j])
        fabric_points[i][j] = wp.transform_point(inv_world, node_world)

    tail_world = wp.transform_point(body_q[offset + count - 1], wp.vec3(0.0, 0.0, float(last_edge_lengths[i])))
    fabric_points[i][count] = wp.transform_point(inv_world, tail_world)


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the VBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    _newton_cable_body_offset_attr = "newton:cableBodyOffset"
    _newton_cable_body_count_attr = "newton:cableBodyCount"
    _newton_cable_last_edge_length_attr = "newton:cableLastEdgeLength"
    _curves_dirty: bool = False
    _cable_body_q_cpu = None
    _fk_mask: wp.array | None = None
    """``False`` for articulations VBD owns directly in ``body_q`` (CABLE or FREE joints), ``True`` elsewhere."""
    _free_joint_ids: wp.array | None = None
    """Indices of FREE joints whose ``body_q`` must be copied back to ``joint_q`` each step."""

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """

        # Deformable body registry and extension hooks.
        # Experimental deformable support registers callbacks here so the manager
        # and cloner can invoke them without hard-coding deformable logic.
        install_deformable_builder_hooks()
        install_cable_builder_hooks()

        super().initialize(sim_context)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear VBD-specific Fabric sync state and shared builder hooks."""
        cls._curves_dirty = False
        cls._cable_body_q_cpu = None
        cls._fk_mask = None
        cls._free_joint_ids = None
        NewtonManager._cable_registry = []
        NewtonManager._pending_cable_attachments = []
        NewtonManager._pending_cable_sdf_captures = []
        NewtonManager._deformable_registry = []
        NewtonManager._per_world_builder_hooks = []

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset the VBD physics simulation.

        ``soft=True`` snaps bodies back to their rest pose without rebuilding.
        :attr:`SolverVBD.body_q_prev` must also be restored, since AVBD derives
        velocity as ``(body_q - body_q_prev) / dt``.

        NOTE: This is a temporary workaround, can be patched once Newton supports maximal coordinates in VBD with FK.

        Args:
            soft: If True, snap state in place; otherwise reinitialize fully.
        """
        super().reset(soft)
        if not soft:
            return
        model = cls._model
        state = cls._state_0
        solver = cls._solver
        if model is None or state is None or solver is None or model.body_count == 0:
            return
        wp.copy(state.body_q, model.body_q)
        wp.copy(solver.body_q_prev, model.body_q)
        zero_qd = wp.zeros(model.body_count, dtype=state.body_qd.dtype, device=state.body_qd.device)
        zero_inertia_q = wp.zeros(
            model.body_count, dtype=solver.body_inertia_q.dtype, device=solver.body_inertia_q.device
        )
        wp.copy(state.body_qd, zero_qd)
        wp.copy(solver.body_inertia_q, zero_inertia_q)
        cls._mark_state_dirty()

    @classmethod
    def _mark_curves_dirty(cls) -> None:
        """Flag that cable curve points have changed and Fabric needs re-sync."""
        cls._curves_dirty = True

    @classmethod
    def _mark_state_dirty(cls) -> None:
        """Flag that all VBD state has changed and Fabric needs re-sync."""
        super()._mark_state_dirty()
        cls._mark_curves_dirty()

    @classmethod
    def _get_deformable_ignore_paths(cls) -> list[str]:
        """Return USD prim paths to skip when calling ``builder.add_usd``.

        For each registered deformable body, both the simulation mesh (which
        carries ``UsdPhysics.CollisionAPI``) and the visual mesh are returned.
        The sim mesh must be skipped so Newton does not create a redundant
        static mesh collider alongside the particles produced by
        ``add_soft_mesh``.  The visual mesh is skipped so Newton does not
        treat it as a collider — Kit reads it directly from USD for rendering.

        Paths may contain regex patterns; Newton's ``add_usd`` matches them
        via :func:`re.match`.
        """
        paths: list[str] = []
        for entry in cls._deformable_registry:
            paths.append(entry.sim_mesh_prim_path)
            paths.append(entry.vis_mesh_prim_path)
        return paths

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.

        TODO: Subclass should not override this method, missing piece is
        having Newton bind a surface mesh to volume deformable tetrahedral mesh
        in addition to removing the deformable_registry data structure.
        """
        super().start_simulation()

        # Apply global model parameters from :class:`NewtonModelCfg` to the finalized model.
        # Sets ``soft_contact_ke/kd/mu`` and optionally overrides per-shape
        # ``shape_material_ke/kd/mu`` on the Newton model.
        cfg = PhysicsManager._cfg
        if cfg is not None and hasattr(cfg, "model_cfg") and cfg.model_cfg is not None:
            model = cls._model
            if model is None:
                return

            model_cfg = cfg.model_cfg
            model.soft_contact_ke = float(model_cfg.soft_contact_ke)
            model.soft_contact_kd = float(model_cfg.soft_contact_kd)
            model.soft_contact_mu = float(model_cfg.soft_contact_mu)

            if model_cfg.shape_material_ke is not None:
                model.shape_material_ke.fill_(float(model_cfg.shape_material_ke))
            if model_cfg.shape_material_kd is not None:
                model.shape_material_kd.fill_(float(model_cfg.shape_material_kd))
            if model_cfg.shape_material_mu is not None:
                model.shape_material_mu.fill_(float(model_cfg.shape_material_mu))

        # Setup USD/Fabric sync for Kit viewport deformable rendering
        setup_registered_deformable_fabric_sync(cls)

        if not cls._clone_physics_only and cls._cable_registry:
            import re

            stage = get_current_stage()
            curves_registered = False
            for entry in cls._cable_registry:
                curve_template_path = entry.curve_prim_path or f"{entry.prim_path}/geometry/mesh"
                for inst_idx, body_offset in enumerate(entry.body_offsets):
                    resolved = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), curve_template_path)
                    resolved = re.sub(r"\.\*", str(inst_idx), resolved)
                    curve_prim = stage.GetPrimAtPath(resolved)
                    if not curve_prim or not curve_prim.IsValid():
                        logger.warning("[setup_fabric_cable_sync] curve prim not found at %s", resolved)
                        continue
                    usd_points = curve_prim.GetAttribute("points").Get()
                    expected_points = len(entry.edges) + 1
                    if usd_points is None or len(usd_points) != expected_points:
                        logger.warning(
                            "[setup_fabric_cable_sync] curve %s has %s points, expected %d; skipping.",
                            resolved,
                            0 if usd_points is None else len(usd_points),
                            expected_points,
                        )
                        continue
                    curves_registered = True
            if curves_registered:
                cls._mark_curves_dirty()

    @classmethod
    def _build_fk_mask(cls) -> None:
        """Build :attr:`_fk_mask` excluding articulations with CABLE or FREE joints.

        Newton's ``eval_fk`` overwrites ``body_q`` for these joints, but VBD owns it directly.
        """
        model = cls._model
        if model is None or model.joint_type is None or model.joint_articulation is None:
            return
        if model.articulation_count == 0:
            return

        joint_type_np = model.joint_type.numpy()
        joint_articulation_np = model.joint_articulation.numpy()
        excluded_types = (int(JointType.CABLE), int(JointType.FREE))
        excluded_art_ids = {
            int(joint_articulation_np[j])
            for j in range(len(joint_type_np))
            if int(joint_type_np[j]) in excluded_types and int(joint_articulation_np[j]) >= 0
        }
        if not excluded_art_ids:
            return

        mask_np = np.ones(model.articulation_count, dtype=np.bool_)
        for art_id in excluded_art_ids:
            mask_np[art_id] = False
        cls._fk_mask = wp.array(mask_np, dtype=wp.bool, device=PhysicsManager._device)

    @classmethod
    def _build_free_joint_ids(cls) -> None:
        """Cache FREE joint indices for the :func:`_write_free_joint_q_from_body_q` kernel.

        VBD integrates FREE-jointed rigids in ``body_q`` only, leaving the
        corresponding ``joint_q[0:7]`` slots stale. Newton's ``ArticulationView``
        reads root transforms from ``joint_q``, so without this writeback
        :attr:`RigidObjectData.body_com_pos_w` and friends never update.

        NOTE: Can be removed once VBD supports maximal coordinates with full state updates.
        """
        model = cls._model
        if model is None or model.joint_type is None or int(model.joint_count) == 0:
            return
        joint_type_np = model.joint_type.numpy()
        free_idx = np.where(joint_type_np == int(JointType.FREE))[0]
        if free_idx.size == 0:
            return
        cls._free_joint_ids = wp.array(free_idx.astype(np.int32), dtype=int, device=PhysicsManager._device)

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """Run substeps, then sync FREE-joint ``joint_q`` from ``body_q``.

        See :meth:`_build_free_joint_ids` for why the writeback is required.
        Index arrays are built lazily in :meth:`forward` at init time to avoid
        host-device transfers on the step path (which fail under graph capture).
        """
        super()._run_solver_substeps(contacts)
        if cls._free_joint_ids is None:
            return
        wp.launch(
            _write_free_joint_q_from_body_q,
            dim=int(cls._free_joint_ids.shape[0]),
            inputs=[
                cls._free_joint_ids,
                cls._model.joint_q_start,
                cls._model.joint_child,
                cls._state_0.body_q,
                cls._state_0.joint_q,
            ],
            device=PhysicsManager._device,
        )

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics, skipping articulations VBD owns directly."""
        if cls._fk_mask is None:
            cls._build_fk_mask()
            cls._build_free_joint_ids()
            if cls._fk_mask is None:
                super().forward()
                return
        eval_fk(
            cls._model,
            cls._state_0.joint_q,
            cls._state_0.joint_qd,
            cls._state_0,
            cls._fk_mask,
        )

    @classmethod
    def sync_curves_to_usd(cls) -> None:
        """Update cable ``UsdGeomBasisCurves.points`` from Newton ``body_q``.

        Kit renders the authored USD ``BasisCurves`` asset for the waterhose.
        Write those curve points through pxr USD instead of Fabric array
        attributes: Kit 110 rejects Python-side Fabric data sources for
        ``Point3fArray`` in this path, which leaves the viewport in a bad state.
        """
        if cls._clone_physics_only or cls._state_0 is None or cls._state_0.body_q is None:
            return
        if not getattr(cls, "_cable_registry", None) or not cls._curves_dirty:
            return
        stage = get_current_stage()
        if stage is None:
            return

        try:
            from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415
        except Exception:
            return

        try:
            body_q = cls._state_0.body_q.numpy()
        except Exception:
            return

        import re

        xform_cache = UsdGeom.XformCache()
        for entry in cls._cable_registry:
            curve_template_path = entry.curve_prim_path or f"{entry.prim_path}/geometry/mesh"
            point_count = len(entry.edges) + 1
            if point_count < 2:
                continue
            for inst_idx, body_offset in enumerate(entry.body_offsets):
                if int(body_offset) + len(entry.edges) > body_q.shape[0]:
                    continue
                resolved = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), curve_template_path)
                resolved = re.sub(r"\.\*", str(inst_idx), resolved)
                curve_prim = stage.GetPrimAtPath(resolved)
                if not curve_prim or not curve_prim.IsValid():
                    continue

                world_to_local = xform_cache.GetLocalToWorldTransform(curve_prim).GetInverse()
                local_points = []
                for point_idx in range(len(entry.edges)):
                    pose = body_q[int(body_offset) + point_idx]
                    point_world = Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]))
                    local_points.append(world_to_local.Transform(point_world))

                tail_pose = body_q[int(body_offset) + len(entry.edges) - 1]
                tail_offset = _rotate_vector_by_quat_np(
                    np.asarray(tail_pose[3:7], dtype=np.float64),
                    np.asarray((0.0, 0.0, float(entry.last_edge_length)), dtype=np.float64),
                )
                tail_world_np = np.asarray(tail_pose[:3], dtype=np.float64) + tail_offset
                tail_world = Gf.Vec3d(float(tail_world_np[0]), float(tail_world_np[1]), float(tail_world_np[2]))
                local_points.append(world_to_local.Transform(tail_world))

                curve = UsdGeom.BasisCurves(curve_prim)
                curve.GetPointsAttr().Set(
                    Vt.Vec3fArray(
                        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in local_points]
                    )
                )
                vertex_counts_attr = curve.GetCurveVertexCountsAttr()
                vertex_counts = vertex_counts_attr.Get() if vertex_counts_attr else None
                if vertex_counts is None or sum(int(count) for count in vertex_counts) != point_count:
                    vertex_counts_attr.Set(Vt.IntArray([point_count]))
        cls._curves_dirty = False

    @classmethod
    def pre_render(cls) -> None:
        super().pre_render()
        cls.sync_curves_to_usd()

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create builder from USD stage with special treatment for deformable
        bodies, as these are not read from USD yet.

        Detects env Xforms (e.g. ``/World/Env_0``, ``/World/Env_1``) and builds
        each as a separate Newton world via ``begin_world``/``end_world``.
        Falls back to a flat ``add_usd`` when no env Xforms are found.

        TODO: Subclass should not override this method, once deformables
        supported on Newton import_usd, this can be unified with NewtonManager's
        implementation.
        """
        import re

        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)

        # Scan /World children for env-like Xforms (Env_0, env_1, ...)
        env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
        world_prim = stage.GetPrimAtPath("/World")
        env_paths: list[tuple[int, str]] = []
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                m = env_pattern.match(child.GetName())
                if m:
                    env_paths.append((int(m.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = ModelBuilder(up_axis=up_axis)

        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

        # Deformable sim/visual mesh paths must be skipped by ``add_usd``
        # so they don't get duplicated as static colliders.
        deformable_ignore_paths = cls._get_deformable_ignore_paths()

        if not env_paths:
            # No env Xforms — flat loading
            builder.add_usd(stage, ignore_paths=deformable_ignore_paths, schema_resolvers=schema_resolvers)

            # Run per-world builder hooks for the single world at origin.
            # Hooks include deformable and cable registries; each owns its own registration.
            if hasattr(cls, "_per_world_builder_hooks"):
                for hook in cls._per_world_builder_hooks:
                    hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        else:
            # Load everything except the env subtrees (ground plane, lights, etc.)
            ignore_paths = [path for _, path in env_paths] + deformable_ignore_paths
            builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)

            # Build a prototype from the first env (all envs assumed identical)
            _, proto_path = env_paths[0]
            proto = ModelBuilder(up_axis=up_axis)
            proto.add_usd(
                stage,
                root_path=proto_path,
                ignore_paths=deformable_ignore_paths,
                schema_resolvers=schema_resolvers,
            )

            # Inject registered sites into the proto before replication
            global_sites, proto_sites = cls._cl_inject_sites(builder, {proto_path: proto})
            global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}
            num_worlds = len(env_paths)
            local_site_map: dict[str, list[list[int]]] = {}
            site_entries = proto_sites.get(id(proto), {})

            # Add each env as a separate Newton world
            xform_cache = UsdGeom.XformCache()
            for col, (_, env_path) in enumerate(env_paths):
                builder.begin_world()
                offset = builder.shape_count
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                pos = (translation[0], translation[1], translation[2])
                quat = (
                    rotation.GetImaginary()[0],
                    rotation.GetImaginary()[1],
                    rotation.GetImaginary()[2],
                    rotation.GetReal(),
                )
                builder.add_builder(proto, xform=wp.transform(pos, quat))
                for label, proto_shape_indices in site_entries.items():
                    if label not in local_site_map:
                        local_site_map[label] = [[] for _ in range(num_worlds)]
                    for proto_shape_idx in proto_shape_indices:
                        local_site_map[label][col].append(offset + proto_shape_idx)

                # Run per-world builder hooks for this world (deformables, cables, ...).
                if hasattr(cls, "_per_world_builder_hooks"):
                    for hook in cls._per_world_builder_hooks:
                        hook(builder, col, list(pos), list(quat))

                builder.end_world()

            NewtonManager._cl_site_index_map = {
                **global_site_map,
                **{label: (None, per_world) for label, per_world in local_site_map.items()},
            }
            NewtonManager._num_envs = len(env_paths)

        # run vbd builder coloring
        builder.color()

        cls.set_builder(builder)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct :class:`SolverVBD` and populate the base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        kwargs = cls._filter_solver_kwargs(SolverVBD, solver_cfg)
        NewtonManager._solver = SolverVBD(model, **kwargs)
        if not getattr(solver_cfg, "rigid_joint_hard", True):
            for joint_index in range(int(getattr(model, "joint_count", 0))):
                NewtonManager._solver.set_joint_constraint_mode(joint_index, hard=False)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        # Guard against Newton versions where ``SolverVBD`` did not initialize
        # ``particle_enable_self_contact`` when ``model.particle_count == 0``
        # (rigid-body-only or cable-only scenes).  In that case ``rebuild_bvh``
        # would raise ``AttributeError``; the call is a no-op anyway since there
        # are no particles to rebuild BVH for.
        if hasattr(cls._solver, "rebuild_bvh") and getattr(cls._solver, "particle_enable_self_contact", False):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
