# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable / 1D-rod asset class, registry entry, and replicate-hook plumbing.

The structure mirrors :mod:`isaaclab_contrib.deformable.deformable_object`. Cables
differ from deformables in two respects only:

1. They subclass :class:`Articulation` (not :class:`BaseDeformableObject`) because
   ``newton.ModelBuilder.add_rod_graph`` produces a Newton articulation, and
   ``ArticulationView`` already covers state read/write.
2. Their authored curve and material data is read once from USD and consumed
   in-memory by the cable replicate hook.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .cable_object_cfg import CableObjectCfg


def _rotate_vector_by_quat_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by an ``xyzw`` quaternion."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    q_vec = quat[:3]
    t = 2.0 * np.cross(q_vec, vec)
    return vec + quat[3] * t + np.cross(q_vec, t)


@dataclass
class CableRegistryEntry:
    """Mutable bridge between :class:`CableObject` and the replicate hook.

    Populated by :meth:`CableObject._register_cable` (reads the spawned
    ``UsdGeomBasisCurves`` and its Newton physics material) and consumed by
    :func:`add_cable_entry_to_builder`. Material-field semantics and defaults
    mirror :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`.
    """

    prim_path: str
    node_positions: list[wp.vec3]
    edges: list[tuple[int, int]]
    radius: float
    curve_prim_path: str = ""

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    stretch_stiffness: float = 1.0e9
    bend_stiffness: float = 0.0
    stretch_damping: float = 0.0
    bend_damping: float = 0.0
    density: float = 1500.0

    # Filled by :func:`add_cable_entry_to_builder`.
    body_offsets: list[int] = field(default_factory=list)
    last_edge_length: float = 0.0


def add_cable_entry_to_builder(
    builder,
    entry: CableRegistryEntry,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
    cable_idx: int = 0,
) -> None:
    """Add one cable to a Newton ``ModelBuilder`` for one environment.

    Composes the env transform with the cable's init transform and applies it to
    each control point, then calls :meth:`newton.ModelBuilder.add_rod_graph` with
    the explicit stiffness / damping / density fields stored on the entry.
    Density flows through :class:`newton.ModelBuilder.ShapeConfig` so Newton
    computes per-segment mass from ``density * pi * r^2 * segment_length``. The
    articulation is labelled ``"{entry.prim_path}/cable"`` so the cloner's
    ``_rename_builder_labels`` rewrites the source prefix to each env's
    destination prefix during replication.

    All capsules of this cable share a unique negative ``collision_group``
    (``-(1 + cable_idx)``), which disables segment-vs-segment self-collision while
    still letting them collide with the ground and other cables (Newton's group
    rule: same negative group = filtered, negative-vs-positive = collides).

    Args:
        builder: The Newton ``ModelBuilder``.
        entry: Registry entry describing the cable's geometry and material.
        env_idx: Zero-based environment (world) index.
        env_position: World translation ``[x, y, z]`` [m] for this environment.
        env_rotation: World orientation as quaternion ``(x, y, z, w)`` for this environment.
        cable_idx: Zero-based index of this cable within
            :attr:`SimulationManager._cable_registry`. Used to assign a unique
            negative ``shape_collision_group`` per cable so segments don't
            self-collide.
    """
    if env_idx == 0:
        entry.body_offsets.clear()
        entry.last_edge_length = 0.0

    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]),
        float(env_rotation[1]),
        float(env_rotation[2]),
        float(env_rotation[3]),
    )
    init_pos = wp.vec3(float(entry.init_pos[0]), float(entry.init_pos[1]), float(entry.init_pos[2]))
    init_rot = wp.quat(
        float(entry.init_rot[0]),
        float(entry.init_rot[1]),
        float(entry.init_rot[2]),
        float(entry.init_rot[3]),
    )

    # Compose: world = env_T ∘ init_T ∘ local
    composed_pos = env_pos + wp.quat_rotate(env_rot, init_pos)
    composed_rot = env_rot * init_rot

    world_nodes: list[wp.vec3] = []
    for node in entry.node_positions:
        rotated = wp.quat_rotate(composed_rot, node)
        world_nodes.append(composed_pos + rotated)

    # Builder hooks run after the task has installed its NewtonShapeCfg defaults.
    # Preserve those contact values for procedurally authored capsules instead of
    # falling back to Newton's standalone ShapeConfig defaults (notably a 100 mm
    # detection gap and 100 N.s/m damping on current Newton).
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = float(entry.density)
    # Unique negative collision group → cable's own capsules don't collide with
    # each other (Newton: same negative group is filtered), while still colliding
    # with the ground and other cables (negative-vs-positive collides).
    shape_cfg.collision_group = -(1 + cable_idx)

    # ``label`` is load-bearing: Newton suffixes ``_articulation`` to produce
    # ``{prim_path}/cable_articulation``, which is the path :class:`ArticulationView`
    # searches for per env after the cloner rewrites the source prefix.
    entry.body_offsets.append(builder.body_count)
    expanded_prim_path = entry.prim_path.replace("env_.*", f"env_{env_idx}")
    builder.add_rod_graph(
        node_positions=world_nodes,
        edges=entry.edges,
        radius=entry.radius,
        body_frame_origin="start",
        cfg=shape_cfg,
        stretch_stiffness=entry.stretch_stiffness,
        stretch_damping=entry.stretch_damping,
        bend_stiffness=entry.bend_stiffness,
        bend_damping=entry.bend_damping,
        label=f"{expanded_prim_path}/cable",
        wrap_in_articulation=True,
    )
    if env_idx == 0:
        u, v = entry.edges[-1]
        entry.last_edge_length = float(wp.length(entry.node_positions[v] - entry.node_positions[u]))


def add_registered_cables_to_builder(
    builder,
    world_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Loop function for ``_per_world_builder_hooks``.

    Iterates :attr:`SimulationManager._cable_registry` and calls
    :func:`add_cable_entry_to_builder` for each registered cable.
    Mirrors :func:`isaaclab_contrib.deformable.deformable_object.add_registered_deformables_to_builder`.
    """
    for cable_idx, entry in enumerate(SimulationManager._cable_registry):
        add_cable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation, cable_idx=cable_idx)


def install_cable_builder_hooks() -> None:
    """Set up the cable registry and per-world hook on ``SimulationManager``.

    Resets ``_cable_registry`` to an empty list on each call — install is intended
    to be called once per scene setup, not per asset.

    Mirrors :func:`isaaclab_contrib.deformable.deformable_object.install_deformable_builder_hooks`.
    """
    SimulationManager._cable_registry = []
    if not hasattr(SimulationManager, "_per_world_builder_hooks"):
        SimulationManager._per_world_builder_hooks = []
    if add_registered_cables_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.append(add_registered_cables_to_builder)
    if not hasattr(SimulationManager, "_post_replicate_hooks"):
        SimulationManager._post_replicate_hooks = []
    if color_registered_cables not in SimulationManager._post_replicate_hooks:
        SimulationManager._post_replicate_hooks.append(color_registered_cables)
    SimulationManager.register_pre_render_callback("cable_curve_sync", sync_registered_cable_curves_to_usd)


class CableObject(Articulation):
    """Cable / 1D-rod asset (Newton backend).

    Subclasses :class:`Articulation` so the cable's per-segment poses and
    per-cable-joint state are exposed via :class:`ArticulationData` with no
    parallel data class.

    Override surface beyond the base:

    - :meth:`__init__` defers to the base ``__init__`` and then calls
      :meth:`_register_cable` (mirroring :meth:`DeformableObject._register_deformable`),
      which builds a :class:`CableRegistryEntry` from cfg and appends it to the
      cable registry. Caller must have called :func:`install_cable_builder_hooks`
      before constructing any :class:`CableObject` (typical: from a solver manager
      init, mirroring how the deformable contrib package wires things up).
    - :meth:`reset` snaps each environment's cable bodies back to the
      rest pose stored in ``model.body_q``.
    """

    cfg: CableObjectCfg

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        # Read the cable's centerline / material from cfg and register in the
        # cable registry. Mirrors :meth:`DeformableObject._register_deformable`.
        self._registry_entry = self._register_cable()

    def _register_cable(self) -> CableRegistryEntry:
        """Read cable geometry + material from the spawned USD prim and register on
        :attr:`SimulationManager._cable_registry`.

        Mirrors :meth:`DeformableObject._register_deformable`:

        1. Locate the spawned template prim (via ``cfg.spawn.spawn_path`` or
           ``cfg.prim_path``).
        2. Walk the template prim's descendants and find the single
           ``UsdGeomBasisCurves`` prim, then read its ``points`` and ``widths``
           attributes. Cable geometry is loaded from an authored USD through
           :class:`~isaaclab.sim.spawners.UsdFileCfg`.
        3. Bake the template prim's xform into the per-node positions so the
           replicate hook only needs to apply the env transform.
        4. Look up the bound Newton cable physics material on the curve prim
           and read each ``newton:*`` attribute into the entry.

        Returns:
            The registry entry (also appended to ``SimulationManager._cable_registry``).

        Raises:
            ValueError: If the template prim has no ``UsdGeomBasisCurves``
                descendant, or the curve is missing its ``widths`` attribute.
            NotImplementedError: If more than one ``UsdGeomBasisCurves``
                descendant is found under the template prim — multi-curve
                cables under a single :class:`CableObject` are not supported.
            RuntimeError: If the template prim cannot be located, or the active
                Newton solver is not a VBD variant (only :class:`VBDSolverCfg`
                and its coupled variants register the cable builder hooks; no
                other Newton solver steps :attr:`newton.JointType.CABLE`).

        Note:
            ``pxr`` imports are deferred to this method (not module level) so
            that ``resolve_task_config`` can import the env-cfg module before
            Kit starts without polluting the ``pxr`` module cache.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        if not hasattr(SimulationManager, "_cable_registry"):
            raise RuntimeError(
                "CableObject can only be simulated under the Newton VBD solver"
                " (`VBDSolverCfg` or one of its coupled variants:"
                " `CoupledMJWarpVBDSolverCfg`, `CoupledFeatherstoneVBDSolverCfg`)."
                " The cable registry is installed by the VBD manager's `initialize()`"
                " hook via `install_cable_builder_hooks()`, and `JointType.CABLE`"
                " is not stepped by any other Newton solver. Switch the solver cfg"
                " or remove the CableObject from the scene."
            )

        if self.cfg.spawn is None:
            raise ValueError(
                f"CableObjectCfg(prim_path='{self.cfg.prim_path}') has no `spawn` configuration."
                " CableObject requires an `UsdFileCfg` (or compatible USD-loading cfg) containing"
                " one authored `UsdGeomBasisCurves` descendant; pass it via `CableObjectCfg.spawn`."
            )

        # Resolve the spawned template prim. ``spawn_path`` is set by InteractiveScene's
        # template-based cloning flow; falls back to ``prim_path`` for direct envs that
        # spawn straight at the cloned regex.
        lookup_path = self.cfg.spawn.spawn_path if self.cfg.spawn.spawn_path is not None else self.cfg.prim_path
        template_prim = sim_utils.find_first_matching_prim(lookup_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find cable template prim for expression: '{lookup_path}'.")
        template_prim_path = template_prim.GetPrimPath()

        # Discover the authored BasisCurves by descendant traversal so the curve
        # may live anywhere within the loaded USD hierarchy.
        stage = template_prim.GetStage()
        curve_prims = [
            descendant for descendant in Usd.PrimRange(template_prim) if descendant.GetTypeName() == "BasisCurves"
        ]
        if not curve_prims:
            raise ValueError(f"No UsdGeomBasisCurves prim found under '{template_prim_path}'.")
        if len(curve_prims) > 1:
            paths = ", ".join(str(p.GetPrimPath()) for p in curve_prims)
            raise NotImplementedError(
                f"Found {len(curve_prims)} BasisCurves prims under '{template_prim_path}' ({paths}); "
                "multi-curve cables under a single CableObject are not supported yet."
            )
        curve_prim = curve_prims[0]
        curves = UsdGeom.BasisCurves(curve_prim)

        # Bake the curve prim's xform into the per-node positions so the replicate
        # hook only needs to apply the env transform.
        xform_cache = UsdGeom.XformCache()
        curve_to_parent_frame = (
            xform_cache.GetLocalToWorldTransform(curve_prim)
            * xform_cache.GetLocalToWorldTransform(template_prim.GetParent()).GetInverse()
        )
        raw_points = curves.GetPointsAttr().Get()
        node_positions: list[wp.vec3] = []
        for p in raw_points:
            q = curve_to_parent_frame.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            node_positions.append(wp.vec3(float(q[0]), float(q[1]), float(q[2])))

        # Read the capsule width. Every authored control point must use the same width.
        raw_widths = curves.GetWidthsAttr().Get()
        if raw_widths is None or len(raw_widths) == 0:
            raise ValueError(f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `widths` attribute.")
        widths_list = [float(w) for w in raw_widths]
        if max(widths_list) - min(widths_list) > 1e-9:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' has non-uniform `widths`"
                f" (min={min(widths_list)}, max={max(widths_list)}); tapered cables are not supported."
                " Author a constant width across all control points."
            )
        radius = widths_list[0] / 2.0

        # Read edge topology from the authored curve prim's ``int2[] connections`` attribute.
        connections_attr = curve_prim.GetAttribute("connections")
        if not connections_attr.IsValid() or connections_attr.Get() is None:
            raise ValueError(
                f"UsdGeomBasisCurves at '{curve_prim.GetPrimPath()}' is missing the `connections`"
                " attribute (expected `int2[]` listing each edge as a pair of control-point indices)."
                " Author this attribute on the curve prim in the source USD."
            )
        edges = [(int(e[0]), int(e[1])) for e in connections_attr.Get()]

        # Look up the bound Newton cable physics material via the standard
        # MaterialBindingAPI on the curve prim. The source USD must apply
        # :class:`UsdPhysics.CollisionAPI` and bind a
        # :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`.
        material_targets = (
            UsdShade.MaterialBindingAPI(curve_prim).GetDirectBindingRel("physics").GetTargets()
            if curve_prim.HasAPI(UsdShade.MaterialBindingAPI)
            else []
        )
        material_prim = None
        for mat_path in material_targets:
            mat_prim = stage.GetPrimAtPath(mat_path)
            if mat_prim.GetAttribute("newton:density").IsValid():
                material_prim = mat_prim
                break
        if material_prim is None:
            has_collision_api = curve_prim.HasAPI(UsdPhysics.CollisionAPI)
            hint = (
                ""
                if has_collision_api
                else (
                    " Hint: author `UsdPhysics.CollisionAPI` on the curve before binding its"
                    " Newton cable physics material."
                )
            )
            raise ValueError(
                f"Could not find a Newton cable physics material bound to '{curve_prim.GetPrimPath()}'." + hint
            )

        def _get_material_attr(name: str, default):
            attr = material_prim.GetAttribute(name)
            return attr.Get() if attr.IsValid() else default

        stretch_stiffness = _get_material_attr("newton:stretchStiffness", CableRegistryEntry.stretch_stiffness)
        bend_stiffness = _get_material_attr("newton:bendStiffness", CableRegistryEntry.bend_stiffness)
        stretch_damping = _get_material_attr("newton:stretchDamping", CableRegistryEntry.stretch_damping)
        bend_damping = _get_material_attr("newton:bendDamping", CableRegistryEntry.bend_damping)
        density = _get_material_attr("newton:density", CableRegistryEntry.density)

        # init_pos/init_rot default to identity — the template xform is already baked
        # into ``node_positions`` above, so the replicate hook only applies the env
        # transform. Matches DeformableObject._register_deformable.
        entry = CableRegistryEntry(
            prim_path=self.cfg.prim_path,
            curve_prim_path=str(curve_prim.GetPrimPath()),
            node_positions=node_positions,
            edges=edges,
            radius=radius,
            stretch_stiffness=float(stretch_stiffness),
            bend_stiffness=float(bend_stiffness),
            stretch_damping=float(stretch_damping),
            bend_damping=float(bend_damping),
            density=float(density),
        )
        SimulationManager._cable_registry.append(entry)
        return entry

    def reset(
        self,
        env_ids: Sequence[int] | slice | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Snap each env's cable bodies back to the spawn pose.

        Restores four arrays per-env body slice. ``state.body_q`` and
        ``solver.body_q_prev`` come from :attr:`Model.body_q` (the rest-pose
        template that :class:`SolverVBD` itself reads at init);
        ``state.body_qd`` and ``solver.body_inertia_q`` are zeroed.
        ``body_q_prev`` is load-bearing — AVBD computes implicit velocity as
        ``(body_q - body_q_prev) / dt``, so without this the snap-back
        produces ~700 m/s spurious velocities.

        Joint state and AVBD penalty/Dahl buffers are intentionally not
        touched: they are global to the world (penalty ``k``) or would need
        joint offsets in the registry (Dahl, ``joint_q``); in practice the
        body-side reset is sufficient to keep post-reset dynamics bounded.

        Args:
            env_ids: Environment indices to reset. ``None`` means all.
            env_mask: Parent-class compatibility; unused.
        """
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if not getattr(self, "_is_initialized", False) or SimulationManager._solver is None:
            return
        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()
        body_offsets = self._registry_entry.body_offsets
        n = len(self._registry_entry.edges)
        # Per-call zero buffer for velocity slices (one segment chain wide).
        zero_qd = wp.zeros(n, dtype=state.body_qd.dtype, device=state.body_qd.device)
        env_iter = range(len(body_offsets)) if env_ids is None or env_ids == slice(None) else list(env_ids)
        for env_idx in env_iter:
            offset = int(body_offsets[env_idx])
            wp.copy(dest=state.body_q, src=model.body_q, dest_offset=offset, src_offset=offset, count=n)
            wp.copy(dest=state.body_qd, src=zero_qd, dest_offset=offset, count=n)
        solver = SimulationManager._solver
        if hasattr(solver, "body_q_prev") and hasattr(solver, "body_inertia_q"):
            zero_q = wp.zeros(n, dtype=solver.body_inertia_q.dtype, device=solver.body_inertia_q.device)
            for env_idx in env_iter:
                offset = int(body_offsets[env_idx])
                wp.copy(dest=solver.body_q_prev, src=model.body_q, dest_offset=offset, src_offset=offset, count=n)
                wp.copy(dest=solver.body_inertia_q, src=zero_q, dest_offset=offset, count=n)


def color_registered_cables(builder) -> None:
    """Color the final Newton builder when procedural cables were registered."""
    if SimulationManager._cable_registry:
        builder.color()


def sync_registered_cable_curves_to_usd() -> None:
    """Update registered cable ``UsdGeomBasisCurves.points`` from Newton body poses.

    Cable segment bodies are created by builder hooks and intentionally do not
    have authored USD prims. Kit renders the authored cable curve USD, so the
    curve control points must be updated explicitly from the live Newton rod
    segment body transforms before each render.
    """
    if getattr(SimulationManager, "_clone_physics_only", False):
        return
    state = SimulationManager._state_0
    if state is None or state.body_q is None:
        return
    registry = getattr(SimulationManager, "_cable_registry", None)
    if not registry:
        return

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    if stage is None:
        return

    try:
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415
    except Exception:
        return

    try:
        body_q = state.body_q.numpy()
    except Exception:
        return

    xform_cache = UsdGeom.XformCache()
    for entry in registry:
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
                logger.debug("[cable_curve_sync] curve prim not found at %s", resolved)
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
                Vt.Vec3fArray([Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in local_points])
            )
            vertex_counts_attr = curve.GetCurveVertexCountsAttr()
            vertex_counts = vertex_counts_attr.Get() if vertex_counts_attr else None
            if vertex_counts is None or sum(int(count) for count in vertex_counts) != point_count:
                vertex_counts_attr.Set(Vt.IntArray([point_count]))
