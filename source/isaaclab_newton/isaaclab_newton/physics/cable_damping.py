# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import Newton cable-joint damping authored on USD curve materials."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

from newton import JointType, ModelBuilder

from pxr import Usd, UsdShade

_DAMPING_ATTRS = (
    "stretchDamping",
    "shearDamping",
    "bendDamping",
    "twistDamping",
)
_STIFFNESS_ATTRS = ("stretchStiffness", "shearStiffness", "bendStiffness", "twistStiffness")


def _bound_physics_material(prim: Usd.Prim) -> Usd.Prim:
    """Return the effectively bound physics material, including inherited bindings."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    if material:
        material_prim = material.GetPrim()
        if material_prim.IsValid():
            return material_prim
    return Usd.Prim()


def _read_nonnegative(material_prim: Usd.Prim, name: str, cable_path: str) -> tuple[float | None, bool]:
    """Read one authored finite, nonnegative ``physics:*`` value."""
    attr = material_prim.GetAttribute(f"physics:{name}")
    if not attr or not attr.HasAuthoredValue():
        return None, False
    value = float(attr.Get())
    if not math.isfinite(value) or value < 0.0:
        warnings.warn(
            f"{cable_path}: invalid physics:{name} {value!r} on {material_prim.GetPath()} "
            "(expected a finite, nonnegative value); ignoring it.",
            stacklevel=3,
        )
        return None, False
    return value, True


def _resolve_material_damping(stage: Usd.Stage, cable_path: str) -> tuple[float, float, float, float] | None:
    """Resolve the four Newton cable damping slots from a curve's bound material."""
    cable_prim = stage.GetPrimAtPath(cable_path)
    if not cable_prim.IsValid():
        return None
    material_prim = _bound_physics_material(cable_prim)
    if not material_prim.IsValid():
        return None

    damping_values: list[float | None] = []
    damping_authored: list[bool] = []
    for name in _DAMPING_ATTRS:
        value, authored = _read_nonnegative(material_prim, name, cable_path)
        damping_values.append(value)
        damping_authored.append(authored)
    if not any(damping_authored):
        return None

    stiffness_authored = [_read_nonnegative(material_prim, name, cable_path)[1] for name in _STIFFNESS_ATTRS]
    stretch = damping_values[0] if damping_authored[0] else 0.0
    shear = damping_values[1] if damping_authored[1] else (stretch if not stiffness_authored[1] else 0.0)
    bend = damping_values[2] if damping_authored[2] else 0.0
    twist = damping_values[3] if damping_authored[3] else (bend if not stiffness_authored[3] else 0.0)
    return float(stretch), float(shear), float(bend), float(twist)


def _graph_cable_joints(builder: ModelBuilder, body_indices: set[int]) -> list[int]:
    """Find the cable joints internal to a welded rod-graph component."""
    cable_type = int(JointType.CABLE)
    return [
        joint
        for joint, (joint_type, parent, child) in enumerate(
            zip(builder.joint_type, builder.joint_parent, builder.joint_child, strict=True)
        )
        if int(joint_type) == cable_type and int(parent) in body_indices and int(child) in body_indices
    ]


def _set_joint_damping(
    builder: ModelBuilder, joints: Sequence[int], damping: tuple[float, float, float, float]
) -> None:
    """Set Newton's stretch/shear/bend/twist cable constraint damping slots."""
    cable_type = int(JointType.CABLE)
    for joint in joints:
        if int(builder.joint_type[joint]) != cable_type:
            continue
        if tuple(builder.joint_dof_dim[joint]) != (2, 2):
            raise RuntimeError(f"Newton cable joint {joint} does not have the expected 2 linear + 2 angular DOFs.")
        dof_start = int(builder.joint_qd_start[joint])
        builder.joint_target_kd[dof_start : dof_start + 4] = damping


def apply_cable_damping_from_usd(builder: ModelBuilder, stage: Usd.Stage, import_result: dict[str, Any]) -> None:
    """Apply authored cable damping to an imported Newton builder.

    Newton stores cable stretch, shear, bend, and twist damping in four passive
    constraint slots on each ``JointType.CABLE`` joint. The VBD solver evaluates
    those constraints from relative parent/child motion, so this bridge does not
    add an actuator or world/body drag.

    Args:
        builder: Newton builder populated by ``ModelBuilder.add_usd``.
        stage: USD stage from which the builder was imported.
        import_result: Import result produced with ``return_deformable_results=True``.
    """
    path_cable_map = import_result.get("path_cable_map", {})
    path_cable_attrs = import_result.get("path_cable_attrs", {})
    if not path_cable_map:
        return

    groups: dict[tuple[str, str], list[str]] = {}
    for path in path_cable_map:
        graph_component = path_cable_attrs.get(path, {}).get("graph_component")
        key = ("graph", str(graph_component)) if graph_component is not None else ("curve", path)
        groups.setdefault(key, []).append(path)

    for (kind, _name), paths in groups.items():
        representative_path = min(paths)
        damping_by_path = {path: _resolve_material_damping(stage, path) for path in paths}
        damping = damping_by_path[representative_path]

        if kind == "graph":
            if any(path_damping != damping for path_damping in damping_by_path.values()):
                warnings.warn(
                    f"cable graph '{_name}' has differing damping materials; using "
                    f"'{representative_path}' for the whole component.",
                    stacklevel=2,
                )
            if damping is None:
                continue
            body_indices = {int(body) for path in paths for body in path_cable_map[path][0]}
            joints = _graph_cable_joints(builder, body_indices)
        else:
            if damping is None:
                continue
            joints = [int(joint) for joint in path_cable_map[representative_path][1]]
        _set_joint_damping(builder, joints, damping)


def add_usd_with_cable_damping(
    builder: ModelBuilder,
    stage: Usd.Stage,
    **kwargs,
) -> dict[str, Any]:
    """Import a USD stage and bridge cable-material damping into Newton joints.

    This compatibility bridge can be removed once Newton's native USD cable
    importer consumes the four ``physics:*Damping`` material attributes.

    Args:
        builder: Newton builder to populate.
        stage: USD stage to import.
        **kwargs: Arguments forwarded to :meth:`newton.ModelBuilder.add_usd`.

    Returns:
        The Newton USD import result, including deformable result mappings.
    """
    kwargs["return_deformable_results"] = True
    import_result = builder.add_usd(stage, **kwargs)
    apply_cable_damping_from_usd(builder, stage, import_result)
    return import_result
