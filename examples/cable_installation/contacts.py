# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only native Newton contact measurements for the scripted example."""

from dataclasses import dataclass

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.utils.math import quat_apply


@dataclass(frozen=True)
class GraspContacts:
    """Measured connector contacts and maximum positive penetration [m]."""

    left: bool
    right: bool
    penetration_m: float
    overflow: bool
    deepest_pair: str
    pair_penetrations: tuple[tuple[str, float], ...] = ()
    thumb_latch: bool = False


def grasp_contact_failure(contacts: GraspContacts, grasp_penetration_limit: float = 0.003) -> str | None:
    """Check scene collisions without confusing allowed compliant pad overlap [m]."""
    if contacts.overflow:
        return f"Contact buffer overflow: {contacts}"
    pairs = contacts.pair_penetrations or ((contacts.deepest_pair, contacts.penetration_m),)
    for pair, depth in pairs:
        penetration_limit = 0.003 if "/Robot/" in pair and "/Cable/" in pair else 0.002
        labels = pair.split(" | ")
        if any("/Robot/" in label and label.endswith("/hydroelastic_grasp") for label in labels) and any(
            label.endswith("/Cable/head/grip") for label in labels
        ):
            penetration_limit = grasp_penetration_limit
        if depth > penetration_limit:
            return f"Invalid contact: {pair}: {depth:.6f} m exceeds {penetration_limit:.6f} m"
    return None


class GraspContactReader:
    """Cache static shape topology and measure contacts from the current native state.

    The hand and connector are both owned by MJWarp. Internal fixed-component
    overlaps and static/static pairs are excluded: they cannot cause relative motion.

    Args:
        env_index: Environment whose contact labels are inspected.
        opposing_body_patterns: Thumb and supporting-finger body-label patterns.
    """

    def __init__(
        self,
        env_index: int = 0,
        opposing_body_patterns: tuple[tuple[str, ...], tuple[str, ...]] = (
            ("right_thumb_elastomer",),
            ("right_index_elastomer", "right_middle_elastomer"),
        ),
    ):
        model = NewtonManager.get_model()
        self.prefix = f"/World/envs/env_{env_index}/"
        self.opposing_body_patterns = opposing_body_patterns
        self.shape_body = wp.to_torch(model.shape_body).long()
        self.shape_labels = tuple(str(label) for label in model.shape_label)
        body_labels = tuple(str(label) for label in model.body_label)
        self.shape_body_labels = tuple("" if body < 0 else body_labels[body] for body in self.shape_body.cpu().tolist())
        inverse_mass = model.body_inv_mass.numpy()
        parents = [0, *(0 if mass == 0 else body + 1 for body, mass in enumerate(inverse_mass))]

        def root(body: int) -> int:
            while parents[body] != body:
                body = parents[body]
            return body

        for kind, parent, child in zip(
            model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy(), strict=True
        ):
            if kind == newton.JointType.FIXED:
                parents[root(int(child) + 1)] = root(int(parent) + 1)
        self.shape_groups = tuple(0 if body < 0 else root(body + 1) for body in self.shape_body.cpu().tolist())

    def read(self) -> GraspContacts:
        """Read bilateral housing contact, separate thumb/clip contact and overlap [m]."""
        contacts = NewtonManager.get_contacts()
        if contacts is None:
            raise RuntimeError("Cable installation requires the native collision contact buffer.")
        count = int(wp.to_torch(contacts.rigid_contact_count)[0])
        overflow = count > contacts.rigid_contact_max
        count = min(count, contacts.rigid_contact_max)
        shape_ids = [wp.to_torch(getattr(contacts, f"rigid_contact_shape{i}"))[:count].long() for i in (0, 1)]
        poses = wp.to_torch(NewtonManager.get_state_0().body_q)
        points = []
        for i, shapes in enumerate(shape_ids):
            point = wp.to_torch(getattr(contacts, f"rigid_contact_point{i}"))[:count].clone()
            bodies = self.shape_body[shapes]
            mask = bodies >= 0
            point[mask] = poses[bodies[mask], :3] + quat_apply(poses[bodies[mask], 3:], point[mask])
            points.append(point)
        normals = wp.to_torch(contacts.rigid_contact_normal)[:count]
        separation = ((points[1] - points[0]) * normals).sum(-1)
        for i in (0, 1):
            separation -= wp.to_torch(getattr(contacts, f"rigid_contact_margin{i}"))[:count]
        # Zero-normal entries are placeholders, not load-bearing contacts.
        active = (separation <= 0.0002) & (torch.linalg.vector_norm(normals, dim=-1) > 0.9)
        local_shapes = torch.stack(shape_ids, dim=1)[active].cpu().tolist()
        depths = (-separation[active]).cpu().tolist()
        left = right = thumb_latch = False
        penetration = 0.0
        deepest = ""
        pair_penetrations: dict[str, float] = {}
        for shapes, depth in zip(local_shapes, depths, strict=True):
            if self.shape_groups[shapes[0]] == self.shape_groups[shapes[1]]:
                continue
            labels = [self.shape_labels[shape] for shape in shapes]
            pair = " | ".join(labels)
            if self.prefix not in pair:
                continue
            if any(self.shape_body_labels[shape] == f"{self.prefix}Cable/head" for shape in shapes):
                left |= any(pattern in pair for pattern in self.opposing_body_patterns[0])
                right |= any(pattern in pair for pattern in self.opposing_body_patterns[1])
            if any(self.shape_body_labels[shape] == f"{self.prefix}Cable/latch" for shape in shapes):
                thumb_latch |= any(pattern in pair for pattern in self.opposing_body_patterns[0])
            if depth > penetration:
                penetration, deepest = depth, pair
            if depth > 0.0:
                pair_penetrations[pair] = max(depth, pair_penetrations.get(pair, 0.0))
        return GraspContacts(left, right, penetration, overflow, deepest, tuple(pair_penetrations.items()), thumb_latch)
