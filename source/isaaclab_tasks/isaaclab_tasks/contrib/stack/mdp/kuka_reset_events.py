# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KUKA-Allegro reset-state generation for order-invariant cube stacking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg

from .kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_FULL_HAND_PALM_TO_HELD_CUBE_QUATERNIONS_XYZW,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
    KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT,
    KUKA_ALLEGRO_STACK_ARM_POSES,
    kuka_allegro_grasp_pair_pose,
    kuka_allegro_pinch_pose,
    matrix_from_quaternion_xyzw,
    quaternion_xyzw_from_matrix,
    solve_kuka_allegro_reset_ik,
)
from .reset_events import (
    _PREGRASP_POSE_INDEX,
    _RECIPE_TARGET_POTENTIAL,
    StackResetRecipe,
    StackResetStateTable,
    _oriented_cube_pair_intersections,
    _segment_oriented_box_intersections,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class KukaAllegroResetStateTable(StackResetStateTable):
    """Physics-validated reset manifold for the fully actuated KUKA-Allegro task.

    The immutable bank contains exactly 65,536 rows. Each non-table recipe has
    6,144 rows over 256 layouts and broad wrist-orientation bins, while table
    starts have 16,384 independently scattered layouts. Reset-authored grasps
    use the validated index/thumb side pinch; the policy still controls and
    observes all 16 hand joints. A batched DLS solve holds the pair center
    fixed while applying each requested wrist rotation.
    """

    _ARM_WORKSPACE_LOWER = KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER
    _ARM_WORKSPACE_UPPER = KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER
    _ARM_JOINT_NAMES = "iiwa7_joint_(1|2|3|4|5|6|7)"
    _HAND_JOINT_NAMES = KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES
    _EXPECTED_HAND_JOINTS = 16
    _ARM_POSE_VALUES = KUKA_ALLEGRO_STACK_ARM_POSES

    _CUBE_HEIGHT = KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH
    _TABLE_HEIGHT = KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT
    _ROWS_PER_RECIPE = 6144
    _TABLE_ROWS = 16384
    _SEMANTIC_LAYOUT_COUNT = 256
    _EXPECTED_ROW_COUNT = 65536

    _GRASP_PAIR_OPEN_COMMANDS = KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS
    _GRASP_PAIR_RESET_CLOSED_COMMANDS = KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_COMMANDS
    _GRASP_PAIR_TOOL_OFFSETS = KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS
    _PALM_TO_HELD_CUBE_QUATERNIONS_XYZW = KUKA_ALLEGRO_FULL_HAND_PALM_TO_HELD_CUBE_QUATERNIONS_XYZW

    _RESET_CLEARANCE_MARGIN = 2.5e-4
    _FIRST_PLACE_SUPPORT_MARGIN = 0.015
    _SECOND_PLACE_SUPPORT_MARGIN = 0.010
    _PAIR_READY_CLEARANCE_ARC_HEIGHT = 0.10
    _LAYOUT_MINIMUM_SEPARATION = 0.12
    _PICK_PREGRASP_TOP_CLEARANCE = 0.1195
    _PICK_CONTACT_CENTER_LIFT = 0.0035
    _FIRST_TRANSPORT_BOTTOM_CLEARANCE = 0.1195
    _FIRST_PLACE_CENTER_CLEARANCE = 0.080
    _SECOND_PLACE_CENTER_CLEARANCE = 0.070
    _PAIR_READY_SOURCE_TOP_CLEARANCE = 0.1195
    _TABLE_APPROACH_TOP_CLEARANCE = 0.1195
    _TABLE_APPROACH_HEIGHT_RANGE = 0.050
    _RING_TRANSPORT_BOTTOM_CLEARANCE = 0.0745
    _SEMANTIC_X_LOWER = 0.48
    _SEMANTIC_X_EXTENT = 0.14
    _SEMANTIC_Y_LOWER = -0.14
    _SEMANTIC_Y_EXTENT = 0.28
    _TABLE_X_LOWER = _SEMANTIC_X_LOWER
    _TABLE_X_EXTENT = _SEMANTIC_X_EXTENT
    _TABLE_Y_LOWER = _SEMANTIC_Y_LOWER
    _TABLE_Y_EXTENT = _SEMANTIC_Y_EXTENT
    _FINAL_IK_POSITION_RESIDUAL_LIMIT = 8.0e-4
    _FINAL_CLEARANCE_MAX_PASSES = 12
    _GLOBAL_TILT_LIMIT = math.radians(45.0)
    _SECOND_TRANSPORT_BOTTOM_CLEARANCE = (
        1.5 * _CUBE_HEIGHT
        + 0.5 * _CUBE_HEIGHT * (math.cos(_GLOBAL_TILT_LIMIT) + math.sqrt(2.0) * math.sin(_GLOBAL_TILT_LIMIT))
        + 4.0 * _RESET_CLEARANCE_MARGIN
    )
    _PICK_GRASP_PROGRESS_BY_PAIR = (0.75,)
    _PICK_HELD_PROGRESS_BY_PAIR = (0.75,)
    _PICK_HELD_START_HEIGHT_BY_PAIR = (_TABLE_HEIGHT + 0.0035,)
    _PICK_HELD_TILT_DEGREES_BY_PAIR = (0.0,)
    _PICK_LIFT_BRIDGE_END_PROGRESS = 0.875
    _PICK_LIFT_BRIDGE_HEIGHT = 0.025
    _TRANSPORT_INITIAL_TILT_DEGREES = 15.0
    _PLACE_TILT_LIMIT = math.radians(45.0)
    _OBJECT_AXIS_APPROACH_DISTANCE = 0.10
    _PICK_APPROACH_DISTANCES_BY_YAW = (0.10,) * 8
    _MINIMUM_ACQUISITION_CORRIDOR_CENTER_DISTANCE = 0.15
    _MINIMUM_SEMANTIC_BASE_SOURCE_DISTANCE = 0.13
    _SECOND_TRANSPORT_HAND_STACK_CLEARANCE = 0.020
    _SECOND_TRANSPORT_CUBE_STACK_CLEARANCE = 0.015
    _SECOND_TRANSPORT_IK_REPAIR_BUFFER = 0.002
    _FINAL_RELEASE_INDEX_LINK_2_CLOSED_OFFSET = (0.08886563, -0.05273560, 0.02398680)
    _FINAL_RELEASE_INDEX_LINK_2_OPEN_OFFSET = (0.08809996, -0.05582009, 0.02321215)
    _FINAL_RELEASE_MIDDLE_LINK_2_OFFSET = (0.08730662, 0.00877876, 0.03099778)
    _FINAL_RELEASE_INDEX_LINK_2_CLEARANCE = 0.030
    _FINAL_RELEASE_MIDDLE_LINK_2_CLEARANCE = 0.030
    _FINAL_RELEASE_LOWER_STACK_HAND_CLEARANCE = 0.020
    _FINAL_RELEASE_RETREAT_CANDIDATES = (0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060)
    _SAFE_SEMANTIC_YAW_RECIPES = (
        StackResetRecipe.FIRST_PICK,
        StackResetRecipe.PAIR_READY,
        StackResetRecipe.SECOND_PICK,
    )

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        # RSL-RL enables TF32 before manager terms are resolved. Keep the
        # complete one-time bank construction in IEEE FP32, including the FK
        # checks around each IK solve, then restore fast policy matmuls.
        tf32_was_enabled = torch.backends.cuda.matmul.allow_tf32
        if tf32_was_enabled:
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            super().__init__(cfg, env)
        finally:
            if tf32_was_enabled:
                torch.backends.cuda.matmul.allow_tf32 = True

    @property
    def grasp_pair_ids(self) -> torch.Tensor:
        """Active opposing-finger/thumb pair ID for every reset row."""
        return self._grasp_pair_ids

    @property
    def orientation_bin_ids(self) -> torch.Tensor:
        """Wrist-orientation bin ID for every reset row."""
        return self._orientation_ids

    @property
    def tilt_azimuth_bin_ids(self) -> torch.Tensor:
        """Resolved palm-tilt direction bin for every reset row."""
        return self._tilt_azimuth_ids

    @property
    def authored_tilt_azimuth_bin_ids(self) -> torch.Tensor:
        """Requested palm-tilt direction bin before deterministic IK repair."""
        return self._authored_tilt_azimuth_ids

    @property
    def tilt_magnitude_bin_ids(self) -> torch.Tensor:
        """Authored palm-tilt magnitude bin for every reset row."""
        return self._tilt_magnitude_ids

    @property
    def layout_count(self) -> int:
        """Number of deterministic layouts represented by the diverse bank."""
        return self._SEMANTIC_LAYOUT_COUNT + self._TABLE_ROWS

    @classmethod
    def _table_surface_height(cls) -> float:
        """Return the support surface height [m]."""
        return cls._TABLE_HEIGHT - 0.5 * cls._CUBE_HEIGHT

    @classmethod
    def _pick_pregrasp_height(cls) -> float:
        """Return the collision-clear open-hand approach height [m]."""
        return cls._TABLE_HEIGHT + 0.5 * cls._CUBE_HEIGHT + cls._PICK_PREGRASP_TOP_CLEARANCE

    @classmethod
    def _pick_contact_height(cls) -> float:
        """Return the first physically held pick height [m]."""
        return cls._TABLE_HEIGHT + cls._PICK_CONTACT_CENTER_LIFT

    @classmethod
    def _pick_supported_height(cls) -> float:
        """Return the tool height for non-held PICK rows [m]."""
        return cls._pick_contact_height()

    @classmethod
    def _transport_height(cls, *, second_pick: bool) -> float:
        """Return the carry height that preserves cube-bottom clearance [m]."""
        bottom_clearance = (
            cls._SECOND_TRANSPORT_BOTTOM_CLEARANCE if second_pick else cls._FIRST_TRANSPORT_BOTTOM_CLEARANCE
        )
        return cls._table_surface_height() + 0.5 * cls._CUBE_HEIGHT + bottom_clearance

    @classmethod
    def _pair_ready_source_height(cls) -> float:
        """Return the loose-cube pre-grasp height [m]."""
        return cls._TABLE_HEIGHT + 0.5 * cls._CUBE_HEIGHT + cls._PAIR_READY_SOURCE_TOP_CLEARANCE

    @classmethod
    def _table_approach_minimum_height(cls) -> float:
        """Return the lowest open-hand table-approach height [m]."""
        return cls._TABLE_HEIGHT + 0.5 * cls._CUBE_HEIGHT + cls._TABLE_APPROACH_TOP_CLEARANCE

    @classmethod
    def _transport_minimum_height(cls) -> float:
        """Return the carry floor that preserves bottom clearance [m]."""
        return cls._table_surface_height() + 0.5 * cls._CUBE_HEIGHT + cls._RING_TRANSPORT_BOTTOM_CLEARANCE

    @classmethod
    def _default_target_potential(cls, recipe: StackResetRecipe) -> float:
        """Return the curriculum completion potential for one reset recipe."""
        return _RECIPE_TARGET_POTENTIAL[recipe]

    def _grasp_pair_pose(
        self,
        arm_positions: torch.Tensor,
        grasp_pair_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the configured pair-center pose for this cube geometry."""
        return kuka_allegro_grasp_pair_pose(
            arm_positions,
            grasp_pair_ids,
            self._GRASP_PAIR_TOOL_OFFSETS,
        )

    def _base_pick_phase(
        self,
        source_positions: torch.Tensor,
        progress: torch.Tensor,
        grasp_pair_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Author open-approach, supported closure, and optional held lift states."""
        approach_progress = torch.clamp(2.0 * progress, max=1.0)
        target_positions = source_positions.clone()
        target_positions[:, 2] = torch.lerp(
            target_positions.new_full(progress.shape, self._pick_pregrasp_height()),
            target_positions.new_full(progress.shape, self._pick_supported_height()),
            approach_progress,
        )
        grasp_progress = progress.new_tensor(self._PICK_GRASP_PROGRESS_BY_PAIR)[grasp_pair_ids]
        if bool(torch.any(grasp_progress <= 0.5)) or bool(torch.any(grasp_progress > 1.0)):
            raise RuntimeError("PICK grasp progress must lie in (0.5, 1.0].")
        held_progress = progress.new_tensor(self._PICK_HELD_PROGRESS_BY_PAIR)[grasp_pair_ids]
        if bool(torch.any(held_progress < grasp_progress)) or bool(torch.any(held_progress > 1.0)):
            raise RuntimeError("PICK held progress must lie between grasp completion and 1.0.")

        closure = torch.clamp(
            (progress - 0.5) / (grasp_progress - 0.5),
            min=0.0,
            max=1.0,
        )
        held = progress >= held_progress - 1.0e-6
        lift_progress = torch.clamp(
            (progress - held_progress) / (1.0 - held_progress).clamp_min(1.0e-6),
            min=0.0,
            max=1.0,
        )
        endpoint_heights = self._pick_endpoint_heights(grasp_pair_ids, progress)
        held_start_heights = progress.new_tensor(self._PICK_HELD_START_HEIGHT_BY_PAIR)[grasp_pair_ids]
        if bool(torch.any(held_start_heights > endpoint_heights)):
            raise RuntimeError("PICK held-start heights must not exceed their endpoint heights.")
        target_positions[:, 2] = torch.where(
            held,
            torch.lerp(
                held_start_heights,
                endpoint_heights,
                lift_progress,
            ),
            target_positions[:, 2],
        )
        maximum_tilt = torch.deg2rad(45.0 * (1.0 - torch.clamp(progress / grasp_progress, max=1.0)))
        held_maximum_tilt = torch.deg2rad(progress.new_tensor(self._PICK_HELD_TILT_DEGREES_BY_PAIR)[grasp_pair_ids])
        maximum_tilt = torch.where(held, held_maximum_tilt, maximum_tilt)
        return target_positions, closure, held, maximum_tilt

    def _palm_to_held_cube_rotations(
        self,
        grasp_pair_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return pair-conditioned cube rotations in palm coordinates."""
        quaternions = reference.new_tensor(self._PALM_TO_HELD_CUBE_QUATERNIONS_XYZW)
        return matrix_from_quaternion_xyzw(quaternions[grasp_pair_ids], reference)

    def _active_grasp_pair_ids(self, grasp_pair_ids: torch.Tensor) -> torch.Tensor:
        """Select the validated index/thumb reset grasp."""
        return torch.zeros_like(grasp_pair_ids)

    @classmethod
    def _default_orientation_ids_for_recipe(cls, recipe: StackResetRecipe) -> tuple[int, ...]:
        """Return wrist-yaw bins represented by one semantic recipe."""
        del recipe
        return tuple(range(8))

    @classmethod
    def _table_approach_role_ids(cls, row_ids: torch.Tensor) -> torch.Tensor:
        """Choose only a movable cube as the table-start approach target.

        Role zero is the curriculum's designated base. Approaching it cannot
        increase the table row's acquisition potential, so those starts waste
        a third of the sparse pickup stream. An odd modular permutation keeps
        roles one and two exactly balanced within every wrist-yaw bin.
        """
        permuted = torch.remainder(row_ids * 40503 + 17, cls._TABLE_ROWS)
        return 1 + (permuted >= cls._TABLE_ROWS // 2).long()

    def _resolved_table_approach_role_ids(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Return the target roles selected while the immutable table was built."""
        return self._table_approach_role_ids_by_row[row_ids]

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        """Return one coordinate of a deterministic Halton sequence."""
        inverse_base = 1.0 / base
        fraction = inverse_base
        value = 0.0
        while index:
            value += (index % base) * fraction
            index //= base
            fraction *= inverse_base
        return value

    @classmethod
    def _sample_layout_candidate(cls, sample_id: int, *, table_start: bool) -> tuple[tuple[float, float], ...]:
        """Sample three separated cube centers from a low-discrepancy layout."""
        x_lower, x_extent = (
            (cls._TABLE_X_LOWER, cls._TABLE_X_EXTENT)
            if table_start
            else (cls._SEMANTIC_X_LOWER, cls._SEMANTIC_X_EXTENT)
        )
        y_lower, y_extent = (
            (cls._TABLE_Y_LOWER, cls._TABLE_Y_EXTENT)
            if table_start
            else (cls._SEMANTIC_Y_LOWER, cls._SEMANTIC_Y_EXTENT)
        )
        minimum_separation = cls._LAYOUT_MINIMUM_SEPARATION
        layout_count = cls._TABLE_ROWS if table_start else cls._SEMANTIC_LAYOUT_COUNT
        sample_index = sample_id + 1
        # Keep the designated base on its own complete Halton sequence. Varying
        # all three roles during rejection heavily biased accepted large-cube
        # layouts toward one Y octile; only the two side-cube candidates should
        # advance between attempts.
        base = (
            x_lower + x_extent * cls._radical_inverse(sample_index, 5),
            y_lower + y_extent * cls._radical_inverse(sample_index, 7),
        )
        for attempt in range(96):
            sequence_id = sample_index + attempt * layout_count
            layout = (
                base,
                (
                    x_lower + x_extent * cls._radical_inverse(sequence_id + 11, 11),
                    y_lower + y_extent * cls._radical_inverse(sequence_id + 17, 13),
                ),
                (
                    x_lower + x_extent * cls._radical_inverse(sequence_id + 23, 17),
                    y_lower + y_extent * cls._radical_inverse(sequence_id + 29, 19),
                ),
            )
            if all(
                math.dist(layout[first], layout[second]) >= minimum_separation
                for first in range(3)
                for second in range(first + 1, 3)
            ):
                return layout
        # A small deterministic grid is a final guard for future workspace
        # changes. Fail loudly if its constraints are inconsistent.
        candidates = tuple(
            (x_lower + x_extent * x_fraction, y_lower + y_extent * y_fraction)
            for x_fraction in (0.0, 0.5, 1.0)
            for y_fraction in (0.0, 0.5, 1.0)
        )
        for first_source in candidates:
            for second_source in candidates:
                layout = (base, first_source, second_source)
                if all(
                    math.dist(layout[first], layout[second]) >= minimum_separation
                    for first in range(3)
                    for second in range(first + 1, 3)
                ):
                    return layout
        raise RuntimeError("Diverse KUKA cube-layout constraints are inconsistent with the configured workspace.")

    @classmethod
    def _pair_ready_targets(
        cls,
        base_positions: torch.Tensor,
        second_sources: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate the open hand over the completed first stack.

        A straight pair-center interpolation is not collision-safe for an
        Allegro hand: its open fingers sweep below the pair center, especially
        under the intentionally broad wrist tilts. Lift the middle of the
        bridge while preserving the after-place start and planar path. Its
        loose-cube endpoint remains a pre-grasp: SECOND_PICK owns the final
        descent into contact. The existing phase tilt also follows
        ``sin(pi * progress)``, so the extra clearance is largest exactly
        where the hand envelope is widest.
        """
        start = base_positions + base_positions.new_tensor(
            (0.0, 0.0, cls._CUBE_HEIGHT + cls._FIRST_PLACE_CENTER_CLEARANCE)
        )
        end = second_sources.clone()
        end[:, 2] = cls._pair_ready_source_height()
        targets = torch.lerp(start, end, progress.unsqueeze(1))
        # Two square roots broaden the arc near both shoulders while leaving
        # the exact endpoints untouched. This matters for a dexterous hand
        # because even a modest wrist tilt has a much larger swept envelope
        # than the pair-center point used by IK.
        clearance_profile = torch.sqrt(torch.sqrt(torch.sin(math.pi * progress).clamp_min(0.0)))
        targets[:, 2] += cls._PAIR_READY_CLEARANCE_ARC_HEIGHT * clearance_profile
        return targets

    @staticmethod
    def _axis_angle_rotation(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Build batched rotation matrices from normalized axes and angles."""
        axes = axes / torch.linalg.vector_norm(axes, dim=1, keepdim=True).clamp_min(1.0e-12)
        x, y, z = axes.unbind(dim=1)
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            (
                zeros,
                -z,
                y,
                z,
                zeros,
                -x,
                -y,
                x,
                zeros,
            ),
            dim=1,
        ).reshape(-1, 3, 3)
        identity = torch.eye(3, dtype=axes.dtype, device=axes.device).expand(axes.shape[0], -1, -1)
        outer = axes.unsqueeze(2) * axes.unsqueeze(1)
        cosine = torch.cos(angles).view(-1, 1, 1)
        sine = torch.sin(angles).view(-1, 1, 1)
        return cosine * identity + sine * skew + (1.0 - cosine) * outer

    def _target_wrist_rotations(
        self,
        orientation_ids: torch.Tensor,
        tilt_angles: torch.Tensor,
        tilt_azimuth_ids: torch.Tensor | None = None,
        grasp_pair_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return desired palm rotations and signed yaw angles."""
        del grasp_pair_ids
        yaw_steps = torch.where(orientation_ids <= 3, orientation_ids, orientation_ids - 8)
        yaw_angles = yaw_steps.to(self._arm_anchors.dtype) * (math.pi / 4.0)
        cosine, sine = torch.cos(yaw_angles), torch.sin(yaw_angles)
        zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
        yaw_rotation = torch.stack(
            (
                cosine,
                -sine,
                zeros,
                sine,
                cosine,
                zeros,
                zeros,
                zeros,
                ones,
            ),
            dim=1,
        ).reshape(-1, 3, 3)
        if tilt_azimuth_ids is None:
            tilt_azimuth_ids = orientation_ids
        tilt_azimuth = tilt_azimuth_ids.to(self._arm_anchors.dtype) * (math.pi / 4.0)
        tilt_axes = torch.stack(
            (
                torch.cos(tilt_azimuth),
                torch.sin(tilt_azimuth),
                torch.zeros_like(tilt_azimuth),
            ),
            dim=1,
        )
        tilt_rotation = self._axis_angle_rotation(tilt_axes, tilt_angles)
        _, nominal_rotation = kuka_allegro_pinch_pose(self._arm_anchors[4, _PREGRASP_POSE_INDEX])
        target_rotation = torch.matmul(torch.matmul(yaw_rotation, nominal_rotation), tilt_rotation)
        return target_rotation, yaw_angles

    def _screen_wrist_rotations(
        self,
        row_ids: torch.Tensor,
        target_positions: torch.Tensor,
        palm_rotations: torch.Tensor,
        tool_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return which candidate held-cube poses clear the table and support cubes."""
        valid = torch.ones(palm_rotations.shape[:2], dtype=torch.bool, device=self.device)
        held_rows = self._held_roles[row_ids] >= 0
        if not bool(torch.any(held_rows)):
            return valid

        held_pair_ids = self._grasp_pair_ids[row_ids[held_rows]]
        palm_to_cube = self._palm_to_held_cube_rotations(held_pair_ids, self._arm_anchors)
        cube_rotations = torch.matmul(palm_rotations[held_rows], palm_to_cube.unsqueeze(1))
        vertical_half_extent = (
            0.5
            * self._CUBE_HEIGHT
            * torch.sum(
                torch.abs(cube_rotations[..., 2, :]),
                dim=-1,
            )
        )
        if tool_positions is None:
            tool_positions = target_positions[row_ids, None].expand(-1, palm_rotations.shape[1], -1)
        cube_bottom = tool_positions[held_rows, :, 2] - vertical_half_extent
        table_surface = self._TABLE_HEIGHT - 0.5 * self._CUBE_HEIGHT
        held_valid = cube_bottom >= table_surface + self._RESET_CLEARANCE_MARGIN

        held_recipe_ids = self._recipe_ids[row_ids[held_rows]]
        first_place = held_recipe_ids == int(StackResetRecipe.FIRST_PLACE)
        if bool(torch.any(first_place)):
            support_top = self._role_positions[row_ids[held_rows][first_place], 0, 2] + 0.5 * self._CUBE_HEIGHT
            held_valid[first_place] &= (
                cube_bottom[first_place] >= support_top.unsqueeze(1) + self._RESET_CLEARANCE_MARGIN
            )

        second_place = held_recipe_ids == int(StackResetRecipe.SECOND_PLACE)
        if bool(torch.any(second_place)):
            support_top = self._role_positions[row_ids[held_rows][second_place], 1, 2] + 0.5 * self._CUBE_HEIGHT
            held_valid[second_place] &= (
                cube_bottom[second_place] >= support_top.unsqueeze(1) + self._RESET_CLEARANCE_MARGIN
            )

        valid[held_rows] = held_valid
        return valid

    def _held_cube_clearance_lift(
        self,
        row_ids: torch.Tensor,
        target_positions: torch.Tensor,
        palm_rotations: torch.Tensor,
        tool_positions: torch.Tensor | None = None,
        clearance_margin: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the vertical rewind needed for candidate held-cube poses."""
        if clearance_margin is None:
            clearance_margin = self._RESET_CLEARANCE_MARGIN
        elif isinstance(clearance_margin, torch.Tensor) and clearance_margin.ndim == 1:
            clearance_margin = clearance_margin.unsqueeze(1)
        if not bool(torch.all(self._held_roles[row_ids] >= 0)):
            raise RuntimeError("Held-cube clearance is only defined for rows with an authored grasp.")
        palm_to_cube = self._palm_to_held_cube_rotations(
            self._grasp_pair_ids[row_ids],
            self._arm_anchors,
        )
        cube_rotations = torch.matmul(palm_rotations, palm_to_cube.unsqueeze(1))
        vertical_half_extent = (
            0.5
            * self._CUBE_HEIGHT
            * torch.sum(
                torch.abs(cube_rotations[..., 2, :]),
                dim=-1,
            )
        )
        required_bottom = target_positions.new_full(
            vertical_half_extent.shape,
            self._TABLE_HEIGHT - 0.5 * self._CUBE_HEIGHT,
        )
        recipe_ids = self._recipe_ids[row_ids]
        first_place = recipe_ids == int(StackResetRecipe.FIRST_PLACE)
        if bool(torch.any(first_place)):
            support_top = self._role_positions[row_ids[first_place], 0, 2] + 0.5 * self._CUBE_HEIGHT
            required_bottom[first_place] = support_top.unsqueeze(1)
        second_place = recipe_ids == int(StackResetRecipe.SECOND_PLACE)
        if bool(torch.any(second_place)):
            support_top = self._role_positions[row_ids[second_place], 1, 2] + 0.5 * self._CUBE_HEIGHT
            required_bottom[second_place] = support_top.unsqueeze(1)
        if tool_positions is None:
            tool_positions = target_positions[row_ids, None].expand(-1, palm_rotations.shape[1], -1)
        current_bottom = tool_positions[..., 2] - vertical_half_extent
        return torch.clamp(required_bottom + clearance_margin - current_bottom, min=0.0)

    def _solve_arm_targets(
        self,
        seed_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_rotations: torch.Tensor,
        tool_offsets: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve reset IK and deterministically repair infeasible desired tilts."""
        arm_positions, position_residuals, rotation_residuals = solve_kuka_allegro_reset_ik(
            seed_positions,
            target_positions,
            target_rotations,
            tool_offsets,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
            arm_positions,
            self._grasp_pair_ids,
        )
        collision_free = self._collision_free_wrist_rotations(
            torch.arange(target_positions.shape[0], device=self.device),
            target_positions,
            actual_palm_rotations.unsqueeze(1),
            actual_tool_positions.unsqueeze(1),
        ).squeeze(1)
        failed_rows = torch.nonzero(
            (position_residuals > 8.0e-4) | (rotation_residuals > 1.0e-2) | ~collision_free,
            as_tuple=False,
        ).flatten()
        self._ik_fallback_count = int(failed_rows.numel())
        self._ik_vertical_rewind_count = 0
        if failed_rows.numel() == 0:
            return arm_positions, position_residuals, rotation_residuals, target_rotations

        # Negative angles are redundant once all eight axes in the local XY
        # plane are present. Search the exact authored target, zero once, and
        # nine magnitudes through 110 degrees about eight axes. Candidate
        # validity enforces the row's phase/pair-specific physical tilt limit.
        azimuth_shifts = torch.tensor((0, 1, -1, 2, -2, 3, -3, 4), device=self.device)
        candidate_azimuth_ids = torch.cat(
            (
                self._tilt_azimuth_ids[failed_rows, None],
                self._tilt_azimuth_ids[failed_rows, None],
                torch.remainder(
                    self._tilt_azimuth_ids[failed_rows, None, None] + azimuth_shifts[None, None],
                    8,
                )
                .expand(-1, 9, -1)
                .reshape(failed_rows.numel(), -1),
            ),
            dim=1,
        )
        candidate_tilt_angles = torch.cat(
            (
                self._resolved_tilt_angles[failed_rows, None],
                target_positions.new_zeros((failed_rows.numel(), 1)),
                torch.deg2rad(
                    torch.tensor(
                        (5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 110.0),
                        device=self.device,
                    )
                )[None, :, None]
                .expand(failed_rows.numel(), -1, 8)
                .reshape(failed_rows.numel(), -1),
            ),
            dim=1,
        )
        candidate_count = candidate_tilt_angles.shape[1]
        flat_row_ids = failed_rows[:, None].expand(-1, candidate_count).reshape(-1)
        candidate_rotations, _ = self._target_wrist_rotations(
            self._orientation_ids[flat_row_ids],
            candidate_tilt_angles.reshape(-1),
            candidate_azimuth_ids.reshape(-1),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_rotations = candidate_rotations.reshape(
            failed_rows.numel(),
            candidate_count,
            3,
            3,
        )
        candidate_seed_positions = seed_positions[flat_row_ids].reshape(
            failed_rows.numel(),
            candidate_count,
            7,
        )
        # The initial solve is already the best seed for the newly included
        # exact-target candidate, particularly for support-clearance rewinds.
        candidate_seed_positions[:, 0] = arm_positions[failed_rows]
        candidate_arm_positions, candidate_position_residuals, candidate_rotation_residuals = (
            solve_kuka_allegro_reset_ik(
                candidate_seed_positions.reshape(-1, 7),
                target_positions[flat_row_ids],
                candidate_rotations.reshape(-1, 3, 3),
                tool_offsets[flat_row_ids],
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                max_iterations=96,
            )
        )
        candidate_arm_positions = candidate_arm_positions.reshape(
            failed_rows.numel(),
            candidate_count,
            7,
        )
        candidate_position_residuals = candidate_position_residuals.reshape(
            failed_rows.numel(),
            candidate_count,
        )
        candidate_rotation_residuals = candidate_rotation_residuals.reshape(
            failed_rows.numel(),
            candidate_count,
        )
        candidate_arm_positions[:, 0] = arm_positions[failed_rows]
        candidate_position_residuals[:, 0] = position_residuals[failed_rows]
        candidate_rotation_residuals[:, 0] = rotation_residuals[failed_rows]
        candidate_actual_tool_positions, candidate_actual_palm_rotations = self._grasp_pair_pose(
            candidate_arm_positions.reshape(-1, 7),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_actual_tool_positions = candidate_actual_tool_positions.reshape(
            failed_rows.numel(),
            candidate_count,
            3,
        )
        candidate_actual_palm_rotations = candidate_actual_palm_rotations.reshape(
            failed_rows.numel(),
            candidate_count,
            3,
            3,
        )
        within_tilt_limit = candidate_tilt_angles <= self._resolved_tilt_limits[failed_rows, None] + 1.0e-6
        failed_index_first_place_rows = (self._recipe_ids[failed_rows] == int(StackResetRecipe.FIRST_PLACE)) & (
            self._grasp_pair_ids[failed_rows] == 0
        )
        within_tilt_limit &= ~(
            failed_index_first_place_rows[:, None]
            & (candidate_azimuth_ids == 6)
            & (candidate_tilt_angles > math.radians(75.0) + 1.0e-6)
        )
        valid = (
            within_tilt_limit
            & (candidate_position_residuals <= 8.0e-4)
            & (candidate_rotation_residuals <= 1.0e-2)
            & self._collision_free_wrist_rotations(
                failed_rows,
                target_positions,
                candidate_actual_palm_rotations,
                candidate_actual_tool_positions,
            )
        )
        relative_rotations = torch.matmul(
            candidate_rotations,
            target_rotations[failed_rows, None].transpose(-1, -2),
        )
        cosine_distance = 0.5 * (torch.diagonal(relative_rotations, dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
        orientation_deviation = torch.acos(torch.clamp(cosine_distance, -1.0, 1.0))
        geodesic_distance = orientation_deviation.masked_fill(~valid, torch.inf)
        best_distance, best_candidate_ids = torch.min(geodesic_distance, dim=1)
        row_indices = torch.arange(failed_rows.numel(), device=self.device)
        orientation_fallback = torch.isfinite(best_distance)
        if bool(torch.any(orientation_fallback)):
            fallback_rows = failed_rows[orientation_fallback]
            fallback_indices = row_indices[orientation_fallback]
            fallback_candidates = best_candidate_ids[orientation_fallback]
            arm_positions[fallback_rows] = candidate_arm_positions[fallback_indices, fallback_candidates]
            position_residuals[fallback_rows] = candidate_position_residuals[fallback_indices, fallback_candidates]
            rotation_residuals[fallback_rows] = candidate_rotation_residuals[fallback_indices, fallback_candidates]
            target_rotations[fallback_rows] = candidate_rotations[fallback_indices, fallback_candidates]
            self._tilt_azimuth_ids[fallback_rows] = candidate_azimuth_ids[fallback_indices, fallback_candidates]
            self._resolved_tilt_angles[fallback_rows] = candidate_tilt_angles[fallback_indices, fallback_candidates]

        rewound_rows = failed_rows[~orientation_fallback]
        self._ik_vertical_rewind_count = int(rewound_rows.numel())
        if rewound_rows.numel() > 0:
            nonheld_rows = rewound_rows[self._held_roles[rewound_rows] < 0]
            if nonheld_rows.numel() > 0:
                recipe_counts = torch.bincount(
                    self._recipe_ids[nonheld_rows],
                    minlength=len(StackResetRecipe),
                ).tolist()
                pair_counts = torch.bincount(self._grasp_pair_ids[nonheld_rows], minlength=3).tolist()
                raise RuntimeError(
                    "Diverse KUKA reset IK has no reachable orientation fallback "
                    f"for {int(nonheld_rows.numel())} non-held rows "
                    f"(recipes={recipe_counts}, "
                    f"pairs={pair_counts}, "
                    f"yaw={torch.bincount(self._orientation_ids[nonheld_rows], minlength=8).tolist()}, "
                    f"rows={nonheld_rows[:16].tolist()})."
                )
            unresolved_indices = row_indices[~orientation_fallback]
            kinematically_valid = (
                within_tilt_limit[unresolved_indices]
                & (candidate_position_residuals[unresolved_indices] <= 8.0e-4)
                & (candidate_rotation_residuals[unresolved_indices] <= 1.0e-2)
            )
            clearance_lift = self._held_cube_clearance_lift(
                rewound_rows,
                target_positions,
                candidate_actual_palm_rotations[unresolved_indices],
                candidate_actual_tool_positions[unresolved_indices],
                clearance_margin=2.0 * self._RESET_CLEARANCE_MARGIN,
            )
            # Prefer the smallest physical rewind; geodesic distance provides
            # a deterministic tie-break without changing that priority.
            rewind_score = clearance_lift + 1.0e-6 * orientation_deviation[unresolved_indices]
            rewind_score.masked_fill_(~kinematically_valid, torch.inf)
            best_rewind, rewind_candidate_ids = torch.min(rewind_score, dim=1)
            if not bool(torch.all(torch.isfinite(best_rewind))):
                unresolved = rewound_rows[~torch.isfinite(best_rewind)]
                recipe_counts = torch.bincount(
                    self._recipe_ids[unresolved],
                    minlength=len(StackResetRecipe),
                ).tolist()
                pair_counts = torch.bincount(self._grasp_pair_ids[unresolved], minlength=3).tolist()
                raise RuntimeError(
                    "Diverse KUKA reset IK has no kinematically valid vertical rewind "
                    f"for {int(unresolved.numel())} held place rows "
                    f"(recipes={recipe_counts}, "
                    f"pairs={pair_counts}, "
                    f"rows={unresolved[:16].tolist()}, "
                    f"yaw={self._orientation_ids[unresolved[:16]].tolist()}, "
                    f"tilt_azimuth={self._tilt_azimuth_ids[unresolved[:16]].tolist()}, "
                    f"tilt={torch.rad2deg(self._resolved_tilt_angles[unresolved[:16]]).tolist()}, "
                    f"target_xy={target_positions[unresolved[:16], :2].tolist()}, "
                    f"target_z=[{float(target_positions[unresolved, 2].min()):.4f}, "
                    f"{float(target_positions[unresolved, 2].max()):.4f}] m)."
                )

            rewind_indices = torch.arange(rewound_rows.numel(), device=self.device)
            selected_lift = clearance_lift[rewind_indices, rewind_candidate_ids]
            selected_rotations = candidate_rotations[unresolved_indices, rewind_candidate_ids]
            selected_seeds = candidate_arm_positions[unresolved_indices, rewind_candidate_ids]
            target_positions[rewound_rows, 2] += selected_lift
            self._role_positions[
                rewound_rows,
                self._held_roles[rewound_rows],
                2,
            ] += selected_lift
            (
                rewound_arm_positions,
                rewound_position_residuals,
                rewound_rotation_residuals,
            ) = solve_kuka_allegro_reset_ik(
                selected_seeds,
                target_positions[rewound_rows],
                selected_rotations,
                tool_offsets[rewound_rows],
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                max_iterations=72,
            )
            # This first rewind only needs to remain kinematically valid.
            # Re-solving at the lifted target can change the achieved cube
            # extent by sub-millimetres; the iterative final-clearance pass
            # below owns strict support clearance and will lift again as needed.
            rewind_valid = (rewound_position_residuals <= 8.0e-4) & (rewound_rotation_residuals <= 1.0e-2)
            if not bool(torch.all(rewind_valid)):
                invalid_rows = rewound_rows[~rewind_valid]
                recipe_counts = torch.bincount(
                    self._recipe_ids[invalid_rows],
                    minlength=len(StackResetRecipe),
                ).tolist()
                pair_counts = torch.bincount(self._grasp_pair_ids[invalid_rows], minlength=3).tolist()
                raise RuntimeError(
                    "Diverse KUKA reset IK vertical rewind failed strict validation "
                    f"for {int(invalid_rows.numel())} held place rows "
                    f"(recipes={recipe_counts}, "
                    f"pairs={pair_counts}, "
                    f"rows={invalid_rows[:16].tolist()}, "
                    f"target_z={target_positions[invalid_rows[:16], 2].tolist()}, "
                    f"position_residual={rewound_position_residuals[~rewind_valid][:16].tolist()}, "
                    f"rotation_residual={rewound_rotation_residuals[~rewind_valid][:16].tolist()})."
                )
            arm_positions[rewound_rows] = rewound_arm_positions
            position_residuals[rewound_rows] = rewound_position_residuals
            rotation_residuals[rewound_rows] = rewound_rotation_residuals
            target_rotations[rewound_rows] = selected_rotations
            self._tilt_azimuth_ids[rewound_rows] = candidate_azimuth_ids[unresolved_indices, rewind_candidate_ids]
            self._resolved_tilt_angles[rewound_rows] = candidate_tilt_angles[unresolved_indices, rewind_candidate_ids]
        return arm_positions, position_residuals, rotation_residuals, target_rotations

    def _finalize_held_cube_clearance(
        self,
        target_positions: torch.Tensor,
        target_rotations: torch.Tensor,
        tool_offsets: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-solve any final held pose that lacks a robust support margin."""
        held_rows = torch.nonzero(self._held_roles >= 0, as_tuple=False).flatten()
        rewound = torch.zeros(self._EXPECTED_ROW_COUNT, dtype=torch.bool, device=self.device)
        actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
            self._arm_positions,
            self._grasp_pair_ids,
        )
        held_recipe_ids = self._recipe_ids[held_rows]
        clearance_margins = target_positions.new_full(
            (held_rows.numel(),),
            4.0 * self._RESET_CLEARANCE_MARGIN,
        )
        # Include a 20 µm numerical guard so the achieved FP32 geometry still
        # meets the validated nominal 15/10 mm support margins.
        clearance_margins[held_recipe_ids == int(StackResetRecipe.FIRST_PLACE)] = (
            self._FIRST_PLACE_SUPPORT_MARGIN + 2.0e-5
        )
        clearance_margins[held_recipe_ids == int(StackResetRecipe.SECOND_PLACE)] = (
            self._SECOND_PLACE_SUPPORT_MARGIN + 2.0e-5
        )
        clearance_margin_by_row = target_positions.new_full(
            (self._EXPECTED_ROW_COUNT,),
            4.0 * self._RESET_CLEARANCE_MARGIN,
        )
        clearance_margin_by_row[held_rows] = clearance_margins
        for _ in range(self._FINAL_CLEARANCE_MAX_PASSES):
            clearance_lift = self._held_cube_clearance_lift(
                held_rows,
                target_positions,
                actual_palm_rotations[held_rows].unsqueeze(1),
                actual_tool_positions[held_rows].unsqueeze(1),
                clearance_margin=clearance_margins,
            ).squeeze(1)
            local_rows = torch.nonzero(clearance_lift > 1.0e-5, as_tuple=False).flatten()
            if local_rows.numel() == 0:
                break
            row_ids = held_rows[local_rows]
            selected_lift = clearance_lift[local_rows]
            rewound[row_ids] = True
            target_positions[row_ids, 2] += selected_lift
            self._role_positions[row_ids, self._held_roles[row_ids], 2] += selected_lift
            (
                self._arm_positions[row_ids],
                self._ik_position_residuals[row_ids],
                self._ik_rotation_residuals[row_ids],
            ) = solve_kuka_allegro_reset_ik(
                self._arm_positions[row_ids],
                target_positions[row_ids],
                target_rotations[row_ids],
                tool_offsets[row_ids],
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                max_iterations=96,
            )
            actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
                self._arm_positions,
                self._grasp_pair_ids,
            )
        self._ik_final_clearance_rewind_count = int(rewound.sum())

        # GPU FP32 rounding can leave a clearance-rewound or otherwise
        # near-limit held row just outside the strict 0.8 mm target tolerance
        # on one architecture while passing on another. Polish only this small
        # set in float64 so the immutable bank remains hardware-independent.
        polish_mask = rewound.clone()
        polish_mask[held_rows] |= self._ik_position_residuals[held_rows] > 7.5e-4
        polished_rows = torch.nonzero(polish_mask, as_tuple=False).flatten()
        self._ik_final_polish_count = 0
        if polished_rows.numel() > 0:
            polished_positions, _, _ = solve_kuka_allegro_reset_ik(
                self._arm_positions[polished_rows].double(),
                target_positions[polished_rows].double(),
                target_rotations[polished_rows].double(),
                tool_offsets[polished_rows].double(),
                joint_lower=joint_lower.double(),
                joint_upper=joint_upper.double(),
                max_iterations=256,
            )
            polished_positions = polished_positions.to(self._arm_positions.dtype)
            polished_tool_positions, polished_palm_rotations = self._grasp_pair_pose(
                polished_positions,
                self._grasp_pair_ids[polished_rows],
            )
            polished_position_residuals = torch.linalg.vector_norm(
                target_positions[polished_rows] - polished_tool_positions,
                dim=1,
            )
            relative_rotation = torch.matmul(
                target_rotations[polished_rows],
                polished_palm_rotations.transpose(-1, -2),
            )
            cosine = 0.5 * (torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=1) - 1.0)
            polished_rotation_residuals = torch.acos(torch.clamp(cosine, -1.0, 1.0))
            polished_clearance_lift = self._held_cube_clearance_lift(
                polished_rows,
                target_positions,
                polished_palm_rotations.unsqueeze(1),
                polished_tool_positions.unsqueeze(1),
                clearance_margin=clearance_margin_by_row[polished_rows],
            ).squeeze(1)
            improved = (
                (polished_position_residuals < self._ik_position_residuals[polished_rows])
                & (polished_rotation_residuals <= 1.0e-2)
                & (polished_clearance_lift <= 0.0)
            )
            accepted_rows = polished_rows[improved]
            self._ik_final_polish_count = int(accepted_rows.numel())
            self._arm_positions[accepted_rows] = polished_positions[improved]
            self._ik_position_residuals[accepted_rows] = polished_position_residuals[improved]
            self._ik_rotation_residuals[accepted_rows] = polished_rotation_residuals[improved]
            actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
                self._arm_positions,
                self._grasp_pair_ids,
            )

        remaining_lift = self._held_cube_clearance_lift(
            held_rows,
            target_positions,
            actual_palm_rotations[held_rows].unsqueeze(1),
            actual_tool_positions[held_rows].unsqueeze(1),
            clearance_margin=clearance_margins,
        ).squeeze(1)
        if bool(torch.any(remaining_lift > 1.0e-5)):
            invalid_rows = held_rows[remaining_lift > 1.0e-5]
            invalid_recipe_counts = torch.bincount(
                self._recipe_ids[invalid_rows],
                minlength=len(StackResetRecipe),
            ).tolist()
            raise RuntimeError(
                "Diverse KUKA reset IK could not establish final held-cube clearance "
                f"for {int(invalid_rows.numel())} rows "
                f"(maximum remaining lift={float(remaining_lift.max()):.6f} m, "
                f"recipe counts={invalid_recipe_counts})."
            )
        return actual_tool_positions, actual_palm_rotations

    def _build_table(
        self,
        *,
        closed_finger_position: float,
        placed_finger_position: float,
        open_finger_position: float,
        table_rows_per_layout: int = StackResetStateTable._TABLE_ROWS_PER_LAYOUT,
    ) -> None:
        """Build the exact-size pair-conditioned reset bank."""
        del table_rows_per_layout
        if not closed_finger_position < placed_finger_position < open_finger_position:
            raise ValueError("Diverse hand scalars must satisfy closed < placed < open.")

        semantic_layouts = self._arm_anchors.new_tensor(
            tuple(self._sample_layout(layout_id, table_start=False) for layout_id in range(self._SEMANTIC_LAYOUT_COUNT))
        )
        self._prepare_semantic_reset_plans(semantic_layouts)
        table_layouts = self._arm_anchors.new_tensor(
            tuple(self._sample_layout(layout_id, table_start=True) for layout_id in range(self._TABLE_ROWS))
        )
        table_height = self._TABLE_HEIGHT
        cube_height = self._CUBE_HEIGHT
        pick_contact_height = self._pick_contact_height()
        first_transport_height = self._transport_height(second_pick=False)
        second_transport_height = self._transport_height(second_pick=True)

        arm_targets: list[torch.Tensor] = []
        hand_rows: list[torch.Tensor] = []
        role_position_rows: list[torch.Tensor] = []
        recipe_rows: list[torch.Tensor] = []
        progress_rows: list[torch.Tensor] = []
        held_role_rows: list[torch.Tensor] = []
        layout_rows: list[torch.Tensor] = []
        pair_rows: list[torch.Tensor] = []
        orientation_rows: list[torch.Tensor] = []
        maximum_tilt_rows: list[torch.Tensor] = []
        closure_rows: list[torch.Tensor] = []

        def append_semantic(recipe: StackResetRecipe) -> None:
            local_rows = torch.arange(self._ROWS_PER_RECIPE, device=self.device)
            combination_ids = torch.remainder(local_rows, 24)
            layout_ids = torch.div(local_rows, 24, rounding_mode="floor")
            pair_ids = torch.div(combination_ids, 8, rounding_mode="floor")
            pair_ids = self._active_grasp_pair_ids(pair_ids)
            orientation_choices = local_rows.new_tensor(self._orientation_ids_for_recipe(recipe))
            if orientation_choices.numel() == 0:
                raise RuntimeError(f"Diverse KUKA recipe {recipe.name} must enable at least one wrist-yaw bin.")
            orientation_ids = orientation_choices[torch.remainder(local_rows, orientation_choices.numel())]
            orientation_ids = self._semantic_orientation_assignments(
                recipe,
                layout_ids,
                orientation_ids,
            )
            if orientation_ids.shape != local_rows.shape or not bool(
                torch.all(torch.isin(orientation_ids, orientation_choices))
            ):
                raise RuntimeError(
                    f"Diverse KUKA recipe {recipe.name} produced an invalid semantic wrist-yaw assignment."
                )
            # The odd multiplier is coprime to the power-of-two layout count.
            # Every pair/orientation
            # combination therefore gets the complete [0, 1] progression.
            progression_ids = torch.remainder(
                73 * layout_ids + 19 * combination_ids,
                self._SEMANTIC_LAYOUT_COUNT,
            )
            progress = progression_ids.to(self._arm_anchors.dtype) / (self._SEMANTIC_LAYOUT_COUNT - 1)
            layout = semantic_layouts[layout_ids]
            base_position = torch.cat(
                (layout[:, 0], self._arm_anchors.new_full((self._ROWS_PER_RECIPE, 1), table_height)),
                dim=1,
            )
            first_source = torch.cat(
                (layout[:, 1], self._arm_anchors.new_full((self._ROWS_PER_RECIPE, 1), table_height)),
                dim=1,
            )
            second_source = torch.cat(
                (layout[:, 2], self._arm_anchors.new_full((self._ROWS_PER_RECIPE, 1), table_height)),
                dim=1,
            )
            first_stack = base_position + base_position.new_tensor((0.0, 0.0, cube_height))
            second_stack = first_stack + base_position.new_tensor((0.0, 0.0, cube_height))
            role_positions = torch.stack((base_position, first_source, second_source), dim=1)
            held_roles = torch.full_like(local_rows, -1)
            closure = torch.zeros_like(progress)

            if recipe == StackResetRecipe.FINAL_RELEASE:
                role_positions = torch.stack((base_position, first_stack, second_stack), dim=1)
                target_positions = second_stack + torch.stack(
                    (torch.zeros_like(progress), torch.zeros_like(progress), 0.065 * progress),
                    dim=1,
                )
                closure = 1.0 - progress
                maximum_tilt = torch.deg2rad(95.0 * progress)
            elif recipe == StackResetRecipe.SECOND_PLACE:
                placement_progress = torch.clamp(progress / 0.92, max=1.0)
                second_place_approach = second_stack + second_stack.new_tensor(
                    (0.0, 0.0, self._SECOND_PLACE_CENTER_CLEARANCE)
                )
                second_position = torch.lerp(
                    second_place_approach,
                    second_stack,
                    placement_progress.unsqueeze(1),
                )
                role_positions = torch.stack((base_position, first_stack, second_position), dim=1)
                held = progress < 0.92
                held_roles[held] = 2
                closure = torch.where(held, torch.ones_like(progress), (1.0 - progress) / 0.08)
                closure.clamp_(0.0, 1.0)
                target_positions = second_position
                maximum_tilt = torch.deg2rad(80.0 * (1.0 - placement_progress))
            elif recipe == StackResetRecipe.SECOND_TRANSPORT:
                lift_progress = torch.clamp(2.0 * progress, max=1.0)
                carry_progress = torch.clamp(2.0 * progress - 1.0, min=0.0)
                lifted_source = second_source.clone()
                lifted_source[:, 2] = second_transport_height
                aligned_target = second_stack + second_stack.new_tensor((0.0, 0.0, self._SECOND_PLACE_CENTER_CLEARANCE))
                initial_target = second_source.clone()
                initial_target[:, 2] = pick_contact_height
                target_positions = torch.where(
                    (progress < 0.5).unsqueeze(1),
                    torch.lerp(initial_target, lifted_source, lift_progress.unsqueeze(1)),
                    torch.lerp(lifted_source, aligned_target, carry_progress.unsqueeze(1)),
                )
                # Second-cube phases continue from a completed first
                # placement. Leaving role one at ``first_source`` makes the
                # target potential (>= 8) unreachable.
                role_positions = torch.stack((base_position, first_stack, target_positions), dim=1)
                held_roles[:] = 2
                closure[:] = 1.0
                clearance = torch.clamp(
                    (target_positions[:, 2] - pick_contact_height) / (second_transport_height - pick_contact_height),
                    0.0,
                    1.0,
                )
                maximum_tilt = torch.deg2rad(
                    self._TRANSPORT_INITIAL_TILT_DEGREES + (110.0 - self._TRANSPORT_INITIAL_TILT_DEGREES) * clearance
                )
            elif recipe == StackResetRecipe.SECOND_PICK:
                role_positions = torch.stack((base_position, first_stack, second_source), dim=1)
                target_positions, closure, held, maximum_tilt = self._pick_phase(
                    second_source,
                    progress,
                    pair_ids,
                )
                held_roles[held] = 2
                role_positions[held_roles == 2, 2] = target_positions[held_roles == 2]
            elif recipe == StackResetRecipe.PAIR_READY:
                role_positions = torch.stack((base_position, first_stack, second_source), dim=1)
                target_positions = self._pair_ready_targets(base_position, second_source, progress)
                maximum_tilt = torch.deg2rad(70.0 * torch.sin(math.pi * progress))
            elif recipe == StackResetRecipe.FIRST_PLACE:
                placement_progress = torch.clamp(progress / 0.92, max=1.0)
                first_place_approach = first_stack + first_stack.new_tensor(
                    (0.0, 0.0, self._FIRST_PLACE_CENTER_CLEARANCE)
                )
                first_position = torch.lerp(
                    first_place_approach,
                    first_stack,
                    placement_progress.unsqueeze(1),
                )
                role_positions = torch.stack((base_position, first_position, second_source), dim=1)
                held = progress < 0.92
                held_roles[held] = 1
                closure = torch.where(held, torch.ones_like(progress), (1.0 - progress) / 0.08)
                closure.clamp_(0.0, 1.0)
                release_progress = torch.clamp((progress - 0.92) / 0.08, min=0.0, max=1.0)
                retreat_target = first_place_approach
                target_positions = torch.where(
                    held.unsqueeze(1),
                    first_position,
                    torch.lerp(first_stack, retreat_target, release_progress.unsqueeze(1)),
                )
                maximum_tilt = torch.deg2rad(80.0 * (1.0 - placement_progress))
            elif recipe == StackResetRecipe.FIRST_TRANSPORT:
                lift_progress = torch.clamp(2.0 * progress, max=1.0)
                carry_progress = torch.clamp(2.0 * progress - 1.0, min=0.0)
                lifted_source = first_source.clone()
                lifted_source[:, 2] = first_transport_height
                aligned_target = first_stack + first_stack.new_tensor((0.0, 0.0, self._FIRST_PLACE_CENTER_CLEARANCE))
                initial_target = first_source.clone()
                initial_target[:, 2] = pick_contact_height
                target_positions = torch.where(
                    (progress < 0.5).unsqueeze(1),
                    torch.lerp(initial_target, lifted_source, lift_progress.unsqueeze(1)),
                    torch.lerp(lifted_source, aligned_target, carry_progress.unsqueeze(1)),
                )
                role_positions[:, 1] = target_positions
                held_roles[:] = 1
                closure[:] = 1.0
                clearance = torch.clamp(
                    (target_positions[:, 2] - pick_contact_height) / (first_transport_height - pick_contact_height),
                    0.0,
                    1.0,
                )
                maximum_tilt = torch.deg2rad(
                    self._TRANSPORT_INITIAL_TILT_DEGREES + (110.0 - self._TRANSPORT_INITIAL_TILT_DEGREES) * clearance
                )
            elif recipe == StackResetRecipe.FIRST_PICK:
                target_positions, closure, held, maximum_tilt = self._pick_phase(
                    first_source,
                    progress,
                    pair_ids,
                )
                held_roles[held] = 1
                role_positions[held_roles == 1, 1] = target_positions[held_roles == 1]
            else:
                raise RuntimeError(f"Unsupported diverse semantic recipe: {recipe}.")

            open_commands = self._arm_anchors.new_tensor(self._GRASP_PAIR_OPEN_COMMANDS)[pair_ids]
            closed_commands = self._arm_anchors.new_tensor(self._GRASP_PAIR_RESET_CLOSED_COMMANDS)[pair_ids]
            hand_positions = torch.lerp(open_commands, closed_commands, closure.unsqueeze(1))
            arm_targets.append(target_positions)
            hand_rows.append(hand_positions)
            role_position_rows.append(role_positions)
            recipe_rows.append(torch.full_like(local_rows, int(recipe)))
            progress_rows.append(progress)
            held_role_rows.append(held_roles)
            layout_rows.append(layout_ids)
            pair_rows.append(pair_ids)
            orientation_rows.append(orientation_ids)
            maximum_tilt_rows.append(maximum_tilt)
            closure_rows.append(closure)

        for recipe in StackResetRecipe:
            if recipe != StackResetRecipe.TABLE:
                append_semantic(recipe)

        table_local_rows = torch.arange(self._TABLE_ROWS, device=self.device)
        table_selected_roles, table_orientation_ids = self._table_approach_assignments(
            table_local_rows,
            table_layouts,
        )
        if (
            table_selected_roles.shape != table_local_rows.shape
            or table_orientation_ids.shape != table_local_rows.shape
        ):
            raise RuntimeError("TABLE target-role and wrist-yaw assignments must match the table row count.")
        self._table_approach_role_ids_by_row = table_selected_roles
        table_pair_ids = torch.remainder(
            torch.div(table_local_rows, 8, rounding_mode="floor") + table_orientation_ids,
            3,
        )
        table_pair_ids = self._active_grasp_pair_ids(table_pair_ids)
        table_role_positions = torch.cat(
            (
                table_layouts,
                self._arm_anchors.new_full((self._TABLE_ROWS, 3, 1), table_height),
            ),
            dim=2,
        )
        selected_roles = self._resolved_table_approach_role_ids(table_local_rows)
        selected_xy = table_layouts.gather(1, selected_roles.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        table_progress = torch.remainder(table_local_rows * 4099 + 127, self._TABLE_ROWS).to(
            self._arm_anchors.dtype
        ) / (self._TABLE_ROWS - 1)
        table_targets = torch.cat(
            (
                selected_xy,
                (self._table_approach_minimum_height() + self._TABLE_APPROACH_HEIGHT_RANGE * table_progress).unsqueeze(
                    1
                ),
            ),
            dim=1,
        )
        table_open_commands = self._arm_anchors.new_tensor(self._GRASP_PAIR_OPEN_COMMANDS)[table_pair_ids]
        arm_targets.append(table_targets)
        hand_rows.append(table_open_commands)
        role_position_rows.append(table_role_positions)
        recipe_rows.append(torch.full_like(table_local_rows, int(StackResetRecipe.TABLE)))
        progress_rows.append(table_progress)
        held_role_rows.append(torch.full_like(table_local_rows, -1))
        layout_rows.append(self._SEMANTIC_LAYOUT_COUNT + table_local_rows)
        pair_rows.append(table_pair_ids)
        orientation_rows.append(table_orientation_ids)
        maximum_tilt_rows.append(torch.zeros_like(table_progress))
        closure_rows.append(torch.zeros_like(table_progress))

        target_positions = torch.cat(arm_targets)
        self._hand_positions = torch.cat(hand_rows)
        self._role_positions = torch.cat(role_position_rows)
        self._recipe_ids = torch.cat(recipe_rows)
        self._progress = torch.cat(progress_rows)
        self._held_roles = torch.cat(held_role_rows)
        self._layout_ids = torch.cat(layout_rows)
        self._grasp_pair_ids = torch.cat(pair_rows)
        self._orientation_ids = torch.cat(orientation_rows)
        self._authored_orientation_ids = self._orientation_ids.clone()
        closure = torch.cat(closure_rows)

        table_rows = self._recipe_ids == int(StackResetRecipe.TABLE)
        self._tilt_azimuth_ids = torch.remainder(
            5 * self._layout_ids + 3 * self._grasp_pair_ids + self._recipe_ids,
            8,
        )
        self._tilt_azimuth_ids[table_rows] = torch.remainder(
            torch.div(
                self._layout_ids[table_rows] - self._SEMANTIC_LAYOUT_COUNT,
                8,
                rounding_mode="floor",
            ),
            8,
        )
        self._authored_tilt_azimuth_ids = self._tilt_azimuth_ids.clone()
        self._tilt_magnitude_ids = torch.remainder(
            torch.div(self._layout_ids, 8, rounding_mode="floor")
            + 3 * self._orientation_ids
            + self._grasp_pair_ids
            + self._recipe_ids,
            4,
        )
        self._tilt_magnitude_ids[table_rows] = torch.remainder(
            torch.div(
                self._layout_ids[table_rows] - self._SEMANTIC_LAYOUT_COUNT,
                64,
                rounding_mode="floor",
            ),
            4,
        )
        transport_rows = (self._recipe_ids == int(StackResetRecipe.FIRST_TRANSPORT)) | (
            self._recipe_ids == int(StackResetRecipe.SECOND_TRANSPORT)
        )
        place_rows = (self._recipe_ids == int(StackResetRecipe.FIRST_PLACE)) | (
            self._recipe_ids == int(StackResetRecipe.SECOND_PLACE)
        )
        floored_transport_rows = transport_rows
        # Dynamically fragile low carries begin at the first repeatedly
        # validated one-second retention height.
        target_positions[floored_transport_rows, 2] = target_positions[floored_transport_rows, 2].clamp_min(
            self._transport_minimum_height()
        )

        tilt_scales = (self._tilt_magnitude_ids.to(self._arm_anchors.dtype) + 1.0) / 4.0
        phase_tilt_limits = torch.minimum(
            torch.cat(maximum_tilt_rows),
            target_positions.new_full(
                (self._EXPECTED_ROW_COUNT,),
                self._GLOBAL_TILT_LIMIT,
            ),
        )
        tilt_angles = phase_tilt_limits * tilt_scales
        self._resolved_tilt_limits = torch.full_like(phase_tilt_limits, self._GLOBAL_TILT_LIMIT)
        self._resolved_tilt_limits[floored_transport_rows] = phase_tilt_limits[floored_transport_rows]
        pick_rows = (self._recipe_ids == int(StackResetRecipe.FIRST_PICK)) | (
            self._recipe_ids == int(StackResetRecipe.SECOND_PICK)
        )
        self._resolved_tilt_limits[pick_rows] = phase_tilt_limits[pick_rows]
        tilt_angles[place_rows] = torch.minimum(
            tilt_angles[place_rows],
            torch.full_like(tilt_angles[place_rows], self._PLACE_TILT_LIMIT),
        )
        self._resolved_tilt_limits[place_rows] = torch.minimum(
            self._resolved_tilt_limits[place_rows],
            torch.full_like(self._resolved_tilt_limits[place_rows], self._PLACE_TILT_LIMIT),
        )
        index_first_place_cap = (
            (self._recipe_ids == int(StackResetRecipe.FIRST_PLACE))
            & (self._grasp_pair_ids == 0)
            & (self._tilt_azimuth_ids == 6)
            & (tilt_angles > math.radians(75.0))
        )
        tilt_angles[index_first_place_cap] = math.radians(75.0)
        self._resolved_tilt_angles = tilt_angles.clone()
        target_rotations, yaw_angles = self._target_wrist_rotations(
            self._orientation_ids,
            tilt_angles,
            self._tilt_azimuth_ids,
            self._grasp_pair_ids,
        )
        target_positions = self._adjust_target_positions_for_rotation(
            target_positions,
            target_rotations,
        )
        anchor_positions, _ = kuka_allegro_pinch_pose(self._arm_anchors.reshape(-1, 7))
        nearest_anchor_ids = torch.argmin(torch.cdist(target_positions, anchor_positions), dim=1)
        seed_positions = self._arm_anchors.reshape(-1, 7)[nearest_anchor_ids].clone()
        seed_positions[:, 6] -= yaw_angles
        seed_positions[:, 6] = torch.remainder(seed_positions[:, 6] + math.pi, 2.0 * math.pi) - math.pi
        joint_lower = seed_positions.new_tensor(self._ARM_WORKSPACE_LOWER)
        joint_upper = seed_positions.new_tensor(self._ARM_WORKSPACE_UPPER)
        seed_positions.clamp_(min=joint_lower, max=joint_upper)
        tool_offsets = seed_positions.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS)[self._grasp_pair_ids]
        (
            self._arm_positions,
            self._ik_position_residuals,
            self._ik_rotation_residuals,
            target_rotations,
        ) = self._solve_diverse_arm_targets(
            seed_positions,
            target_positions,
            target_rotations,
            tool_offsets,
            joint_lower,
            joint_upper,
        )

        actual_tool_positions, actual_palm_rotations = self._finalize_held_cube_clearance(
            target_positions,
            target_rotations,
            tool_offsets,
            joint_lower,
            joint_upper,
        )
        self._resolved_palm_rotations = actual_palm_rotations
        held_rows = torch.nonzero(self._held_roles >= 0, as_tuple=False).flatten()
        if held_rows.numel() > 0:
            self._role_positions[held_rows, self._held_roles[held_rows]] = actual_tool_positions[held_rows]

        row_ids = torch.arange(self._EXPECTED_ROW_COUNT, device=self.device).unsqueeze(1)
        role_ids = torch.arange(3, device=self.device).unsqueeze(0)
        yaw_fraction = torch.remainder(row_ids * 193 + role_ids * 389 + 17, 4093).float() / 4092.0
        half_yaw = math.pi * (2.0 * yaw_fraction - 1.0)
        self._role_quaternions = torch.zeros(
            (self._EXPECTED_ROW_COUNT, 3, 4),
            dtype=self._arm_anchors.dtype,
            device=self.device,
        )
        self._role_quaternions[..., 2] = torch.sin(half_yaw)
        self._role_quaternions[..., 3] = torch.cos(half_yaw)
        # Closing/release rows must present the same cube face that the
        # selected fingertip pair reaches at contact. Keep these supported
        # cubes upright, but derive their yaw from the achieved palm and
        # pair-specific palm-to-cube transform. TABLE rows remain fully
        # randomized and therefore retain deployment-start diversity.
        contact_roles = torch.full(
            (self._EXPECTED_ROW_COUNT,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        contact_roles[self._recipe_ids == int(StackResetRecipe.FINAL_RELEASE)] = 2
        contact_roles[self._recipe_ids == int(StackResetRecipe.SECOND_PLACE)] = 2
        contact_roles[self._recipe_ids == int(StackResetRecipe.SECOND_PICK)] = 2
        contact_roles[self._recipe_ids == int(StackResetRecipe.PAIR_READY)] = 2
        contact_roles[self._recipe_ids == int(StackResetRecipe.FIRST_PLACE)] = 1
        contact_roles[self._recipe_ids == int(StackResetRecipe.FIRST_PICK)] = 1
        table_rows = torch.nonzero(
            self._recipe_ids == int(StackResetRecipe.TABLE),
            as_tuple=False,
        ).flatten()
        table_local_rows = self._layout_ids[table_rows] - self._SEMANTIC_LAYOUT_COUNT
        contact_roles[table_rows] = self._resolved_table_approach_role_ids(table_local_rows)
        contact_rows = torch.nonzero(contact_roles >= 0, as_tuple=False).flatten()
        palm_to_cube = self._palm_to_held_cube_rotations(
            self._grasp_pair_ids[contact_rows],
            self._arm_anchors,
        )
        contact_rotations = torch.matmul(actual_palm_rotations[contact_rows], palm_to_cube)
        contact_yaw = torch.atan2(contact_rotations[:, 1, 0], contact_rotations[:, 0, 0])
        contact_quaternions = torch.zeros(
            (contact_rows.numel(), 4),
            dtype=self._arm_anchors.dtype,
            device=self.device,
        )
        contact_quaternions[:, 2] = torch.sin(0.5 * contact_yaw)
        contact_quaternions[:, 3] = torch.cos(0.5 * contact_yaw)
        self._role_quaternions[
            contact_rows,
            contact_roles[contact_rows],
        ] = contact_quaternions
        if held_rows.numel() > 0:
            palm_to_cube = self._palm_to_held_cube_rotations(
                self._grasp_pair_ids[held_rows],
                self._arm_anchors,
            )
            held_rotations = torch.matmul(actual_palm_rotations[held_rows], palm_to_cube)
            self._role_quaternions[held_rows, self._held_roles[held_rows]] = quaternion_xyzw_from_matrix(held_rotations)

        self._finger_positions = open_finger_position + closure * (closed_finger_position - open_finger_position)
        self._target_potentials = torch.tensor(
            tuple(self._target_potential(StackResetRecipe(int(recipe))) for recipe in self._recipe_ids.cpu()),
            dtype=torch.float32,
            device=self.device,
        )

    def _validate_kuka_table(self) -> None:  # noqa: C901
        """Validate exact balance, IK quality, bounds, and cube clearance."""
        super()._validate_table()
        if self.row_count != self._EXPECTED_ROW_COUNT:
            raise RuntimeError(f"Diverse KUKA reset bank must have {self._EXPECTED_ROW_COUNT} rows.")
        expected_recipe_counts = torch.full(
            (len(StackResetRecipe),),
            self._ROWS_PER_RECIPE,
            dtype=torch.long,
            device=self.device,
        )
        expected_recipe_counts[int(StackResetRecipe.TABLE)] = self._TABLE_ROWS
        if not torch.equal(torch.bincount(self._recipe_ids, minlength=len(StackResetRecipe)), expected_recipe_counts):
            raise RuntimeError("Diverse KUKA reset bank recipe counts are not exactly balanced.")
        table_role_ids = self._resolved_table_approach_role_ids(torch.arange(self._TABLE_ROWS, device=self.device))
        expected_table_role_counts = torch.tensor(
            (0, self._TABLE_ROWS // 2, self._TABLE_ROWS // 2),
            dtype=torch.long,
            device=self.device,
        )
        if not torch.equal(torch.bincount(table_role_ids, minlength=3), expected_table_role_counts):
            raise RuntimeError("Diverse KUKA TABLE rows are not exactly balanced over the two movable roles.")
        expected_orientation_counts = torch.zeros(8, dtype=torch.long, device=self.device)
        safe_semantic_yaw_recipes = set(self._SAFE_SEMANTIC_YAW_RECIPES)
        for recipe in StackResetRecipe:
            orientation_ids = self._grasp_pair_ids.new_tensor(self._orientation_ids_for_recipe(recipe))
            recipe_row_count = self._TABLE_ROWS if recipe == StackResetRecipe.TABLE else self._ROWS_PER_RECIPE
            if orientation_ids.numel() == 0:
                raise RuntimeError(f"Diverse KUKA recipe {recipe.name} must enable at least one wrist-yaw bin.")
            if recipe == StackResetRecipe.TABLE or recipe in safe_semantic_yaw_recipes:
                table_orientation_ids = self._orientation_ids[self._recipe_ids == int(recipe)]
                if not bool(torch.all(torch.isin(table_orientation_ids, orientation_ids))):
                    raise RuntimeError(f"Diverse KUKA {recipe.name} rows contain a disabled wrist-yaw bin.")
                if recipe != StackResetRecipe.TABLE and bool(
                    torch.any(torch.bincount(table_orientation_ids, minlength=8)[orientation_ids] == 0)
                ):
                    raise RuntimeError(f"Diverse KUKA {recipe.name} rows omit an enabled safe wrist-yaw bin.")
                expected_orientation_counts += torch.bincount(table_orientation_ids, minlength=8)
                continue
            repeated_orientation_ids = orientation_ids[
                torch.remainder(
                    torch.arange(recipe_row_count, device=self.device),
                    orientation_ids.numel(),
                )
            ]
            expected_orientation_counts += torch.bincount(repeated_orientation_ids, minlength=8)
        if not torch.equal(
            torch.bincount(self._authored_orientation_ids, minlength=8),
            expected_orientation_counts,
        ):
            raise RuntimeError("Diverse KUKA reset bank is not balanced across wrist-yaw bins.")
        if bool(torch.any(self._grasp_pair_ids != 0)):
            raise RuntimeError("KUKA reset rows must use the calibrated index/thumb grasp.")
        active_pair_ids = self._grasp_pair_ids.new_tensor((0,))
        if bool(torch.any((self._tilt_azimuth_ids < 0) | (self._tilt_azimuth_ids >= 8))):
            raise RuntimeError("Diverse KUKA reset bank contains an invalid palm-tilt azimuth.")
        if bool(torch.any((self._tilt_magnitude_ids < 0) | (self._tilt_magnitude_ids >= 4))):
            raise RuntimeError("Diverse KUKA reset bank contains an invalid palm-tilt magnitude.")
        maximum_tilt_degrees = math.degrees(self._GLOBAL_TILT_LIMIT)
        if float(torch.rad2deg(self._resolved_tilt_angles).max()) > maximum_tilt_degrees + 1.0e-4:
            raise RuntimeError(
                f"Diverse KUKA reset bank contains a palm tilt above {maximum_tilt_degrees:.4g} degrees."
            )
        pair_orientation_count = 8
        pair_orientation_ids = self._authored_orientation_ids
        wrist_orientation_ids = 8 * self._orientation_ids + self._tilt_azimuth_ids
        reset_mode_ids = 4 * (8 * pair_orientation_ids + self._authored_tilt_azimuth_ids) + self._tilt_magnitude_ids
        reset_mode_count = 32 * pair_orientation_count
        for recipe in StackResetRecipe:
            recipe_mask = self._recipe_ids == int(recipe)
            orientation_ids = self._grasp_pair_ids.new_tensor(self._orientation_ids_for_recipe(recipe))
            if recipe != StackResetRecipe.TABLE:
                if (
                    orientation_ids.numel() == 0
                    or bool(torch.any((orientation_ids < 0) | (orientation_ids >= 8)))
                    or torch.unique(orientation_ids).numel() != orientation_ids.numel()
                ):
                    raise RuntimeError(f"Diverse KUKA recipe {recipe.name} has invalid wrist-yaw bins.")
                if recipe not in safe_semantic_yaw_recipes:
                    expected_pair_orientation_ids = (
                        8 * torch.arange(active_pair_ids.numel(), device=self.device)[:, None]
                        + orientation_ids[None, :]
                    ).flatten()
                    expected_pair_orientation_counts = torch.zeros(
                        pair_orientation_count,
                        dtype=torch.long,
                        device=self.device,
                    )
                    if self._ROWS_PER_RECIPE % expected_pair_orientation_ids.numel() != 0:
                        raise RuntimeError(f"Diverse KUKA recipe {recipe.name} cannot balance its wrist-yaw bins.")
                    expected_pair_orientation_counts[expected_pair_orientation_ids] = (
                        self._ROWS_PER_RECIPE // expected_pair_orientation_ids.numel()
                    )
                    if not torch.equal(
                        torch.bincount(pair_orientation_ids[recipe_mask], minlength=pair_orientation_count),
                        expected_pair_orientation_counts,
                    ):
                        raise RuntimeError(f"Diverse KUKA recipe {recipe.name} is not pair/orientation balanced.")
                expected_wrist_orientation_ids = (
                    8 * orientation_ids[:, None] + torch.arange(8, device=self.device)[None, :]
                ).flatten()
                wrist_orientation_counts = torch.bincount(wrist_orientation_ids[recipe_mask], minlength=64)
                if bool(torch.any(wrist_orientation_counts[expected_wrist_orientation_ids] == 0)):
                    raise RuntimeError(f"Diverse KUKA recipe {recipe.name} does not span all wrist directions.")
                if recipe not in safe_semantic_yaw_recipes:
                    expected_reset_mode_ids = torch.stack(
                        torch.meshgrid(
                            torch.arange(active_pair_ids.numel(), device=self.device),
                            orientation_ids,
                            torch.arange(8, device=self.device),
                            torch.arange(4, device=self.device),
                            indexing="ij",
                        ),
                        dim=-1,
                    ).reshape(-1, 4)
                    expected_reset_mode_ids = (
                        4
                        * (
                            8 * (8 * expected_reset_mode_ids[:, 0] + expected_reset_mode_ids[:, 1])
                            + expected_reset_mode_ids[:, 2]
                        )
                        + expected_reset_mode_ids[:, 3]
                    )
                    expected_reset_mode_counts = torch.zeros(
                        reset_mode_count,
                        dtype=torch.long,
                        device=self.device,
                    )
                    if self._ROWS_PER_RECIPE % expected_reset_mode_ids.numel() != 0:
                        raise RuntimeError(f"Diverse KUKA recipe {recipe.name} cannot balance its reset modes.")
                    expected_reset_mode_counts[expected_reset_mode_ids] = (
                        self._ROWS_PER_RECIPE // expected_reset_mode_ids.numel()
                    )
                    if not torch.equal(
                        torch.bincount(reset_mode_ids[recipe_mask], minlength=reset_mode_count),
                        expected_reset_mode_counts,
                    ):
                        raise RuntimeError(f"Diverse KUKA recipe {recipe.name} is not pair/yaw/tilt-mode balanced.")

            # A reset mode must not be inferable from one coarse cube
            # coordinate. This caught a base-two Halton alias where the table
            # base-cube X octile identified wrist yaw perfectly.
            spatial_lower = self._role_positions.new_tensor(
                (self._TABLE_X_LOWER, self._TABLE_Y_LOWER)
                if recipe == StackResetRecipe.TABLE
                else (self._SEMANTIC_X_LOWER, self._SEMANTIC_Y_LOWER)
            )
            spatial_extent = self._role_positions.new_tensor(
                (self._TABLE_X_EXTENT, self._TABLE_Y_EXTENT)
                if recipe == StackResetRecipe.TABLE
                else (self._SEMANTIC_X_EXTENT, self._SEMANTIC_Y_EXTENT)
            )
            spatial_bins = torch.clamp(
                (8.0 * (self._role_positions[recipe_mask, 0, :2] - spatial_lower) / spatial_extent).long(),
                min=0,
                max=7,
            )
            orientation_ranks = torch.full((8,), -1, dtype=torch.long, device=self.device)
            orientation_ranks[orientation_ids] = torch.arange(orientation_ids.numel(), device=self.device)
            mode_dimensions = (
                (
                    orientation_ranks[self._authored_orientation_ids[recipe_mask]],
                    orientation_ids.numel(),
                    "yaw",
                ),
                (self._authored_tilt_azimuth_ids[recipe_mask], 8, "tilt azimuth"),
                (self._tilt_magnitude_ids[recipe_mask], 4, "tilt magnitude"),
            )
            for spatial_axis, axis_name in enumerate(("X", "Y")):
                for mode_ids, mode_count, mode_name in mode_dimensions:
                    if recipe in safe_semantic_yaw_recipes and mode_name == "yaw":
                        continue
                    joint_ids = mode_count * spatial_bins[:, spatial_axis] + mode_ids
                    joint_counts = torch.bincount(joint_ids, minlength=8 * mode_count)
                    if bool(torch.any(joint_counts == 0)):
                        raise RuntimeError(
                            f"Diverse KUKA recipe {recipe.name} aliases spatial {axis_name} bins with {mode_name} "
                            f"(empty joint bins={torch.nonzero(joint_counts == 0, as_tuple=False).flatten().tolist()})."
                        )
        lower = self._arm_positions.new_tensor(self._ARM_WORKSPACE_LOWER)
        upper = self._arm_positions.new_tensor(self._ARM_WORKSPACE_UPPER)
        if bool(torch.any((self._arm_positions < lower - 1.0e-6) | (self._arm_positions > upper + 1.0e-6))):
            raise RuntimeError("Diverse KUKA reset IK produced a pose outside its safe workspace.")
        if float(self._ik_position_residuals.max()) > self._FINAL_IK_POSITION_RESIDUAL_LIMIT:
            invalid_rows = torch.nonzero(
                self._ik_position_residuals > self._FINAL_IK_POSITION_RESIDUAL_LIMIT,
                as_tuple=False,
            ).flatten()
            raise RuntimeError(
                "Diverse KUKA reset IK position residual exceeds its configured limit "
                f"(max={float(self._ik_position_residuals.max()):.6f} m, "
                f"count={invalid_rows.numel()}, "
                f"recipes={torch.bincount(self._recipe_ids[invalid_rows], minlength=len(StackResetRecipe)).tolist()}, "
                f"pairs={torch.bincount(self._grasp_pair_ids[invalid_rows], minlength=3).tolist()}, "
                f"yaw={torch.bincount(self._orientation_ids[invalid_rows], minlength=8).tolist()}, "
                f"rows={invalid_rows[:16].tolist()})."
            )
        if float(self._ik_rotation_residuals.max()) > 1.0e-2:
            raise RuntimeError(
                "Diverse KUKA reset IK orientation residual exceeds 0.01 rad "
                f"(max={float(self._ik_rotation_residuals.max()):.6f} rad)."
            )

        held_rows = torch.nonzero(self._held_roles >= 0, as_tuple=False).flatten()
        if held_rows.numel() > 0:
            held_rotations = matrix_from_quaternion_xyzw(
                self._role_quaternions[held_rows, self._held_roles[held_rows]],
                self._arm_positions,
            )
            vertical_half_extent = (
                0.5
                * self._CUBE_HEIGHT
                * torch.sum(
                    torch.abs(held_rotations[:, 2]),
                    dim=1,
                )
            )
            held_heights = self._role_positions[held_rows, self._held_roles[held_rows], 2]
            table_surface = self._TABLE_HEIGHT - 0.5 * self._CUBE_HEIGHT
            cube_bottom = held_heights - vertical_half_extent
            if bool(torch.any(cube_bottom < table_surface - 1.0e-5)):
                invalid = cube_bottom < table_surface - 1.0e-5
                raise RuntimeError(
                    "A rotated held cube violates table clearance in the diverse reset bank "
                    f"(count={int(invalid.sum())}, minimum={float((cube_bottom - table_surface).min()):.6f} m)."
                )

            held_recipe_ids = self._recipe_ids[held_rows]
            first_place = held_recipe_ids == int(StackResetRecipe.FIRST_PLACE)
            if bool(torch.any(first_place)):
                support_top = self._role_positions[held_rows[first_place], 0, 2] + 0.5 * self._CUBE_HEIGHT
                clearance = cube_bottom[first_place] - support_top
                required_margin = clearance.new_full(
                    clearance.shape,
                    self._FIRST_PLACE_SUPPORT_MARGIN,
                )
                invalid = clearance < required_margin - 3.0e-5
                if bool(torch.any(invalid)):
                    invalid_rows = held_rows[first_place][invalid]
                    raise RuntimeError(
                        "A rotated first-place cube violates its validated support margin "
                        f"(rows={invalid_rows[:8].tolist()}, count={int(invalid.sum())}, "
                        f"minimum={float(clearance.min()):.6f} m)."
                    )

            second_place = held_recipe_ids == int(StackResetRecipe.SECOND_PLACE)
            if bool(torch.any(second_place)):
                support_top = self._role_positions[held_rows[second_place], 1, 2] + 0.5 * self._CUBE_HEIGHT
                clearance = cube_bottom[second_place] - support_top
                invalid = clearance < self._SECOND_PLACE_SUPPORT_MARGIN - 3.0e-5
                if bool(torch.any(invalid)):
                    invalid_rows = held_rows[second_place][invalid]
                    raise RuntimeError(
                        "A rotated second-place cube violates its validated support margin "
                        f"(rows={invalid_rows[:8].tolist()}, count={int(invalid.sum())}, "
                        f"minimum={float(clearance.min()):.6f} m)."
                    )

    def _held_pose(
        self,
        arm_position: torch.Tensor,
        grasp_pair_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full rigid pair-center-to-held-cube SE(3) pose."""
        held_position, palm_rotation = self._grasp_pair_pose(arm_position, grasp_pair_ids)
        palm_to_cube = self._palm_to_held_cube_rotations(grasp_pair_ids, arm_position)
        held_rotation = torch.matmul(palm_rotation, palm_to_cube)
        return held_position, quaternion_xyzw_from_matrix(held_rotation)

    def _pick_endpoint_heights(
        self,
        grasp_pair_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the screened carry-floor endpoint for every held pair."""
        return reference.new_full(grasp_pair_ids.shape, self._transport_minimum_height())

    @classmethod
    def _target_potential(cls, recipe: StackResetRecipe) -> float:
        """Split table acquisition from the retained 25-mm lift bridge.

        TABLE starts first discover a real grasp, then advance through a
        sparse lift ladder. FIRST_PICK rows retain the independently screened
        25-mm bridge endpoint. Keeping those targets separate avoids
        withholding all TABLE credit until two distinct skills happen in the
        same exploration trajectory.
        """
        if recipe == StackResetRecipe.TABLE:
            return 1.05
        if recipe == StackResetRecipe.FIRST_PICK:
            return 1.25
        if recipe in (StackResetRecipe.PAIR_READY, StackResetRecipe.SECOND_PICK):
            return 6.25
        return cls._default_target_potential(recipe)

    @classmethod
    def _sample_layout(cls, sample_id: int, *, table_start: bool) -> tuple[tuple[float, float], ...]:
        """Reject semantic layouts that make a later tilted carry intersect.

        Canonicalizing the order can turn either movable cube into the second
        source. Keep both sources at least 13 cm from the base so the first
        SECOND_TRANSPORT carry remains valid under its independent broad tilt
        augmentation. TABLE layouts use their separate corridor planner.
        """
        if table_start:
            return cls._sample_layout_candidate(sample_id, table_start=True)
        for attempt in range(64):
            candidate_id = sample_id + attempt * cls._SEMANTIC_LAYOUT_COUNT
            layout = cls._sample_layout_candidate(candidate_id, table_start=False)
            if all(
                math.dist(layout[0], layout[role_id]) >= cls._MINIMUM_SEMANTIC_BASE_SOURCE_DISTANCE
                for role_id in (1, 2)
            ):
                return layout
        raise RuntimeError("Full-hand semantic layout rejection could not clear both movable cubes from the base.")

    @classmethod
    def _orientation_ids_for_recipe(cls, recipe: StackResetRecipe) -> tuple[int, ...]:
        """Use the collision-free, kinematically valid index-contact yaws."""
        if recipe in (StackResetRecipe.FIRST_PICK, StackResetRecipe.PAIR_READY, StackResetRecipe.SECOND_PICK):
            # Newton's residual acquisition failures were concentrated almost
            # entirely in yaw three. Yaws two and six retain both side-pinch
            # directions and clear every canonicalized 10 cm approach.
            return (2, 6)
        if recipe == StackResetRecipe.TABLE:
            return (2, 3, 6)
        return cls._default_orientation_ids_for_recipe(recipe)

    @staticmethod
    def _corridor_center_clearance(
        pair_positions: torch.Tensor,
        palm_offsets: torch.Tensor,
        obstacle_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Return worst bridge clearance for every layout/yaw candidate.

        ``pair_positions`` has shape ``(layout, yaw, sample, xyz)`` and
        ``palm_offsets`` is the palm-to-pair vector for each yaw. The selected
        centerline proxy is deliberately the same conservative signal that
        separated stable and unstable Newton reset rows in the dynamic audit.
        """
        palm_positions = pair_positions - palm_offsets[None, :, None]
        bridge_vectors = pair_positions - palm_positions
        palm_to_obstacle = obstacle_centers[:, None, None, :, :] - palm_positions[:, :, :, None, :]
        projection = torch.sum(palm_to_obstacle * bridge_vectors[:, :, :, None, :], dim=4)
        projection /= torch.sum(torch.square(bridge_vectors), dim=3, keepdim=True).clamp_min(1.0e-12)
        projection.clamp_(0.0, 1.0)
        closest_points = palm_positions[:, :, :, None, :] + projection.unsqueeze(4) * bridge_vectors[:, :, :, None, :]
        distances = torch.linalg.vector_norm(
            obstacle_centers[:, None, None, :, :] - closest_points,
            dim=4,
        )
        return distances.amin(dim=(2, 3))

    def _prepare_semantic_reset_plans(self, semantic_layouts: torch.Tensor) -> None:
        """Canonicalize movable roles and choose collision-clearest yaws.

        The task is order-invariant, so each physical layout may make either
        loose cube logical role one. Score both orders over the two complete
        open acquisition corridors, impose an exact 50/50 physical-role quota,
        and swap semantic roles one/two when the second physical cube is the
        safer first target. All downstream recipes then use the same unchanged
        logical stack sequence. PAIR_READY and SECOND_PICK share the logical
        role-two yaw, preserving their exact endpoint.
        """
        if semantic_layouts.shape != (self._SEMANTIC_LAYOUT_COUNT, 3, 2):
            raise RuntimeError("Full-hand semantic layouts have an unexpected shape.")
        orientation_choices = torch.tensor(
            self._orientation_ids_for_recipe(StackResetRecipe.FIRST_PICK),
            dtype=torch.long,
            device=self.device,
        )
        if tuple(orientation_choices.tolist()) != (2, 6):
            raise RuntimeError("Full-hand semantic corridor planning expects the screened yaw bins (2, 6).")
        candidate_rotations, _ = self._target_wrist_rotations(
            orientation_choices,
            torch.zeros(
                orientation_choices.numel(),
                dtype=self._arm_anchors.dtype,
                device=self.device,
            ),
        )
        tool_offset = self._arm_anchors.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS[0])
        palm_offsets = torch.matmul(candidate_rotations, tool_offset.view(1, 3, 1)).squeeze(2)
        approach_axes = palm_offsets / torch.linalg.vector_norm(
            palm_offsets,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)

        original_cube_centers = torch.cat(
            (
                semantic_layouts,
                semantic_layouts.new_full((self._SEMANTIC_LAYOUT_COUNT, 3, 1), self._TABLE_HEIGHT),
            ),
            dim=2,
        )
        base_positions = original_cube_centers[:, 0]
        movable_sources = original_cube_centers[:, 1:3]
        candidate_first_sources = movable_sources
        candidate_second_sources = torch.flip(movable_sources, dims=(1,))
        candidate_base_positions = base_positions[:, None].expand(-1, 2, -1)

        bridge_distances = torch.linspace(
            self._OBJECT_AXIS_APPROACH_DISTANCE,
            0.0,
            9,
            dtype=self._arm_anchors.dtype,
            device=self.device,
        )
        flattened_base = candidate_base_positions.reshape(-1, 3)
        flattened_first = candidate_first_sources.reshape(-1, 3)
        flattened_second = candidate_second_sources.reshape(-1, 3)

        # FIRST_PICK: selected cube excluded; base and other loose cube remain.
        first_contact = flattened_first.clone()
        first_contact[:, 2] = self._pick_contact_height()
        first_pick_positions = first_contact[:, None, None, :] - (
            bridge_distances[None, None, :, None] * approach_axes[None, :, None, :]
        )
        first_obstacles = torch.stack((flattened_base, flattened_second), dim=1)
        candidate_first_bridge_clearances = self._corridor_center_clearance(
            first_pick_positions,
            palm_offsets,
            first_obstacles,
        ).reshape(self._SEMANTIC_LAYOUT_COUNT, 2, -1)
        candidate_first_clearances = self._corridor_center_clearance(
            first_pick_positions[:, :, :1],
            palm_offsets,
            first_obstacles,
        ).reshape(self._SEMANTIC_LAYOUT_COUNT, 2, -1)

        # PAIR_READY's late endpoint and SECOND_PICK share this exact role-two
        # acquisition corridor. Do not include the yaw-invariant early bridge
        # sample in this optimization: it previously forced a meaningless tie.
        second_contact = flattened_second.clone()
        second_contact[:, 2] = self._pick_contact_height()
        second_pick_positions = second_contact[:, None, None, :] - (
            bridge_distances[None, None, :, None] * approach_axes[None, :, None, :]
        )
        flattened_first_stack = flattened_base + flattened_base.new_tensor((0.0, 0.0, self._CUBE_HEIGHT))
        second_obstacles = torch.stack((flattened_base, flattened_first_stack), dim=1)
        candidate_second_bridge_clearances = self._corridor_center_clearance(
            second_pick_positions,
            palm_offsets,
            second_obstacles,
        ).reshape(self._SEMANTIC_LAYOUT_COUNT, 2, -1)
        candidate_second_clearances = self._corridor_center_clearance(
            second_pick_positions[:, :, :1],
            palm_offsets,
            second_obstacles,
        ).reshape(self._SEMANTIC_LAYOUT_COUNT, 2, -1)

        best_first_by_order = candidate_first_clearances.amax(dim=2)
        best_second_by_order = candidate_second_clearances.amax(dim=2)
        acquisition_clearance_by_order = torch.minimum(best_first_by_order, best_second_by_order)
        # A low second-source/base separation is harmless while both cubes are
        # supported, but the first tilted SECOND_TRANSPORT rows can overlap the
        # completed first stack. Use that separation as a canonical-order
        # tie-break without narrowing the later eight-yaw augmentation.
        second_source_base_distance = torch.linalg.vector_norm(
            candidate_second_sources - candidate_base_positions,
            dim=2,
        )
        self._semantic_acquisition_clearance_by_order = acquisition_clearance_by_order
        self._semantic_second_source_base_distance_by_order = second_source_base_distance
        order_score = acquisition_clearance_by_order.clone()
        acquisition_feasible = acquisition_clearance_by_order >= self._MINIMUM_ACQUISITION_CORRIDOR_CENTER_DISTANCE
        order_score[~acquisition_feasible] -= 10.0
        transport_feasible = second_source_base_distance >= self._MINIMUM_SEMANTIC_BASE_SOURCE_DISTANCE
        has_transport_feasible_order = torch.any(transport_feasible, dim=1, keepdim=True)
        order_score[has_transport_feasible_order & ~transport_feasible] -= 1.0
        order_advantage = order_score[:, 0] - order_score[:, 1]
        ranked_layouts = torch.argsort(order_advantage, descending=True, stable=True)
        selected_order_ranks = torch.ones(self._SEMANTIC_LAYOUT_COUNT, dtype=torch.long, device=self.device)
        selected_order_ranks[ranked_layouts[: self._SEMANTIC_LAYOUT_COUNT // 2]] = 0
        self._semantic_first_physical_role_ids = selected_order_ranks + 1

        selected_first_sources = candidate_first_sources.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, 3),
        ).squeeze(1)
        selected_second_sources = candidate_second_sources.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, 3),
        ).squeeze(1)
        semantic_layouts[:, 1] = selected_first_sources[:, :2]
        semantic_layouts[:, 2] = selected_second_sources[:, :2]

        selected_first_clearances = candidate_first_clearances.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, orientation_choices.numel()),
        ).squeeze(1)
        selected_second_clearances = candidate_second_clearances.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, orientation_choices.numel()),
        ).squeeze(1)
        selected_first_bridge_clearances = candidate_first_bridge_clearances.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, orientation_choices.numel()),
        ).squeeze(1)
        selected_second_bridge_clearances = candidate_second_bridge_clearances.gather(
            1,
            selected_order_ranks.view(-1, 1, 1).expand(-1, 1, orientation_choices.numel()),
        ).squeeze(1)
        first_choice_ranks = torch.argmax(selected_first_clearances, dim=1)
        second_choice_ranks = torch.argmax(selected_second_clearances, dim=1)
        self._first_pick_orientation_ids_by_layout = orientation_choices[first_choice_ranks]
        self._first_pick_corridor_center_distances = selected_first_clearances.gather(
            1,
            first_choice_ranks.unsqueeze(1),
        ).squeeze(1)
        self._second_pick_orientation_ids_by_layout = orientation_choices[second_choice_ranks]
        self._second_pick_corridor_center_distances = selected_second_clearances.gather(
            1,
            second_choice_ranks.unsqueeze(1),
        ).squeeze(1)
        self._first_pick_bridge_center_distances = selected_first_bridge_clearances.gather(
            1,
            first_choice_ranks.unsqueeze(1),
        ).squeeze(1)
        self._second_pick_bridge_center_distances = selected_second_bridge_clearances.gather(
            1,
            second_choice_ranks.unsqueeze(1),
        ).squeeze(1)
        minimum_clearance = min(
            float(self._first_pick_corridor_center_distances.min()),
            float(self._second_pick_corridor_center_distances.min()),
        )
        if minimum_clearance < self._MINIMUM_ACQUISITION_CORRIDOR_CENTER_DISTANCE:
            raise RuntimeError(
                "Full-hand semantic acquisition plan violates its minimum centerline clearance "
                f"({minimum_clearance:.6f} < {self._MINIMUM_ACQUISITION_CORRIDOR_CENTER_DISTANCE:.6f} m)."
            )

    def _semantic_orientation_assignments(
        self,
        recipe: StackResetRecipe,
        layout_ids: torch.Tensor,
        default_orientation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply layout-safe yaws only to the open acquisition recipes."""
        if recipe == StackResetRecipe.FIRST_PICK:
            return self._first_pick_orientation_ids_by_layout[layout_ids]
        if recipe in (StackResetRecipe.PAIR_READY, StackResetRecipe.SECOND_PICK):
            return self._second_pick_orientation_ids_by_layout[layout_ids]
        return default_orientation_ids

    def _final_release_non_tip_offsets(
        self,
        release_rows: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return index/middle link-2 proxy origins in the palm frame [m]."""
        progress = self._progress[release_rows].to(reference.dtype).unsqueeze(1)
        index_closed = reference.new_tensor(self._FINAL_RELEASE_INDEX_LINK_2_CLOSED_OFFSET)
        index_open = reference.new_tensor(self._FINAL_RELEASE_INDEX_LINK_2_OPEN_OFFSET)
        index_offsets = torch.lerp(index_closed, index_open, progress)
        middle_offsets = reference.new_tensor(self._FINAL_RELEASE_MIDDLE_LINK_2_OFFSET).expand(
            release_rows.numel(),
            -1,
        )
        return torch.stack((index_offsets, middle_offsets), dim=1)

    def _final_release_stack_clearance(
        self,
        release_rows: torch.Tensor,
        palm_rotations: torch.Tensor,
        tool_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return release poses whose non-tip hand envelope clears the stack."""
        if palm_rotations.ndim != 4 or palm_rotations.shape[:2] != tool_positions.shape[:2]:
            raise ValueError("FINAL_RELEASE candidate rotations and tool positions must share (row, candidate).")
        candidate_count = palm_rotations.shape[1]
        pair_ids = self._grasp_pair_ids[release_rows]
        tool_offsets = tool_positions.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS)[pair_ids]
        palm_positions = tool_positions - torch.matmul(
            palm_rotations,
            tool_offsets[:, None, :, None],
        ).squeeze(-1)

        local_non_tip_offsets = self._final_release_non_tip_offsets(release_rows, tool_positions)
        non_tip_positions = palm_positions[:, :, None] + torch.matmul(
            palm_rotations[:, :, None],
            local_non_tip_offsets[:, None, :, :, None],
        ).squeeze(-1)

        palm_to_cube = self._palm_to_held_cube_rotations(pair_ids, tool_positions)
        contact_rotations = torch.matmul(
            palm_rotations,
            palm_to_cube[:, None],
        )
        top_yaw = torch.atan2(contact_rotations[..., 1, 0], contact_rotations[..., 0, 0])
        cosine, sine = torch.cos(top_yaw), torch.sin(top_yaw)
        zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
        top_rotations = torch.stack(
            (
                cosine,
                -sine,
                zeros,
                sine,
                cosine,
                zeros,
                zeros,
                zeros,
                ones,
            ),
            dim=-1,
        ).reshape(release_rows.numel(), candidate_count, 3, 3)
        top_centers = self._role_positions[release_rows, 2]
        top_center_candidates = top_centers[:, None].expand(-1, candidate_count, -1)
        index_top_intersections = _segment_oriented_box_intersections(
            non_tip_positions[:, :, 0].reshape(-1, 3),
            non_tip_positions[:, :, 0].reshape(-1, 3),
            top_center_candidates.reshape(-1, 3),
            top_rotations.reshape(-1, 3, 3),
            half_extent=0.5 * self._CUBE_HEIGHT + self._FINAL_RELEASE_INDEX_LINK_2_CLEARANCE,
        ).reshape(release_rows.numel(), candidate_count)
        middle_top_intersections = _segment_oriented_box_intersections(
            non_tip_positions[:, :, 1].reshape(-1, 3),
            non_tip_positions[:, :, 1].reshape(-1, 3),
            top_center_candidates.reshape(-1, 3),
            top_rotations.reshape(-1, 3, 3),
            half_extent=0.5 * self._CUBE_HEIGHT + self._FINAL_RELEASE_MIDDLE_LINK_2_CLEARANCE,
        ).reshape(release_rows.numel(), candidate_count)

        lower_centers = self._role_positions[release_rows, :2]
        lower_rotations = self._upright_role_rotations(
            release_rows,
            torch.arange(2, device=self.device),
            tool_positions,
        )
        lower_center_candidates = lower_centers[:, None].expand(-1, candidate_count, -1, -1)
        lower_rotation_candidates = lower_rotations[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
            -1,
        )

        segment_starts = palm_positions[:, :, None].expand(-1, -1, 2, -1)
        segment_ends = tool_positions[:, :, None].expand_as(segment_starts)
        palm_bridge_intersections = _segment_oriented_box_intersections(
            segment_starts.reshape(-1, 3),
            segment_ends.reshape(-1, 3),
            lower_center_candidates.reshape(-1, 3),
            lower_rotation_candidates.reshape(-1, 3, 3),
            half_extent=0.5 * self._CUBE_HEIGHT + self._FINAL_RELEASE_LOWER_STACK_HAND_CLEARANCE,
        ).reshape(release_rows.numel(), candidate_count, 2)
        return ~(index_top_intersections | middle_top_intersections | torch.any(palm_bridge_intersections, dim=2))

    def _repair_final_release_clearance(
        self,
        arm_positions: torch.Tensor,
        position_residuals: torch.Tensor,
        rotation_residuals: torch.Tensor,
        target_positions: torch.Tensor,
        target_rotations: torch.Tensor,
        tool_offsets: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repair release poses whose non-tip hand envelope intersects the stack."""
        release_rows = torch.nonzero(
            self._recipe_ids == int(StackResetRecipe.FINAL_RELEASE),
            as_tuple=False,
        ).flatten()
        if release_rows.numel() == 0:
            self._ik_final_release_repair_count = 0
            self._ik_final_release_orientation_repair_count = 0
            self._final_release_retreat_lifts = torch.zeros(
                target_positions.shape[0],
                dtype=target_positions.dtype,
                device=self.device,
            )
            return arm_positions, position_residuals, rotation_residuals, target_rotations
        actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
            arm_positions[release_rows],
            self._grasp_pair_ids[release_rows],
        )
        clear = self._final_release_stack_clearance(
            release_rows,
            actual_palm_rotations.unsqueeze(1),
            actual_tool_positions.unsqueeze(1),
        ).squeeze(1)
        repair_rows = release_rows[~clear]
        self._ik_final_release_repair_count = int(repair_rows.numel())
        self._final_release_retreat_lifts = torch.zeros(
            target_positions.shape[0],
            dtype=target_positions.dtype,
            device=self.device,
        )
        if repair_rows.numel() == 0:
            self._ik_final_release_orientation_repair_count = 0
            return arm_positions, position_residuals, rotation_residuals, target_rotations

        original_azimuth_ids = self._tilt_azimuth_ids[repair_rows].clone()
        original_tilt_angles = self._resolved_tilt_angles[repair_rows].clone()
        azimuth_shifts = repair_rows.new_tensor((0, 1, -1, 2, -2, 3, -3, 4))
        tilt_scales = target_positions.new_tensor((1.0, 0.75, 0.50, 0.25, 0.0))
        candidate_azimuth_ids = torch.remainder(
            self._tilt_azimuth_ids[repair_rows, None, None] + azimuth_shifts[None, :, None],
            8,
        ).expand(-1, -1, tilt_scales.numel())
        candidate_tilt_angles = (self._resolved_tilt_angles[repair_rows, None, None] * tilt_scales[None, None]).expand(
            -1, azimuth_shifts.numel(), -1
        )
        candidate_azimuth_ids = candidate_azimuth_ids.flatten(start_dim=1)
        candidate_tilt_angles = candidate_tilt_angles.flatten(start_dim=1)
        orientation_count = candidate_azimuth_ids.shape[1]

        retreat_lifts = target_positions.new_tensor(self._FINAL_RELEASE_RETREAT_CANDIDATES)
        retreat_count = retreat_lifts.numel()
        candidate_count = orientation_count * retreat_count
        candidate_azimuth_ids = (
            candidate_azimuth_ids[:, :, None]
            .expand(-1, -1, retreat_count)
            .reshape(
                repair_rows.numel(),
                candidate_count,
            )
        )
        candidate_tilt_angles = (
            candidate_tilt_angles[:, :, None]
            .expand(-1, -1, retreat_count)
            .reshape(
                repair_rows.numel(),
                candidate_count,
            )
        )
        candidate_lifts = (
            retreat_lifts[None, None]
            .expand(
                repair_rows.numel(),
                orientation_count,
                -1,
            )
            .reshape(repair_rows.numel(), candidate_count)
        )
        flat_row_ids = repair_rows[:, None].expand(-1, candidate_count).reshape(-1)
        candidate_rotations, _ = self._target_wrist_rotations(
            self._orientation_ids[flat_row_ids],
            candidate_tilt_angles.reshape(-1),
            candidate_azimuth_ids.reshape(-1),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_rotations = candidate_rotations.reshape(repair_rows.numel(), candidate_count, 3, 3)
        candidate_targets = target_positions[repair_rows, None].expand(-1, candidate_count, -1).clone()
        candidate_targets[:, :, 2] += candidate_lifts
        candidate_seeds = arm_positions[repair_rows, None].expand(-1, candidate_count, -1).clone()
        (
            candidate_arm_positions,
            candidate_position_residuals,
            candidate_rotation_residuals,
        ) = solve_kuka_allegro_reset_ik(
            candidate_seeds.reshape(-1, 7),
            candidate_targets.reshape(-1, 3),
            candidate_rotations.reshape(-1, 3, 3),
            tool_offsets[flat_row_ids],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            max_iterations=96,
        )
        candidate_arm_positions = candidate_arm_positions.reshape(repair_rows.numel(), candidate_count, 7)
        candidate_position_residuals = candidate_position_residuals.reshape(repair_rows.numel(), candidate_count)
        candidate_rotation_residuals = candidate_rotation_residuals.reshape(repair_rows.numel(), candidate_count)
        candidate_tool_positions, candidate_palm_rotations = self._grasp_pair_pose(
            candidate_arm_positions.reshape(-1, 7),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_tool_positions = candidate_tool_positions.reshape(repair_rows.numel(), candidate_count, 3)
        candidate_palm_rotations = candidate_palm_rotations.reshape(repair_rows.numel(), candidate_count, 3, 3)
        valid = (
            (candidate_position_residuals <= 8.0e-4)
            & (candidate_rotation_residuals <= 1.0e-2)
            & self._final_release_stack_clearance(
                repair_rows,
                candidate_palm_rotations,
                candidate_tool_positions,
            )
        )
        relative_rotations = torch.matmul(
            candidate_rotations,
            target_rotations[repair_rows, None].transpose(-1, -2),
        )
        cosine_distance = 0.5 * (torch.diagonal(relative_rotations, dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
        orientation_deviation = torch.acos(torch.clamp(cosine_distance, -1.0, 1.0))
        geodesic_distance = orientation_deviation.masked_fill(~valid, torch.inf)
        best_distance = torch.amin(geodesic_distance, dim=1)
        nearest_orientation = valid & (orientation_deviation <= best_distance[:, None] + 1.0e-6)
        retreat_scores = candidate_lifts.masked_fill(~nearest_orientation, torch.inf)
        best_lifts, best_candidate_ids = torch.min(retreat_scores, dim=1)
        if not bool(torch.all(torch.isfinite(best_distance))):
            unresolved_rows = repair_rows[~torch.isfinite(best_distance)]
            raise RuntimeError(
                "Full-hand FINAL_RELEASE reset IK has no orientation/retreat candidate that clears the stack "
                f"(count={unresolved_rows.numel()}, rows={unresolved_rows[:16].tolist()})."
            )

        row_indices = torch.arange(repair_rows.numel(), device=self.device)
        arm_positions[repair_rows] = candidate_arm_positions[row_indices, best_candidate_ids]
        position_residuals[repair_rows] = candidate_position_residuals[row_indices, best_candidate_ids]
        rotation_residuals[repair_rows] = candidate_rotation_residuals[row_indices, best_candidate_ids]
        target_rotations[repair_rows] = candidate_rotations[row_indices, best_candidate_ids]
        self._tilt_azimuth_ids[repair_rows] = candidate_azimuth_ids[row_indices, best_candidate_ids]
        self._resolved_tilt_angles[repair_rows] = candidate_tilt_angles[row_indices, best_candidate_ids]
        self._final_release_retreat_lifts[repair_rows] = best_lifts
        self._ik_final_release_orientation_repair_count = int(
            torch.sum(
                (self._tilt_azimuth_ids[repair_rows] != original_azimuth_ids)
                | (torch.abs(self._resolved_tilt_angles[repair_rows] - original_tilt_angles) > 1.0e-6)
            )
        )
        return arm_positions, position_residuals, rotation_residuals, target_rotations

    def _solve_diverse_arm_targets(
        self,
        seed_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_rotations: torch.Tensor,
        tool_offsets: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repair release clearances, then stack-blocked second carries."""
        (
            arm_positions,
            position_residuals,
            rotation_residuals,
            target_rotations,
        ) = self._solve_arm_targets(
            seed_positions,
            target_positions,
            target_rotations,
            tool_offsets,
            joint_lower,
            joint_upper,
        )
        (
            arm_positions,
            position_residuals,
            rotation_residuals,
            target_rotations,
        ) = self._repair_final_release_clearance(
            arm_positions,
            position_residuals,
            rotation_residuals,
            target_positions,
            target_rotations,
            tool_offsets,
            joint_lower,
            joint_upper,
        )
        actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
            arm_positions,
            self._grasp_pair_ids,
        )
        second_transport_rows = torch.nonzero(
            self._recipe_ids == int(StackResetRecipe.SECOND_TRANSPORT),
            as_tuple=False,
        ).flatten()
        collision_free = self._collision_free_wrist_rotations(
            second_transport_rows,
            target_positions,
            actual_palm_rotations[second_transport_rows].unsqueeze(1),
            actual_tool_positions[second_transport_rows].unsqueeze(1),
        ).squeeze(1)
        repair_rows = second_transport_rows[~collision_free]
        self._ik_yaw_stack_repair_count = int(repair_rows.numel())
        if repair_rows.numel() == 0:
            return arm_positions, position_residuals, rotation_residuals, target_rotations

        yaw_shifts = repair_rows.new_tensor((0, 1, -1, 2, -2, 3, -3, 4))
        azimuth_shifts = repair_rows.new_tensor((0, 1, -1, 2, -2, 3, -3, 4))
        tilt_choices = torch.deg2rad(target_positions.new_tensor((0.0, 5.0, 15.0, 30.0, 45.0)))
        yaw_ids = torch.remainder(
            self._orientation_ids[repair_rows, None, None, None] + yaw_shifts[None, :, None, None],
            8,
        ).expand(-1, -1, azimuth_shifts.numel(), tilt_choices.numel())
        azimuth_ids = torch.remainder(
            self._tilt_azimuth_ids[repair_rows, None, None, None] + azimuth_shifts[None, None, :, None],
            8,
        ).expand(-1, yaw_shifts.numel(), -1, tilt_choices.numel())
        tilt_angles = tilt_choices[None, None, None].expand_as(yaw_ids)
        candidate_yaw_ids = torch.cat(
            (self._orientation_ids[repair_rows, None], yaw_ids.flatten(start_dim=1)),
            dim=1,
        )
        candidate_azimuth_ids = torch.cat(
            (self._tilt_azimuth_ids[repair_rows, None], azimuth_ids.flatten(start_dim=1)),
            dim=1,
        )
        candidate_tilt_angles = torch.cat(
            (self._resolved_tilt_angles[repair_rows, None], tilt_angles.flatten(start_dim=1)),
            dim=1,
        )
        candidate_count = candidate_yaw_ids.shape[1]
        flat_row_ids = repair_rows[:, None].expand(-1, candidate_count).reshape(-1)
        candidate_rotations, candidate_yaw_angles = self._target_wrist_rotations(
            candidate_yaw_ids.reshape(-1),
            candidate_tilt_angles.reshape(-1),
            candidate_azimuth_ids.reshape(-1),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_rotations = candidate_rotations.reshape(repair_rows.numel(), candidate_count, 3, 3)
        candidate_yaw_angles = candidate_yaw_angles.reshape(repair_rows.numel(), candidate_count)

        candidate_seeds = arm_positions[repair_rows, None].expand(-1, candidate_count, -1).clone()
        _, current_yaw_angles = self._target_wrist_rotations(
            self._orientation_ids[repair_rows],
            self._resolved_tilt_angles[repair_rows],
            self._tilt_azimuth_ids[repair_rows],
            self._grasp_pair_ids[repair_rows],
        )
        candidate_seeds[:, :, 6] -= candidate_yaw_angles - current_yaw_angles.unsqueeze(1)
        candidate_seeds[:, :, 6] = torch.remainder(candidate_seeds[:, :, 6] + math.pi, 2.0 * math.pi) - math.pi
        candidate_seeds.clamp_(min=joint_lower, max=joint_upper)
        (
            candidate_arm_positions,
            candidate_position_residuals,
            candidate_rotation_residuals,
        ) = solve_kuka_allegro_reset_ik(
            candidate_seeds.reshape(-1, 7),
            target_positions[flat_row_ids],
            candidate_rotations.reshape(-1, 3, 3),
            tool_offsets[flat_row_ids],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            max_iterations=96,
        )
        candidate_arm_positions = candidate_arm_positions.reshape(repair_rows.numel(), candidate_count, 7)
        candidate_position_residuals = candidate_position_residuals.reshape(repair_rows.numel(), candidate_count)
        candidate_rotation_residuals = candidate_rotation_residuals.reshape(repair_rows.numel(), candidate_count)
        candidate_tool_positions, candidate_palm_rotations = self._grasp_pair_pose(
            candidate_arm_positions.reshape(-1, 7),
            self._grasp_pair_ids[flat_row_ids],
        )
        candidate_tool_positions = candidate_tool_positions.reshape(repair_rows.numel(), candidate_count, 3)
        candidate_palm_rotations = candidate_palm_rotations.reshape(repair_rows.numel(), candidate_count, 3, 3)
        valid = (
            (candidate_tilt_angles <= self._resolved_tilt_limits[repair_rows, None] + 1.0e-6)
            & (candidate_position_residuals <= 8.0e-4)
            & (candidate_rotation_residuals <= 1.0e-2)
            & self._collision_free_wrist_rotations(
                repair_rows,
                target_positions,
                candidate_palm_rotations,
                candidate_tool_positions,
            )
        )
        relative_rotations = torch.matmul(
            candidate_rotations,
            target_rotations[repair_rows, None].transpose(-1, -2),
        )
        cosine_distance = 0.5 * (torch.diagonal(relative_rotations, dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
        orientation_deviation = torch.acos(torch.clamp(cosine_distance, -1.0, 1.0))
        geodesic_distance = orientation_deviation.masked_fill(~valid, torch.inf)
        best_distance, best_candidate_ids = torch.min(geodesic_distance, dim=1)
        if not bool(torch.all(torch.isfinite(best_distance))):
            unresolved_rows = repair_rows[~torch.isfinite(best_distance)]
            raise RuntimeError(
                "Full-hand SECOND_TRANSPORT reset IK has no yaw/tilt candidate that clears the stack "
                f"(count={unresolved_rows.numel()}, rows={unresolved_rows[:16].tolist()})."
            )

        row_indices = torch.arange(repair_rows.numel(), device=self.device)
        arm_positions[repair_rows] = candidate_arm_positions[row_indices, best_candidate_ids]
        position_residuals[repair_rows] = candidate_position_residuals[row_indices, best_candidate_ids]
        rotation_residuals[repair_rows] = candidate_rotation_residuals[row_indices, best_candidate_ids]
        target_rotations[repair_rows] = candidate_rotations[row_indices, best_candidate_ids]
        self._orientation_ids[repair_rows] = candidate_yaw_ids[row_indices, best_candidate_ids]
        self._tilt_azimuth_ids[repair_rows] = candidate_azimuth_ids[row_indices, best_candidate_ids]
        self._resolved_tilt_angles[repair_rows] = candidate_tilt_angles[row_indices, best_candidate_ids]
        return arm_positions, position_residuals, rotation_residuals, target_rotations

    @staticmethod
    def _upright_role_rotations(
        row_ids: torch.Tensor,
        role_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic upright cube rotations authored per row."""
        yaw_fraction = torch.remainder(
            row_ids[:, None] * 193 + role_ids[None, :] * 389 + 17,
            4093,
        ).to(reference.dtype)
        yaw = math.pi * (2.0 * yaw_fraction / 4092.0 - 1.0)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
        return torch.stack(
            (
                cosine,
                -sine,
                zeros,
                sine,
                cosine,
                zeros,
                zeros,
                zeros,
                ones,
            ),
            dim=-1,
        ).reshape(row_ids.numel(), role_ids.numel(), 3, 3)

    def _collision_free_wrist_rotations(
        self,
        row_ids: torch.Tensor,
        target_positions: torch.Tensor,
        palm_rotations: torch.Tensor,
        tool_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Also keep a held second cube and the swept hand clear of the stack."""
        valid = self._screen_wrist_rotations(
            row_ids,
            target_positions,
            palm_rotations,
            tool_positions,
        )
        second_transport = self._recipe_ids[row_ids] == int(StackResetRecipe.SECOND_TRANSPORT)
        if not bool(torch.any(second_transport)):
            return valid

        transport_rows = row_ids[second_transport]
        candidate_count = palm_rotations.shape[1]
        if tool_positions is None:
            transport_tool_positions = target_positions[transport_rows, None].expand(-1, candidate_count, -1)
        else:
            transport_tool_positions = tool_positions[second_transport]
        # Low early-transport targets are lifted later so a tilted held cube
        # clears the table. Screen the pose at that final height now; checking
        # only the pre-rewind target can select a hand orientation that rises
        # directly into the completed first stack during finalization.
        final_clearance_lift = self._held_cube_clearance_lift(
            transport_rows,
            target_positions,
            palm_rotations[second_transport],
            transport_tool_positions,
            clearance_margin=4.0 * self._RESET_CLEARANCE_MARGIN,
        )
        transport_tool_positions = transport_tool_positions.clone()
        transport_tool_positions[:, :, 2] += final_clearance_lift

        transport_valid = self._second_transport_stack_clearance(
            transport_rows,
            palm_rotations[second_transport],
            transport_tool_positions,
            hand_clearance=(self._SECOND_TRANSPORT_HAND_STACK_CLEARANCE + self._SECOND_TRANSPORT_IK_REPAIR_BUFFER),
            cube_clearance=(self._SECOND_TRANSPORT_CUBE_STACK_CLEARANCE + self._SECOND_TRANSPORT_IK_REPAIR_BUFFER),
        )
        valid[second_transport] &= transport_valid
        return valid

    def _second_transport_stack_clearance(
        self,
        transport_rows: torch.Tensor,
        palm_rotations: torch.Tensor,
        tool_positions: torch.Tensor,
        *,
        hand_clearance: float,
        cube_clearance: float,
    ) -> torch.Tensor:
        """Return candidate poses whose cube and palm envelope clear the stack."""
        candidate_count = palm_rotations.shape[1]
        pair_ids = self._grasp_pair_ids[transport_rows]
        palm_to_cube = self._palm_to_held_cube_rotations(pair_ids, self._arm_anchors)
        held_rotations = torch.matmul(
            palm_rotations,
            palm_to_cube.unsqueeze(1),
        )
        obstacle_centers = self._role_positions[transport_rows, :2]
        obstacle_rotations = self._upright_role_rotations(
            transport_rows,
            torch.arange(2, device=self.device),
            tool_positions,
        )

        held_centers = tool_positions[:, :, None, :].expand(-1, -1, 2, -1)
        held_rotation_candidates = held_rotations[:, :, None].expand(-1, -1, 2, -1, -1)
        obstacle_center_candidates = obstacle_centers[:, None].expand(-1, candidate_count, -1, -1)
        obstacle_rotation_candidates = obstacle_rotations[:, None].expand(-1, candidate_count, -1, -1, -1)
        held_stack_intersections = _oriented_cube_pair_intersections(
            held_centers.reshape(-1, 3),
            held_rotation_candidates.reshape(-1, 3, 3),
            obstacle_center_candidates.reshape(-1, 3),
            obstacle_rotation_candidates.reshape(-1, 3, 3),
            edge_length=self._CUBE_HEIGHT + cube_clearance,
        ).reshape(transport_rows.numel(), candidate_count, 2)

        tool_offsets = tool_positions.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS)[pair_ids]
        palm_positions = tool_positions - torch.matmul(
            palm_rotations,
            tool_offsets[:, None, :, None],
        ).squeeze(-1)
        segment_starts = palm_positions[:, :, None].expand(-1, -1, 2, -1)
        segment_ends = tool_positions[:, :, None, :].expand_as(segment_starts)
        hand_stack_intersections = _segment_oriented_box_intersections(
            segment_starts.reshape(-1, 3),
            segment_ends.reshape(-1, 3),
            obstacle_center_candidates.reshape(-1, 3),
            obstacle_rotation_candidates.reshape(-1, 3, 3),
            half_extent=0.5 * self._CUBE_HEIGHT + hand_clearance,
        ).reshape(transport_rows.numel(), candidate_count, 2)

        return ~(torch.any(held_stack_intersections, dim=2) | torch.any(hand_stack_intersections, dim=2))

    def _table_approach_assignments(
        self,
        row_ids: torch.Tensor,
        layouts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a balanced movable role and the clearest valid approach yaw.

        Large-cube layouts can be cube-cube valid while placing an open
        Allegro hand through a non-target cube. Score every movable-role/yaw
        candidate by the minimum distance from the palm-to-pair-center segment
        to either non-target cube center. A deterministic global role quota
        keeps the two order-invariant plans exactly balanced while allowing
        each layout to use whichever screened yaw leaves the clearest corridor.
        """
        orientation_choices = row_ids.new_tensor(self._orientation_ids_for_recipe(StackResetRecipe.TABLE))
        if tuple(orientation_choices.tolist()) != (2, 3, 6):
            raise RuntimeError("Full-hand TABLE corridor selection expects the screened yaw bins (2, 3, 6).")
        if layouts.shape != (row_ids.numel(), 3, 2):
            raise RuntimeError("Full-hand TABLE layouts must have shape (row_count, 3, 2).")
        if row_ids.numel() % 2 != 0:
            raise RuntimeError("Full-hand TABLE role balancing requires an even row count.")

        row_count = row_ids.numel()
        candidate_role_ids = row_ids.new_tensor((1, 2))
        candidate_orientation_ids = orientation_choices.view(1, 1, -1).expand(row_count, 2, -1)
        candidate_rotations, _ = self._target_wrist_rotations(
            candidate_orientation_ids.reshape(-1),
            torch.zeros(
                row_count * candidate_role_ids.numel() * orientation_choices.numel(),
                dtype=self._arm_anchors.dtype,
                device=self.device,
            ),
        )
        tool_offset = self._arm_anchors.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS[0])
        world_tool_offsets = torch.matmul(candidate_rotations, tool_offset.view(1, 3, 1)).squeeze(2)
        approach_axes = world_tool_offsets / torch.linalg.vector_norm(
            world_tool_offsets,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)

        candidate_xy = layouts[:, 1:3].unsqueeze(2).expand(-1, -1, orientation_choices.numel(), -1)
        contact_positions = torch.cat(
            (
                candidate_xy,
                layouts.new_full(
                    (row_count, candidate_role_ids.numel(), orientation_choices.numel(), 1),
                    self._pick_contact_height(),
                ),
            ),
            dim=3,
        ).reshape(-1, 3)
        pair_centers = contact_positions - self._OBJECT_AXIS_APPROACH_DISTANCE * approach_axes
        palm_centers = pair_centers - world_tool_offsets

        pair_centers = pair_centers.reshape(row_count, 2, orientation_choices.numel(), 3)
        palm_centers = palm_centers.reshape_as(pair_centers)
        cube_centers = torch.cat(
            (
                layouts,
                layouts.new_full((row_count, 3, 1), self._TABLE_HEIGHT),
            ),
            dim=2,
        )
        candidate_to_cube = cube_centers[:, None, None] - pair_centers.unsqueeze(3)
        corridor = palm_centers - pair_centers
        projection = torch.sum(candidate_to_cube * corridor.unsqueeze(3), dim=4)
        projection /= torch.sum(torch.square(corridor), dim=3, keepdim=True).clamp_min(1.0e-12)
        projection.clamp_(0.0, 1.0)
        closest_points = pair_centers.unsqueeze(3) + projection.unsqueeze(4) * corridor.unsqueeze(3)
        corridor_distances = torch.linalg.vector_norm(
            cube_centers[:, None, None] - closest_points,
            dim=4,
        )
        cube_role_ids = torch.arange(3, device=self.device).view(1, 1, 1, 3)
        selected_role_ids = candidate_role_ids.view(1, 2, 1, 1)
        corridor_distances.masked_fill_(cube_role_ids == selected_role_ids, torch.inf)
        candidate_clearances = corridor_distances.amin(dim=3)

        best_clearances_by_role, best_orientation_ranks_by_role = candidate_clearances.max(dim=2)
        role_advantage = best_clearances_by_role[:, 0] - best_clearances_by_role[:, 1]
        ranked_rows = torch.argsort(role_advantage, descending=True, stable=True)
        selected_role_ranks = torch.ones(row_count, dtype=torch.long, device=self.device)
        selected_role_ranks[ranked_rows[: row_count // 2]] = 0
        selected_orientation_ranks = best_orientation_ranks_by_role.gather(
            1,
            selected_role_ranks.unsqueeze(1),
        ).squeeze(1)

        self._table_approach_corridor_center_distances = best_clearances_by_role.gather(
            1,
            selected_role_ranks.unsqueeze(1),
        ).squeeze(1)
        return candidate_role_ids[selected_role_ranks], orientation_choices[selected_orientation_ranks]

    def _pick_phase(
        self,
        source_positions: torch.Tensor,
        progress: torch.Tensor,
        grasp_pair_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Author the contact-to-retained-lift bridge without roll or pitch."""
        target_positions, closure, held, maximum_tilt = self._base_pick_phase(
            source_positions,
            progress,
            grasp_pair_ids,
        )

        # The contact commands and pair-center transforms were screened under
        # live preload. Reuse that rigid grasp while lifting, rather than
        # teleporting the cube from table support directly into the old
        # 20-mm-high held state. The first half of the held interval densely
        # covers the missing 0--25 mm acquisition transition; the remainder
        # continues to the unchanged PICK endpoint.
        bridge_start_progress = progress.new_tensor(self._PICK_HELD_PROGRESS_BY_PAIR)[grasp_pair_ids]
        bridge_end_progress = progress.new_full(progress.shape, self._PICK_LIFT_BRIDGE_END_PROGRESS)
        if bool(torch.any(bridge_end_progress <= bridge_start_progress)):
            raise RuntimeError("PICK lift-bridge end progress must follow held-grasp acquisition.")
        contact_heights = progress.new_full(progress.shape, self._pick_contact_height())
        bridge_end_heights = contact_heights + self._PICK_LIFT_BRIDGE_HEIGHT
        endpoint_heights = self._pick_endpoint_heights(grasp_pair_ids, progress)
        if bool(torch.any(bridge_end_heights > endpoint_heights)):
            raise RuntimeError("PICK lift bridge must not exceed the PICK endpoint height.")
        bridge_fraction = torch.clamp(
            (progress - bridge_start_progress) / (bridge_end_progress - bridge_start_progress),
            min=0.0,
            max=1.0,
        )
        carry_fraction = torch.clamp(
            (progress - bridge_end_progress) / (1.0 - bridge_end_progress),
            min=0.0,
            max=1.0,
        )
        held_heights = torch.where(
            (progress <= bridge_end_progress).unsqueeze(1),
            torch.lerp(contact_heights, bridge_end_heights, bridge_fraction).unsqueeze(1),
            torch.lerp(bridge_end_heights, endpoint_heights, carry_fraction).unsqueeze(1),
        ).squeeze(1)
        target_positions[:, 2] = torch.where(held, held_heights, target_positions[:, 2])
        return target_positions, closure, held, torch.zeros_like(maximum_tilt)

    def _adjust_target_positions_for_rotation(
        self,
        target_positions: torch.Tensor,
        target_rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Replace vertical descents with the dynamically validated approach."""
        adjusted_positions = target_positions.clone()
        tool_offsets = target_positions.new_tensor(self._GRASP_PAIR_TOOL_OFFSETS)[self._grasp_pair_ids]
        approach_axes = torch.matmul(target_rotations, tool_offsets.unsqueeze(2)).squeeze(2)
        approach_axes /= torch.linalg.vector_norm(approach_axes, dim=1, keepdim=True).clamp_min(1.0e-6)

        first_pick = self._recipe_ids == int(StackResetRecipe.FIRST_PICK)
        second_pick = self._recipe_ids == int(StackResetRecipe.SECOND_PICK)
        supported_pick = (first_pick | second_pick) & (self._held_roles < 0)
        if bool(torch.any(supported_pick)):
            supported_rows = torch.nonzero(supported_pick, as_tuple=False).flatten()
            source_roles = torch.where(first_pick[supported_rows], 1, 2)
            contact_positions = self._role_positions[supported_rows, source_roles].clone()
            contact_positions[:, 2] = self._pick_contact_height()
            approach_fraction = torch.clamp(1.0 - 2.0 * self._progress[supported_rows], min=0.0, max=1.0)
            approach_distances = target_positions.new_tensor(self._PICK_APPROACH_DISTANCES_BY_YAW)[
                self._orientation_ids[supported_rows]
            ]
            adjusted_positions[supported_rows] = contact_positions - (
                approach_distances.unsqueeze(1) * approach_fraction.unsqueeze(1) * approach_axes[supported_rows]
            )

        pair_ready = self._recipe_ids == int(StackResetRecipe.PAIR_READY)
        if bool(torch.any(pair_ready)):
            pair_ready_rows = torch.nonzero(pair_ready, as_tuple=False).flatten()
            source_positions = self._role_positions[pair_ready_rows, 2].clone()
            source_positions[:, 2] = self._pick_contact_height()
            desired_endpoints = source_positions - (
                self._OBJECT_AXIS_APPROACH_DISTANCE * approach_axes[pair_ready_rows]
            )
            authored_endpoints = source_positions.clone()
            authored_endpoints[:, 2] = self._pair_ready_source_height()
            adjusted_positions[pair_ready_rows] += self._progress[pair_ready_rows].unsqueeze(1) * (
                desired_endpoints - authored_endpoints
            )

        table = self._recipe_ids == int(StackResetRecipe.TABLE)
        if bool(torch.any(table)):
            table_rows = torch.nonzero(table, as_tuple=False).flatten()
            table_local_rows = self._layout_ids[table_rows] - self._SEMANTIC_LAYOUT_COUNT
            source_roles = self._resolved_table_approach_role_ids(table_local_rows)
            contact_positions = self._role_positions[table_rows, source_roles].clone()
            contact_positions[:, 2] = self._pick_contact_height()
            adjusted_positions[table_rows] = contact_positions - (
                self._OBJECT_AXIS_APPROACH_DISTANCE * approach_axes[table_rows]
            )

        return adjusted_positions

    def _validate_table(self) -> None:
        """Validate the final achieved hand/cube stack-clearance envelope."""
        self._validate_kuka_table()
        row_ids = torch.arange(self.row_count, device=self.device)
        actual_tool_positions, actual_palm_rotations = self._grasp_pair_pose(
            self._arm_positions,
            self._grasp_pair_ids,
        )
        second_transport = self._recipe_ids == int(StackResetRecipe.SECOND_TRANSPORT)
        transport_rows = row_ids[second_transport]
        collision_free = self._second_transport_stack_clearance(
            transport_rows,
            actual_palm_rotations[second_transport].unsqueeze(1),
            actual_tool_positions[second_transport].unsqueeze(1),
            hand_clearance=self._SECOND_TRANSPORT_HAND_STACK_CLEARANCE,
            cube_clearance=self._SECOND_TRANSPORT_CUBE_STACK_CLEARANCE,
        ).squeeze(1)
        invalid_rows = transport_rows[~collision_free]
        if invalid_rows.numel() > 0:
            raise RuntimeError(
                "Full-hand SECOND_TRANSPORT reset rows violate the final stack-clearance envelope "
                f"(count={invalid_rows.numel()}, rows={invalid_rows[:16].tolist()})."
            )

        final_release = self._recipe_ids == int(StackResetRecipe.FINAL_RELEASE)
        release_rows = row_ids[final_release]
        release_clear = self._final_release_stack_clearance(
            release_rows,
            actual_palm_rotations[final_release].unsqueeze(1),
            actual_tool_positions[final_release].unsqueeze(1),
        ).squeeze(1)
        invalid_rows = release_rows[~release_clear]
        if invalid_rows.numel() > 0:
            raise RuntimeError(
                "Full-hand FINAL_RELEASE reset rows violate the non-tip top-cube or lower-stack "
                f"clearance envelope (count={invalid_rows.numel()}, rows={invalid_rows[:16].tolist()})."
            )
