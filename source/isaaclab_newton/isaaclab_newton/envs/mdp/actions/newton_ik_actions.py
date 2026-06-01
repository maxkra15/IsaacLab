# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton import JointType
from newton import Model as NewtonModel
from newton.selection import ArticulationView

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_newton.ik.newton_ik_manager import NewtonIKManager, NewtonIKPoseObjective
from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from .newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg


logger = logging.getLogger(__name__)


class NewtonInverseKinematicsAction(ActionTerm):
    """Newton inverse-kinematics action term."""

    cfg: NewtonInverseKinematicsActionCfg
    _asset: BaseArticulation

    def __init__(self, cfg: NewtonInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not isinstance(self._asset, BaseArticulation):
            raise TypeError(
                f"NewtonInverseKinematicsAction expects a BaseArticulation asset, got {type(self._asset).__name__}."
            )
        if not self._asset.is_fixed_base:
            raise ValueError("NewtonInverseKinematicsAction currently supports fixed-base articulations only.")

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        if len(self._joint_ids) == 0:
            raise ValueError(f"No joints matched Newton IK action joint_names={self.cfg.joint_names}.")
        self._joint_ids_warp = wp.array(self._joint_ids, dtype=wp.int32, device=self.device)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for Newton IK body_name={self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat_b[:, 3] = 1.0

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        self._clip = None
        if self.cfg.clip is not None:
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            action_names = self._action_coordinate_names()
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, action_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
            link_offset_pos = tuple(self.cfg.body_offset.pos)
            link_offset_rot = tuple(self.cfg.body_offset.rot)
        else:
            self._offset_pos, self._offset_rot = None, None
            link_offset_pos = (0.0, 0.0, 0.0)
            link_offset_rot = (0.0, 0.0, 0.0, 1.0)

        prototype_model = self._resolve_ik_model()
        prototype_view = self._resolve_prototype_view(prototype_model)
        self._prototype_joint_coord_ids = self._resolve_prototype_joint_coord_ids(
            prototype_view, self._asset.joint_names
        )
        self._prototype_controlled_coord_ids = self._resolve_prototype_joint_coord_ids(
            prototype_view, self._joint_names
        )
        self._prototype_link_index = self._resolve_prototype_link_index(prototype_view)
        self._fixed_objective_names = [f"fixed_{index}" for index, _ in enumerate(self.cfg.fixed_body_names)]
        self._fixed_body_indices, fixed_link_indices = self._resolve_fixed_bodies(prototype_view)
        self._fixed_target_pos_b = torch.zeros(
            (self.num_envs, len(self._fixed_body_indices), 3), device=self.device
        )
        self._fixed_target_quat_b = torch.zeros(
            (self.num_envs, len(self._fixed_body_indices), 4), device=self.device
        )
        if self._fixed_body_indices:
            self._fixed_target_quat_b[..., 3] = 1.0
        self._prototype_joint_seed = wp.to_torch(prototype_model.joint_q).to(device=self.device, dtype=torch.float32)
        self._prototype_joint_seed = self._prototype_joint_seed.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        self._ik_target_name = "target"

        pose_objectives = [
            NewtonIKPoseObjective(
                name=self._ik_target_name,
                link_index=self._prototype_link_index,
                link_offset_pos=link_offset_pos,
                link_offset_rot=link_offset_rot,
            )
        ]
        fixed_weights = self._fixed_body_weights()
        pose_objectives.extend(
            NewtonIKPoseObjective(
                name=name,
                link_index=link_index,
                position_weight=weight,
                rotation_weight=weight,
            )
            for name, link_index, weight in zip(self._fixed_objective_names, fixed_link_indices, fixed_weights)
        )

        self._ik_manager = NewtonIKManager(
            self.cfg.controller,
            model=prototype_model,
            num_envs=self.num_envs,
            device=self.device,
            pose_objectives=pose_objectives,
        )
        self._refresh_persistent_seed()
        self._capture_fixed_body_targets()
        self._set_fixed_body_targets()

        logger.info(
            "Resolved Newton IK action joints %s [%s] and body %s [%s].",
            self._joint_names,
            self._joint_ids,
            self._body_name,
            self._body_idx,
        )

    def _resolve_ik_model(self) -> NewtonModel:
        model_source = self.cfg.ik_model_source
        if model_source == "prototype":
            prototype_info = NewtonManager.get_prototype_model(self._asset.cfg.prim_path)
            prototype_model = prototype_info.model
            if prototype_model is None:
                raise RuntimeError(f"Newton prototype model for '{self._asset.cfg.prim_path}' was not finalized.")
            return prototype_model
        if model_source == "asset_usd":
            return self._build_asset_usd_ik_model()
        raise ValueError(
            f"Unsupported Newton IK model source '{model_source}'. Supported values are 'prototype' and 'asset_usd'."
        )

    def _build_asset_usd_ik_model(self) -> NewtonModel:
        spawn_cfg = getattr(self._asset.cfg, "spawn", None)
        usd_path = getattr(spawn_cfg, "usd_path", None)
        if usd_path is None:
            raise ValueError(
                "Newton IK ik_model_source='asset_usd' requires the articulation asset to be spawned from a USD file."
            )

        builder = NewtonManager.create_builder()
        root_pos = self._asset.data.root_pos_w.torch[0].detach().cpu().numpy()
        root_quat = self._asset.data.root_quat_w.torch[0].detach().cpu().numpy()
        root_xform = wp.transform(
            wp.vec3(float(root_pos[0]), float(root_pos[1]), float(root_pos[2])),
            wp.quat(float(root_quat[0]), float(root_quat[1]), float(root_quat[2]), float(root_quat[3])),
        )
        builder.add_usd(
            str(usd_path),
            xform=root_xform,
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=False,
            hide_collision_shapes=True,
            only_load_enabled_joints=True,
            only_load_enabled_rigid_bodies=False,
            parse_mujoco_options=True,
            root_path="/",
            override_root_xform=True,
        )

        for body_id, label in enumerate(builder.body_label):
            if str(label).endswith("gripper_dummy") and builder.body_mass[body_id] == 0.0:
                builder.body_mass[body_id] = 1.0e-6
                builder.body_inv_mass[body_id] = 1.0e6
                builder.body_inertia[body_id] = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-10)
                builder.body_inv_inertia[body_id] = wp.inverse(builder.body_inertia[body_id])

        return builder.finalize(device=wp.get_device(str(self.device)))

    @property
    def action_dim(self) -> int:
        if self.cfg.controller.command_type == "position":
            return 3
        if self.cfg.controller.command_type == "pose" and self.cfg.controller.use_relative_mode:
            return 6
        if self.cfg.controller.command_type == "pose":
            return 7
        raise ValueError(f"Unsupported Newton IK command type: {self.cfg.controller.command_type}")

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "NewtonInverseKinematicsAction"
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.clip = self.cfg.clip
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        self._IO_descriptor.extras["body_offset"] = (
            None if self.cfg.body_offset is None else self.cfg.body_offset.__dict__
        )
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        if self.cfg.controller.command_type == "position":
            if self.cfg.controller.use_relative_mode:
                self._target_pos_b[:] = ee_pos_b + self._processed_actions
            else:
                self._target_pos_b[:] = self._processed_actions
            self._target_quat_b[:] = ee_quat_b
        elif self.cfg.controller.use_relative_mode:
            self._target_pos_b[:], self._target_quat_b[:] = math_utils.apply_delta_pose(
                ee_pos_b, ee_quat_b, self._processed_actions
            )
        else:
            self._target_pos_b[:] = self._processed_actions[:, 0:3]
            self._target_quat_b[:] = self._processed_actions[:, 3:7]

    def apply_actions(self) -> None:
        root_pos_proto = self._asset.data.root_pos_w.torch[0:1].repeat(self.num_envs, 1)
        root_quat_proto = self._asset.data.root_quat_w.torch[0:1].repeat(self.num_envs, 1)
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_proto, root_quat_proto, self._target_pos_b, self._target_quat_b
        )
        self._ik_manager.set_target_pose(self._ik_target_name, target_pos_w, target_quat_w)
        self._set_fixed_body_targets()

        if self.cfg.controller.use_persistent_seed:
            joint_pos_des_all = self._ik_manager.solve()
        else:
            joint_seed = self._current_prototype_joint_seed()
            joint_pos_des_all = self._ik_manager.solve(joint_seed)
        joint_pos_des = joint_pos_des_all[:, self._prototype_controlled_coord_ids].contiguous()
        self._asset.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids_warp)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        if self.cfg.controller.use_persistent_seed:
            self._refresh_persistent_seed(env_ids)
        self._capture_fixed_body_targets(env_ids)
        self._set_fixed_body_targets()

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w.torch[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w.torch[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        if self.cfg.body_offset is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        return ee_pos_b, ee_quat_b

    def _resolve_prototype_view(self, model) -> ArticulationView:
        requested_path = NewtonManager._to_first_env_path(self._asset.cfg.prim_path)
        basename = requested_path.rsplit("/", 1)[-1]
        patterns = [requested_path, requested_path.replace(".*", "*"), basename, f"*{basename}"]
        articulation_labels = [str(label) for label in getattr(model, "articulation_label", [])]
        if len(articulation_labels) == 1:
            patterns.insert(0, articulation_labels[0])
        articulation_root_path = getattr(self._asset.cfg, "articulation_root_prim_path", None)
        if articulation_root_path:
            root_suffix = str(articulation_root_path)
            if not root_suffix.startswith("/"):
                root_suffix = f"/{root_suffix}"
            patterns.insert(0, requested_path + root_suffix)
        for label in articulation_labels:
            if label == requested_path or label.startswith(requested_path + "/"):
                patterns.insert(0, label)
        last_error: Exception | None = None
        seen_patterns = set()
        for pattern in patterns:
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)
            try:
                view = ArticulationView(
                    model,
                    pattern,
                    verbose=False,
                    exclude_joint_types=[JointType.FREE, JointType.FIXED],
                )
            except (KeyError, ValueError) as exc:
                last_error = exc
                continue
            if view.world_count != 1 or view.count_per_world != 1:
                raise ValueError(
                    f"Newton IK expected one prototype articulation for '{self._asset.cfg.prim_path}', "
                    f"got world_count={view.world_count}, count_per_world={view.count_per_world}."
                )
            if not view.is_fixed_base:
                raise ValueError("Newton IK currently supports fixed-base prototype articulations only.")
            return view
        raise KeyError(
            f"Failed to resolve Newton prototype articulation for '{self._asset.cfg.prim_path}'."
        ) from last_error

    def _resolve_prototype_joint_coord_ids(
        self, prototype_view: ArticulationView, joint_names: Sequence[str]
    ) -> torch.Tensor:
        layout = prototype_view.frequency_layouts[NewtonModel.AttributeFrequency.JOINT_COORD]
        selected_indices = self._layout_indices(layout)
        coord_indices_by_name = {
            name: layout.offset + selected_indices[index] for index, name in enumerate(prototype_view.joint_coord_names)
        }
        try:
            coord_ids = [coord_indices_by_name[name] for name in joint_names]
        except KeyError as exc:
            raise KeyError(
                f"Joint '{exc.args[0]}' was resolved in Isaac Lab but not in the Newton prototype model."
            ) from exc
        return torch.tensor(coord_ids, device=self.device, dtype=torch.long)

    def _resolve_prototype_link_index(self, prototype_view: ArticulationView) -> int:
        layout = prototype_view.frequency_layouts[NewtonModel.AttributeFrequency.BODY]
        selected_indices = self._layout_indices(layout)
        try:
            local_link_index = prototype_view.link_names.index(self._body_name)
        except ValueError as exc:
            raise ValueError(
                f"Body '{self._body_name}' was resolved in Isaac Lab but not in the Newton prototype model."
            ) from exc
        return layout.offset + selected_indices[local_link_index]

    def _resolve_fixed_bodies(self, prototype_view: ArticulationView) -> tuple[list[int], list[int]]:
        asset_body_ids: list[int] = []
        prototype_link_indices: list[int] = []
        if not self.cfg.fixed_body_names:
            return asset_body_ids, prototype_link_indices

        layout = prototype_view.frequency_layouts[NewtonModel.AttributeFrequency.BODY]
        selected_indices = self._layout_indices(layout)
        for fixed_body_name in self.cfg.fixed_body_names:
            body_ids, body_names = self._asset.find_bodies(fixed_body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"Expected one match for fixed Newton IK body '{fixed_body_name}'. "
                    f"Found {len(body_ids)}: {body_names}."
                )
            asset_body_ids.append(body_ids[0])
            try:
                local_link_index = prototype_view.link_names.index(body_names[0])
            except ValueError as exc:
                raise ValueError(
                    f"Fixed body '{body_names[0]}' was resolved in Isaac Lab but not in the Newton IK model."
                ) from exc
            prototype_link_indices.append(layout.offset + selected_indices[local_link_index])
        return asset_body_ids, prototype_link_indices

    def _fixed_body_weights(self) -> list[float | None]:
        body_count = len(self.cfg.fixed_body_names)
        if body_count == 0:
            return []
        if self.cfg.fixed_body_weights is None:
            return [None] * body_count
        if len(self.cfg.fixed_body_weights) != body_count:
            raise ValueError(
                "Newton IK fixed_body_weights must be omitted or have the same length as fixed_body_names "
                f"({len(self.cfg.fixed_body_weights)} != {body_count})."
            )
        return [float(weight) for weight in self.cfg.fixed_body_weights]

    def _current_prototype_joint_seed(self) -> torch.Tensor:
        joint_seed = self._prototype_joint_seed.clone()
        joint_seed[:, self._prototype_joint_coord_ids] = self._asset.data.joint_pos.torch
        return joint_seed

    def _refresh_persistent_seed(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        joint_seed = self._current_prototype_joint_seed()
        if env_ids is None:
            self._ik_manager.set_joint_seed(joint_seed)
        else:
            ids = self._env_ids_to_tensor(env_ids)
            self._ik_manager.reset(env_ids=env_ids, joint_pos=joint_seed[ids])

    def _capture_fixed_body_targets(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if not self._fixed_body_indices:
            return
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        body_pos_w = self._asset.data.body_pos_w.torch[:, self._fixed_body_indices]
        body_quat_w = self._asset.data.body_quat_w.torch[:, self._fixed_body_indices]
        if env_ids is None:
            for fixed_index in range(len(self._fixed_body_indices)):
                self._fixed_target_pos_b[:, fixed_index], self._fixed_target_quat_b[:, fixed_index] = (
                    math_utils.subtract_frame_transforms(
                        root_pos_w, root_quat_w, body_pos_w[:, fixed_index], body_quat_w[:, fixed_index]
                    )
                )
            return
        ids = self._env_ids_to_tensor(env_ids)
        for fixed_index in range(len(self._fixed_body_indices)):
            self._fixed_target_pos_b[ids, fixed_index], self._fixed_target_quat_b[ids, fixed_index] = (
                math_utils.subtract_frame_transforms(
                    root_pos_w[ids],
                    root_quat_w[ids],
                    body_pos_w[ids, fixed_index],
                    body_quat_w[ids, fixed_index],
                )
            )

    def _set_fixed_body_targets(self) -> None:
        if not self._fixed_objective_names:
            return
        root_pos_proto = self._asset.data.root_pos_w.torch[0:1].repeat(self.num_envs, 1)
        root_quat_proto = self._asset.data.root_quat_w.torch[0:1].repeat(self.num_envs, 1)
        for fixed_index, objective_name in enumerate(self._fixed_objective_names):
            target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
                root_pos_proto,
                root_quat_proto,
                self._fixed_target_pos_b[:, fixed_index],
                self._fixed_target_quat_b[:, fixed_index],
            )
            self._ik_manager.set_target_pose(
                objective_name,
                target_pos_w,
                target_quat_w,
            )

    def _env_ids_to_tensor(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            return torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        if isinstance(env_ids, Integral):
            return torch.as_tensor([env_ids], device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

    @staticmethod
    def _layout_indices(layout) -> list[int]:
        if layout.slice is not None:
            return list(range(layout.slice.start, layout.slice.stop))
        return [int(index) for index in layout.indices.numpy().tolist()]

    def _action_coordinate_names(self) -> list[str]:
        if self.cfg.controller.command_type == "position":
            return ["x", "y", "z"]
        if self.cfg.controller.command_type == "pose" and self.cfg.controller.use_relative_mode:
            return ["x", "y", "z", "roll", "pitch", "yaw"]
        return ["x", "y", "z", "qx", "qy", "qz", "qw"]
