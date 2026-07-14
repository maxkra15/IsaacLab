# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the waterhose task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_cable_to_default(ManagerTermBase):
    """Restore every selected cable segment to its build-time pose with zero velocity.

    Cable segment transforms are owned directly by VBD and are intentionally excluded
    from generic forward kinematics. Therefore, resetting only the inherited
    articulation root and joint state does not restore a deformed cable. This term
    writes the immutable model poses into the parent Newton state and invalidates the
    selected worlds. The coupled manager's teleport protocol then distributes the
    state, clears coupling transients, synchronizes proxies, and re-seeds VBD history
    before the next physics step.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Cache the cable's per-environment segment body IDs.

        Args:
            cfg: Event-term configuration containing ``asset_cfg``.
            env: Environment containing the cable asset.

        Raises:
            RuntimeError: If the cable registry is incomplete or inconsistent.
        """
        super().__init__(cfg, env)
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset = env.scene[self._asset_cfg.name]

        registry_entry = self._asset._registry_entry
        body_offsets = registry_entry.body_offsets
        segment_count = len(registry_entry.edges)
        if len(body_offsets) != env.num_envs:
            raise RuntimeError(
                f"Cable registry for '{self._asset_cfg.name}' contains {len(body_offsets)} worlds; "
                f"expected {env.num_envs}."
            )
        if segment_count == 0:
            raise RuntimeError(f"Cable registry for '{self._asset_cfg.name}' contains no rod segments.")
        segment_body_indices = [
            list(range(int(body_offset), int(body_offset) + segment_count)) for body_offset in body_offsets
        ]
        self._segment_body_ids = torch.tensor(segment_body_indices, dtype=torch.long, device=env.device)
        self._all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Restore cable segment state for the selected environments.

        Args:
            env: Environment containing the cable asset.
            env_ids: Environment indices to reset, or ``None``/``slice(None)`` for every environment.
            asset_cfg: Cable scene entity configured for this event term.

        Raises:
            RuntimeError: If Newton model or state buffers are unavailable.
        """
        del env, asset_cfg
        if env_ids is None:
            reset_env_ids = self._all_env_ids
        elif isinstance(env_ids, slice):
            reset_env_ids = self._all_env_ids[env_ids]
        else:
            reset_env_ids = torch.as_tensor(env_ids, device=self._segment_body_ids.device, dtype=torch.long)
        if reset_env_ids.numel() == 0:
            return

        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        if model is None or state is None or model.body_q is None or state.body_q is None or state.body_qd is None:
            raise RuntimeError("Newton model and body state must be initialized before resetting a cable.")

        body_ids = self._segment_body_ids[reset_env_ids].reshape(-1)
        default_body_q = wp.to_torch(model.body_q)
        state_body_q = wp.to_torch(state.body_q)
        state_body_qd = wp.to_torch(state.body_qd)
        state_body_q.index_copy_(0, body_ids, default_body_q.index_select(0, body_ids))
        state_body_qd[body_ids] = 0.0

        reset_env_ids_wp = wp.from_torch(reset_env_ids.to(dtype=torch.int32).contiguous(), dtype=wp.int32)
        NewtonManager.invalidate_fk(
            env_ids=reset_env_ids_wp,
            articulation_ids=self._asset._root_view.articulation_ids,
        )
