# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera observations and calibration randomization for stack policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import Camera


def normalized_rgb_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_camera"),
) -> torch.Tensor:
    """Return an RGB image in channel-first layout with stationary scaling.

    A fixed affine map preserves absolute brightness and makes the observation
    independent of the other environments in the batch. Per-frame or
    per-batch mean subtraction would make the same physical image change when
    unrelated environments reset, and is unsuitable for real-camera
    deployment.

    Args:
        env: The stack environment containing the camera sensor.
        sensor_cfg: Scene entity selecting the RGB camera.

    Returns:
        RGB images with shape ``(num_envs, 3, height, width)`` and values in
        ``[-0.5, 0.5]``.
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    # ObservationManager calls every term once while it is still being
    # constructed to discover output shapes. Triggering a camera render at
    # that point asks Newton to forward kinematics before manager startup and
    # reset have completed. Return only a shape probe here; every observation
    # after construction reads the real renderer output below.
    if not hasattr(env, "observation_manager"):
        return torch.zeros(
            (env.num_envs, 3, camera.cfg.height, camera.cfg.width),
            device=env.device,
        )
    image = camera.data.output["rgb"].float()
    image = torch.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return (image / 255.0 - 0.5).permute(0, 3, 1, 2).contiguous()


class TemporalNormalizedRgbImage(ManagerTermBase):
    """Return a short RGB history concatenated along the channel axis.

    A single image cannot distinguish a supported cube from one that is
    falling through the same pose. Keeping two frames makes that velocity
    observable while retaining the ordinary ``NCHW`` interface expected by
    RSL-RL's CNN models. Reset environments repeat their first new frame so a
    policy never receives pixels from the preceding episode.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.history_length = int(cfg.params.get("history_length", 2))
        if self.history_length < 2:
            raise ValueError("Temporal RGB history_length must be at least two.")
        self._frames: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate reset histories so the next frame is repeated."""
        if env_ids is None:
            self._initialized.zero_()
        else:
            resolved_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._initialized.device).reshape(-1)
            self._initialized[resolved_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_camera"),
        history_length: int = 2,
    ) -> torch.Tensor:
        """Append the current frame and return oldest-to-newest channels."""
        if history_length != self.history_length:
            raise ValueError("Temporal RGB history_length changed after term construction.")
        current = normalized_rgb_image(env, sensor_cfg=sensor_cfg)
        expected_shape = (env.num_envs, self.history_length, *current.shape[1:])
        if self._frames is None or self._frames.shape != expected_shape:
            self._frames = current.unsqueeze(1).expand(expected_shape).clone()
            self._initialized.fill_(hasattr(env, "observation_manager"))
        else:
            self._frames[:, :-1].copy_(self._frames[:, 1:].clone())
            self._frames[:, -1].copy_(current)
            uninitialized = ~self._initialized
            self._frames[uninitialized] = current[uninitialized].unsqueeze(1)
            self._initialized[uninitialized] = True
        return self._frames.flatten(1, 2).clone()


def randomize_camera_calibration(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    eye_position_noise: tuple[float, float, float],
    lookat_position_noise: tuple[float, float, float],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_camera"),
) -> None:
    """Apply one small, independent extrinsic perturbation per camera.

    This startup-only randomization models mounting and hand-eye calibration
    error without changing the camera during an episode. The real deployment
    therefore needs only an approximately matching calibrated view rather than
    the exact simulated transform.

    Args:
        env: The stack environment containing the camera sensor.
        env_ids: Environments whose camera poses are randomized, or ``None``
            for every environment.
        eye: Nominal camera position relative to each environment origin.
        lookat: Nominal look-at point relative to each environment origin.
        eye_position_noise: Independent uniform position-noise half-widths for
            the camera eye, in meters.
        lookat_position_noise: Independent uniform position-noise half-widths
            for the look-at point, in meters.
        sensor_cfg: Scene entity selecting the camera.
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    if env_ids is None:
        resolved_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        resolved_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if resolved_env_ids.numel() == 0:
        return

    origins = env.scene.env_origins[resolved_env_ids]
    eye_offset = origins.new_tensor(eye).expand(resolved_env_ids.numel(), -1)
    lookat_offset = origins.new_tensor(lookat).expand(resolved_env_ids.numel(), -1)
    eye_amplitude = origins.new_tensor(eye_position_noise)
    lookat_amplitude = origins.new_tensor(lookat_position_noise)
    eye_noise = (2.0 * torch.rand_like(eye_offset) - 1.0) * eye_amplitude
    lookat_noise = (2.0 * torch.rand_like(lookat_offset) - 1.0) * lookat_amplitude

    camera.set_world_poses_from_view(
        origins + eye_offset + eye_noise,
        origins + lookat_offset + lookat_noise,
        env_ids=resolved_env_ids,
    )
