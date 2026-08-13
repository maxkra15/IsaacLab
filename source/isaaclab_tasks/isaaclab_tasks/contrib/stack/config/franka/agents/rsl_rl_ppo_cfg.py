# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Bernoulli, Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


class _StackDeterministicOutput(nn.Module):
    """Convert the gripper logit into the physical binary action during export."""

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        gripper = torch.where(
            output[..., -1:] >= 0.0,
            torch.ones_like(output[..., -1:]),
            -torch.ones_like(output[..., -1:]),
        )
        return torch.cat((output[..., :-1], gripper), dim=-1)


class StackGaussianDistribution(GaussianDistribution):
    """Gaussian arm exploration plus a true Bernoulli binary gripper.

    RSL-RL policies normally emit a Gaussian for every action. That is correct
    for the seven continuous normalized joint residuals but not for the last
    action, which is thresholded into open/close by the environment. A Gaussian log probability
    assigns different likelihoods to samples that become the exact same
    physical gripper command and makes reliable long transports depend on an
    arbitrary mean-to-standard-deviation ratio.

    The arm retains a bounded learnable Gaussian. The gripper MLP output is a
    Bernoulli logit sampled directly as ``-1`` (close) or ``+1`` (open), so PPO
    optimizes the probability of the action that physics actually receives.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.45,
        std_range: tuple[float, float] = (0.15, 0.65),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if output_dim < 2:
            raise ValueError("StackGaussianDistribution requires arm outputs followed by one gripper output.")
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing lower and upper bounds.")
        if not std_range[0] < init_std < std_range[1]:
            raise ValueError("init_std must lie strictly inside std_range.")
        if std_type != "scalar":
            raise ValueError("StackGaussianDistribution supports only scalar standard-deviation parameters.")
        super().__init__(output_dim, init_std=init_std, std_type=std_type, **kwargs)
        self.std_range = (float(std_range[0]), float(std_range[1]))
        self._arm_distribution: Normal | None = None
        self._gripper_distribution: Bernoulli | None = None
        # Store an unconstrained logit so gradients remain nonzero near both
        # bounds. A forward clamp leaves zero-gradient parameters stranded
        # below the minimum, which caused four arm joints to lose exploration.
        initial_fraction = (init_std - self.std_range[0]) / (self.std_range[1] - self.std_range[0])
        initial_logit = torch.logit(torch.tensor(initial_fraction, dtype=self.std_param.dtype))
        with torch.no_grad():
            self.std_param.fill_(initial_logit)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the continuous-arm and categorical-gripper distributions."""
        minimum_std, maximum_std = self.std_range
        arm_std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param[:-1])
        self._arm_distribution = Normal(mlp_output[..., :-1], arm_std)
        self._gripper_distribution = Bernoulli(logits=mlp_output[..., -1:])

    def sample(self) -> torch.Tensor:
        """Sample joint deltas and the exact binary gripper command."""
        gripper_open = self._gripper_distribution.sample()
        return torch.cat((self._arm_distribution.sample(), 2.0 * gripper_open - 1.0), dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return arm means and the most likely physical gripper command."""
        return _StackDeterministicOutput()(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return the export module implementing the same binary decision."""
        return _StackDeterministicOutput()

    @property
    def mean(self) -> torch.Tensor:
        """Return arm means and the expected signed gripper command."""
        gripper_mean = 2.0 * self._gripper_distribution.probs - 1.0
        return torch.cat((self._arm_distribution.mean, gripper_mean), dim=-1)

    @property
    def std(self) -> torch.Tensor:
        """Return arm standard deviations and signed-Bernoulli spread."""
        gripper_std = 2.0 * torch.sqrt(self._gripper_distribution.probs * (1.0 - self._gripper_distribution.probs))
        return torch.cat((self._arm_distribution.stddev, gripper_std), dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        """Return the joint Gaussian-plus-Bernoulli entropy."""
        return self._arm_distribution.entropy().sum(dim=-1) + self._gripper_distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return parameters required for the mixed-distribution KL."""
        return self._arm_distribution.mean, self._arm_distribution.stddev, self._gripper_distribution.logits

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact physical continuous/binary action."""
        arm_log_prob = self._arm_distribution.log_prob(outputs[..., :-1]).sum(dim=-1)
        gripper_open = (outputs[..., -1:] >= 0.0).to(outputs.dtype)
        return arm_log_prob + self._gripper_distribution.log_prob(gripper_open).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return ``KL(old || new)`` for both action families."""
        old_arm_mean, old_arm_std, old_gripper_logits = old_params
        new_arm_mean, new_arm_std, new_gripper_logits = new_params
        arm_kl = torch.distributions.kl_divergence(
            Normal(old_arm_mean, old_arm_std),
            Normal(new_arm_mean, new_arm_std),
        ).sum(dim=-1)
        # Bernoulli's generic probability-space KL loses precision once a
        # learned gripper logit saturates: sigmoid rounds to exactly zero or
        # one and the diagnostic becomes infinity despite finite logits. Use
        # the Bernoulli exponential-family identity directly in logit space:
        #   KL(old || new) = p_old * (eta_old - eta_new)
        #                    - softplus(eta_old) + softplus(eta_new).
        # This remains finite for all finite policy outputs and makes the same
        # KL safe for RSL-RL's optional adaptive learning-rate schedule.
        old_gripper_probability = torch.sigmoid(old_gripper_logits)
        gripper_kl = (
            old_gripper_probability * (old_gripper_logits - new_gripper_logits)
            - torch.nn.functional.softplus(old_gripper_logits)
            + torch.nn.functional.softplus(new_gripper_logits)
        ).sum(dim=-1)
        gripper_kl.clamp_min_(0.0)
        return arm_kl + gripper_kl


@configclass
class StackGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Bounded Gaussian arm exploration with a Bernoulli gripper."""

    class_name: str = "isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg:StackGaussianDistribution"
    std_range: tuple[float, float] = (0.15, 0.65)


@configclass
class StackSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    """Spatial-keypoint camera model introduced by the DexSuite camera task."""

    class_name: str = "isaaclab_tasks.core.lift.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


class StackPPO(PPO):
    """PPO with a read-only post-update KL diagnostic.

    RSL-RL ordinarily computes KL only while its adaptive learning-rate
    schedule is active. This task uses a fixed rate, but still records one
    representative ``KL(rollout policy || updated policy)`` value per
    iteration. The diagnostic is evaluated after all PPO epochs and never
    changes the optimizer.
    """

    kl_measurement_samples = 16_384

    def update(self) -> dict[str, float]:
        """Update the policy and report its post-update KL divergence."""
        if self.storage.distribution_params is None:
            raise RuntimeError("Cannot measure PPO KL before collecting a rollout.")

        observations = self.storage.observations.flatten(0, 1)
        total_samples = observations.shape[0]
        sample_count = min(self.kl_measurement_samples, total_samples)
        sample_indices = (
            torch.arange(sample_count, device=self.device, dtype=torch.long) * total_samples // sample_count
        )
        kl_observations = observations[sample_indices]
        old_distribution_params = tuple(
            parameter.flatten(0, 1)[sample_indices] for parameter in self.storage.distribution_params
        )

        loss_dict = super().update()

        with torch.inference_mode():
            self.actor(kl_observations, stochastic_output=True)
            new_distribution_params = self.actor.output_distribution_params
            kl_mean = self.actor.get_kl_divergence(old_distribution_params, new_distribution_params).mean()
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

        loss_dict["kl"] = kl_mean.item()
        return loss_dict


@configclass
class FrankaStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the order-invariant three-cube Franka stack task."""

    # Match DexSuite's 32-step update cadence. Sixteen minibatches retain
    # 16,384 samples per rank/minibatch with 8,192 production environments.
    num_steps_per_env = 32
    max_iterations = 7000
    save_interval = 25
    experiment_name = "franka_stack"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # The bounded sigmoid retains acquisition-scale exploration while the
        # action term clamps residual joint targets to a safe per-step change.
        distribution_cfg=StackGaussianDistributionCfg(init_std=0.45),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="isaaclab_tasks.contrib.stack.config.franka.agents.rsl_rl_ppo_cfg:StackPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="fixed",
        # At the 50 Hz policy rate, 0.99 discounts a five-second-away terminal
        # reward to eight percent. The full two-pick sequence needs a
        # continuous-time horizon appropriate for 10-20 second episodes.
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FrankaStackCameraPPORunnerCfg(FrankaStackPPORunnerCfg):
    """Asymmetric PPO configuration for RGB-based Franka stacking."""

    max_iterations = 15000
    save_interval = 50
    experiment_name = "franka_stack_camera"
    obs_groups = {
        "actor": ["policy", "base_image"],
        "critic": ["policy", "privileged"],
    }
    actor = StackSpatialSoftmaxCNNModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=StackGaussianDistributionCfg(init_std=0.45),
        cnn_cfg=StackSpatialSoftmaxCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
        ),
    )
    # DexSuite camera training converges only when encoder updates stay on a
    # small fixed scale. Eight minibatches preserve its 16,384 samples per
    # rank/minibatch with the camera task's 4,096 environments.
    algorithm = FrankaStackPPORunnerCfg().algorithm.replace(
        entropy_coef=0.005,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
    )
