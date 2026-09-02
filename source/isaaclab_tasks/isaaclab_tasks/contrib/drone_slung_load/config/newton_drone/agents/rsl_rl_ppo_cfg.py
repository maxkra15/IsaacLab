# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.contrib.drone_slung_load import (
    DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME,
    DRONE_DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME,
    DRONE_SLUNG_LOAD_WANDB_PROJECT,
    ENHANCED_EXPERIMENT_NAME,
)
from isaaclab_tasks.contrib.drone_slung_load.system import (
    ENHANCED_RESIDUAL_BODY_RATE_LIMITS,
    nominal_drone_hover_action,
    nominal_hover_action,
)

_NOMINAL_HOVER_ACTION = nominal_hover_action()
_ENHANCED_NEWTON_THRUST_TRIM = 5.0e-4


@configclass
class BoundedBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """RSL-RL beta policy bounded on ``[-1, 1]`` and initialized at hover."""

    class_name: str = (
        "isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents.bounded_beta_distribution:"
        "HoverBiasedBetaDistribution"
    )
    action_range: tuple[float, float] = (-1.0, 1.0)
    initial_mean: tuple[float, float, float, float] = (_NOMINAL_HOVER_ACTION, 0.0, 0.0, 0.0)
    concentration: float = 1_000.0


@configclass
class DroneSlungLoadPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared bounded PPO choices for the FLARE slung-load tasks.

    Beta sampling is a bounded stochastic implementation choice; FLARE reports
    a final deterministic tanh projection but does not publish its exploration
    distribution or PPO discount/GAE parameters.
    """

    # 1.28 s spans nearly one small-angle 0.50 m pendulum period at 100 Hz.
    num_steps_per_env = 128
    max_iterations = 3000
    save_interval = 100
    init_at_random_ep_len = False
    logger = "wandb"
    wandb_project = DRONE_SLUNG_LOAD_WANDB_PROJECT
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    empirical_normalization = False
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=BoundedBetaDistributionCfg(),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-4,
        schedule="adaptive",
        # At 100 Hz, these correspond to a roughly 2.3 s reward half-life and
        # 0.53 s GAE half-life, long enough to assign cable-swing consequences.
        gamma=0.997,
        lam=0.99,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DroneSlungLoadWaypointPPORunnerCfg(DroneSlungLoadPPORunnerCfg):
    """Production PPO runner for the FLARE waypoint baseline."""

    experiment_name = "drone_slung_load_waypoint_flare"


@configclass
class EnhancedTanhGaussianDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Axis-calibrated bounded Gaussian centered on loaded hover.

    The collective, roll/pitch, and yaw scales were stress-tested for a full
    15-second randomized episode with the enhanced velocity-hold controller.
    They retain strict action bounds and substantially more exploration than
    v5 without reproducing its predecessor's open-loop workspace failures.
    """

    class_name: str = (
        "isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents."
        "bounded_tanh_gaussian_distribution:HoverBiasedTanhGaussianDistribution"
    )
    action_range: tuple[float, float] = (-1.0, 1.0)
    # Newton's finite AVBD solve needs a tiny empirical collective correction:
    # +5e-4 held the level, loaded 8x32-solve system within 0.53 mm over 30 s.
    # Keep this backend-specific trim out of the analytic mass-balance helper
    # and the paper-aligned Beta baseline.
    initial_mean: tuple[float, float, float, float] = (
        _NOMINAL_HOVER_ACTION + _ENHANCED_NEWTON_THRUST_TRIM,
        0.0,
        0.0,
        0.0,
    )
    init_std: tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.03)
    std_range: tuple[float, float] = (0.005, 0.5)
    learn_std: bool = True


@configclass
class DroneSlungLoadDurabilityPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Enhanced-only PPO durability parameters passed to the custom algorithm."""

    physical_body_rate_limits: tuple[float, float, float] = ENHANCED_RESIDUAL_BODY_RATE_LIMITS
    final_learning_rate: float = 3.0e-5
    learning_rate_decay_updates: int = 400
    learning_rate_decay_start_update: int = 1_600
    final_entropy_coef: float = 5.0e-4
    entropy_decay_updates: int = 400
    entropy_decay_start_update: int = 1_600
    kl_guard_threshold: float = 0.015
    kl_rejection_lr_factor: float = 0.5
    kl_acceptance_lr_recovery_factor: float = 1.01
    kl_evaluation_batch_size: int = 16_384


@configclass
class DroneSlungLoadWaypointEnhancedPPORunnerCfg(DroneSlungLoadPPORunnerCfg):
    """PPO runner for aggressive, all-heading slung-load waypoint throughput.

    The later FLARE release uses a 100-step rollout, a 3e-4 learning rate,
    ``gamma=0.99``, ``lambda=0.95``, and an entropy coefficient of 0.002. This
    task keeps its rollout and exploration pressure, while using time-consistent
    credit at 100 Hz and two update epochs so braking and path recovery remain
    learnable over the measured 1--2 s target-arrival interval.
    """

    experiment_name = ENHANCED_EXPERIMENT_NAME
    num_steps_per_env = 100
    max_iterations = 2000
    # Frequent checkpoints let the randomized safety/throughput evaluator retain
    # a fast policy even when later PPO updates deliberately explore harder.
    save_interval = 10
    # Route and swing resets are already independently randomized. Staggering
    # episode clocks would invoke the multi-stage route sampler on most control
    # steps; synchronized timeouts batch that work without reducing diversity.
    init_at_random_ep_len = False
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=EnhancedTanhGaussianDistributionCfg(),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = DroneSlungLoadDurabilityPpoAlgorithmCfg(
        class_name=(
            "isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents."
            "durability_ppo:DroneSlungLoadDurabilityPPO"
        ),
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=2,
        num_mini_batches=4,
        # Keep optimization and exploration pressure active while the staged
        # curriculum reaches full speed at update 1600. Decay only during the
        # final 400 updates, while transactionally rejecting excessive KL.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.997,
        lam=0.99,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DroneSlungLoadDirectCTBRDurabilityPpoAlgorithmCfg(DroneSlungLoadDurabilityPpoAlgorithmCfg):
    """Durability schedule and telemetry for direct full-envelope CTBR."""

    physical_body_rate_limits: tuple[float, float, float] = (15.0, 15.0, 5.0)
    # A 500-step rollout reaches the final hard-route hold after 640 completed
    # updates. Anchoring at 639 applies the first 1/40 decay after update 640
    # and reaches the floor on the last of 680 configured updates.
    learning_rate_decay_updates: int = 40
    learning_rate_decay_start_update: int = 639
    final_entropy_coef: float = 2.0e-4
    entropy_decay_updates: int = 40
    entropy_decay_start_update: int = 639


@configclass
class DirectCTBRTanhGaussianDistributionCfg(EnhancedTanhGaussianDistributionCfg):
    """Bounded direct-CTBR exploration with a conservative learned ceiling."""

    std_range: tuple[float, float] = (0.005, 0.15)


@configclass
class DroneSlungLoadWaypointDirectCTBRPPORunnerCfg(DroneSlungLoadWaypointEnhancedPPORunnerCfg):
    """Long-horizon PPO for policy-owned CTBR on the route-first objective.

    Each five-second rollout contains the observed three-to-five-second failure
    window. At 100 Hz, ``gamma=lambda=0.999`` gives a 6.93-second reward
    half-life and a 3.46-second GAE half-life. Six hundred eighty rollouts cover
    the 340,000-step smooth-route foundation and hard-route continuation while
    preserving two learning epochs and the previous per-minibatch sample count.
    """

    experiment_name = DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME
    num_steps_per_env = 500
    max_iterations = 680
    save_interval = 5
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=DirectCTBRTanhGaussianDistributionCfg(),
    )
    algorithm = DroneSlungLoadDirectCTBRDurabilityPpoAlgorithmCfg(
        class_name=(
            "isaaclab_tasks.contrib.drone_slung_load.config.newton_drone.agents."
            "durability_ppo:DroneSlungLoadDurabilityPPO"
        ),
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=20,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.999,
        lam=0.999,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DroneDirectCTBRTanhGaussianDistributionCfg(DirectCTBRTanhGaussianDistributionCfg):
    """Direct-CTBR exploration centered on unloaded rigid-drone hover."""

    initial_mean: tuple[float, float, float, float] = (nominal_drone_hover_action(), 0.0, 0.0, 0.0)


@configclass
class DroneWaypointDirectCTBRPPORunnerCfg(DroneSlungLoadWaypointDirectCTBRPPORunnerCfg):
    """Route-first Direct-CTBR PPO runner for the rigid-drone task."""

    experiment_name = DRONE_DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=DroneDirectCTBRTanhGaussianDistributionCfg(),
    )
