# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-policy variant of the reset-oriented Franka stack task."""

from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.contrib.stack import mdp, stack_env_cfg
from isaaclab_tasks.contrib.stack.stack_env_cfg import ObjectTableSceneCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .stack_rl_env_cfg import EventCfg, FrankaCubeStackRLEnvCfg


@configclass
class NewtonDefaultStackRendererCfg(MultiBackendRendererCfg):
    """Multi-backend camera renderer with a kitless Newton default."""

    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg()


@configclass
class FrankaStackCameraSceneCfg(ObjectTableSceneCfg):
    """Franka stack scene with one fixed, real-world-deployable RGB camera."""

    # The oblique view resolves both table-plane axes and keeps the complete
    # 0.40-0.56 m by -0.18-0.18 m reset workspace in frame. The quaternion is
    # the OpenGL look-at rotation from ``pos`` to (0.48, 0.0, 0.08).
    base_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        # Four-centimeter cubes project to only 3--6 pixels at 64 px from this
        # deployment camera. At 128 px they retain enough edge and finger
        # detail for the spatial-softmax encoder's final 12 x 12 feature map.
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.90, -0.55, 0.48),
            rot=(0.4734479, 0.1600998, 0.2774602, 0.8205066),
            convention="opengl",
        ),
        renderer_cfg=NewtonDefaultStackRendererCfg(),
    )


@configclass
class CameraEventCfg(EventCfg):
    """Physical reset table plus fixed-per-environment camera calibration error."""

    camera_calibration = EventTerm(
        func=mdp.randomize_camera_calibration,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("base_camera"),
            "eye": (0.90, -0.55, 0.48),
            "lookat": (0.48, 0.0, 0.08),
            "eye_position_noise": (0.020, 0.020, 0.015),
            "lookat_position_noise": (0.015, 0.015, 0.010),
        },
    )


@configclass
class CameraObservationsCfg:
    """Deployable actor observations and an asymmetric training critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Signals available from the real Franka controller."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        gripper_pos = ObsTerm(
            func=mdp.gripper_pos,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # Both signals are computable from the real robot's joint encoders and
        # kinematic model. Supplying them explicitly avoids asking the camera
        # encoder to relearn Franka forward/differential kinematics.
        eef_velocity = ObsTerm(func=mdp.franka_ee_velocity)
        eef_axes = ObsTerm(func=mdp.franka_ee_axes)
        eef_position = ObsTerm(func=mdp.franka_ee_position)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Simulator state used only to reduce critic variance while training."""

        object = ObsTerm(func=mdp.role_conditioned_stack_obs)
        eef_velocity = ObsTerm(func=mdp.franka_ee_velocity)
        eef_axes = ObsTerm(func=mdp.franka_ee_axes)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class BaseImageCfg(ObsGroup):
        """Two-frame, channel-first RGB input for the visual encoder."""

        rgb = ObsTerm(
            func=mdp.TemporalNormalizedRgbImage,
            params={"sensor_cfg": SceneEntityCfg("base_camera"), "history_length": 2},
            clip=(-0.5, 0.5),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    base_image: BaseImageCfg = BaseImageCfg()


@configclass
class FrankaStackTeacherObservationsCfg(stack_env_cfg.ObservationsCfg.PolicyCfg):
    """Exact 100-value state interface used by the proven Franka teacher.

    The inherited term order is intentionally preserved because the teacher's
    empirical normalizer and first MLP layer are tied to this layout. This
    group exists only in the distillation task and is never exposed to the
    camera student.
    """

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    object = ObsTerm(func=mdp.role_conditioned_stack_obs)
    cube_positions = None
    cube_orientations = None
    eef_pos = None
    eef_quat = None
    stack_state = None
    eef_velocity = ObsTerm(func=mdp.franka_ee_velocity)
    eef_axes = ObsTerm(func=mdp.franka_ee_axes)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DistillationContextCfg(ObsGroup):
    """Privileged labels for balanced supervision, never consumed by the actor."""

    recipe = ObsTerm(func=mdp.stack_reset_recipe_one_hot, params={"recipe_count": 9})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CameraDistillationObservationsCfg(CameraObservationsCfg):
    """Student camera observations plus a teacher-only simulator-state group."""

    teacher: FrankaStackTeacherObservationsCfg = FrankaStackTeacherObservationsCfg()
    distillation_context: DistillationContextCfg = DistillationContextCfg()


@configclass
class FrankaCubeStackCameraRLEnvCfg(FrankaCubeStackRLEnvCfg):
    """Stack from RGB and proprioception with no actor-side object state."""

    scene: FrankaStackCameraSceneCfg = FrankaStackCameraSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The parent reinstalls its state-task manager configs. Swap back only
        # the camera-specific events and observation interface; physics,
        # actions, rewards, reset rows, curriculum, and success logic remain
        # byte-for-byte identical to the proven state task.
        self.events = CameraEventCfg()
        self.observations = CameraObservationsCfg()

        # The state teacher receives an abstract base/first/second role order.
        # A camera actor cannot observe the reset-time permutation that used to
        # assign colored cube assets to those roles. Tie the roles to the visible
        # blue/red/green assets so imitation has one identifiable target while
        # the physical success condition remains fully order-invariant.
        self.events.reset_from_state_buffer.params["fixed_role_permutation"] = 0

        # A reset changes the robot and all three cubes after the regular
        # render for that step. Refresh once so the first action of every
        # episode sees the new state instead of another environment's final
        # frame.
        self.num_rerenders_on_reset = 1


@configclass
class FrankaCubeStackCameraDistillationEnvCfg(FrankaCubeStackCameraRLEnvCfg):
    """Camera stack environment exposing state only to the frozen teacher."""

    observations: CameraDistillationObservationsCfg = CameraDistillationObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations = CameraDistillationObservationsCfg()
        # Establish the easiest possible supervised-transfer baseline first.
        # Calibration and proprioceptive noise belong after the visual adapter
        # reproduces the frozen controller, not in the initial sanity check.
        self.events.camera_calibration.params["eye_position_noise"] = (0.0, 0.0, 0.0)
        self.events.camera_calibration.params["lookat_position_noise"] = (0.0, 0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None
        self.observations.policy.gripper_pos.noise = None
        # Reserve four deterministic student rollouts for every reset recipe on
        # each rank. These slots never contribute to optimization and expose
        # exactly which phase blocks end-to-end TABLE success.
        self.events.reset_from_state_buffer.params["evaluation_recipe_ids"] = tuple(range(9))
        self.events.reset_from_state_buffer.params["evaluation_envs_per_recipe"] = 4
        # Keep the deterministic evaluation stream genuinely held out: its
        # outcomes must neither train the student nor steer reset sampling.
        self.curriculum.reset_sampling.params["evaluation_env_count"] = 9 * 4
