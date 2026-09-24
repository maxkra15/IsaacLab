# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-arm textile atelier with MJWarp rigid bodies coupled to VBD cloth."""

from __future__ import annotations

from pathlib import Path

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG

from . import mdp

_KUKA_ARM_JOINTS = ["iiwa7_joint_(1|2|3|4|5|6|7)"]
_HUMANOID_ARM_JOINTS = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
_TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
_FRAME_USD = str(Path(__file__).with_name("atelier_frame.usda"))


def _kuka_cfg() -> ArticulationCfg:
    """Place the KUKA on a pedestal with the authored arm facing the cloth."""
    cfg = KUKA_ALLEGRO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Kuka",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.30, -0.05, 0.70),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos=dict(KUKA_ALLEGRO_CFG.init_state.joint_pos),
        ),
    )
    cfg.spawn.fix_root_link = True
    return cfg


def _humanoid_cfg() -> ArticulationCfg:
    """Fix GR1T2 at its pelvis; only its right arm is actuated by this task."""
    cfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Humanoid",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.34, 0.0, 0.60),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
    cfg.spawn.fix_root_link = True
    return cfg


_SUPPORT_SPAWN = sim_utils.CuboidCfg(
    size=(0.62, 0.035, 0.02),
    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassCfg(mass=1.0),
    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.28, 0.30), roughness=0.7),
)


@configclass
class TextileAtelierSceneCfg(InteractiveSceneCfg):
    """A cloth-draping table between an arm and a fixed-base humanoid."""

    kuka: ArticulationCfg = _kuka_cfg()
    humanoid: ArticulationCfg = _humanoid_cfg()

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.89)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.58, 0.40),
            edge_refinement=16,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.31, 0.20), roughness=0.82),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.002,
                tri_ke=5.0e2,
                tri_ka=5.0e2,
                tri_kd=1.0e-3,
                edge_ke=0.5,
                edge_kd=1.0e-3,
            ),
        ),
    )

    support_neg_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupportNegY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.14, 0.85)),
        spawn=_SUPPORT_SPAWN,
    )
    support_pos_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupportPosY",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.14, 0.85)),
        spawn=_SUPPORT_SPAWN,
    )
    tabletop: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.675)),
        spawn=sim_utils.CuboidCfg(
            size=(1.40, 0.84, 0.05),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visible=False,
        ),
    )
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.70), rot=(0.0, 0.0, 0.70710678, 0.70710678)),
        spawn=UsdFileCfg(
            usd_path=_TABLE_USD,
            make_uninstanceable=True,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
        ),
    )
    kuka_pedestal: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KukaPedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.30, -0.05, 0.175)),
        spawn=sim_utils.CuboidCfg(
            size=(0.32, 0.38, 1.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.24), roughness=0.5),
        ),
    )
    frame: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AtelierFrame",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.90, -0.35)),
        spawn=UsdFileCfg(usd_path=_FRAME_USD),
    )
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.35)),
        spawn=GroundPlaneCfg(),
        collision_group=-1,
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/AtelierLight",
        spawn=sim_utils.DomeLightCfg(color=(0.86, 0.91, 1.0), intensity=2400.0),
    )


@configclass
class JointActionsCfg:
    """Bounded joint-position offsets for initial PPO experiments."""

    kuka_arm = base_mdp.JointPositionActionCfg(
        asset_name="kuka", joint_names=_KUKA_ARM_JOINTS, scale=0.5, use_default_offset=True
    )
    humanoid_arm = base_mdp.JointPositionActionCfg(
        asset_name="humanoid", joint_names=_HUMANOID_ARM_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class IkActionsCfg:
    """KUKA pose and humanoid position actions used only by the scripted demo."""

    kuka_arm = base_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="kuka",
        joint_names=_KUKA_ARM_JOINTS,
        body_name="palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.6}
        ),
    )
    humanoid_arm = base_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="humanoid",
        joint_names=_HUMANOID_ARM_JOINTS,
        body_name="right_hand_pitch_link",
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
    )


@configclass
class ActionsCfg(PresetCfg):
    """Joint-space PPO and task-space state-machine action presets."""

    joint: JointActionsCfg = JointActionsCfg()
    ik: IkActionsCfg = IkActionsCfg()
    default = joint


@configclass
class ObservationsCfg:
    """Proprioception, cloth shape, and last action for PPO."""

    @configclass
    class PolicyCfg(ObsGroup):
        kuka_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("kuka", joint_names=_KUKA_ARM_JOINTS)}
        )
        kuka_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("kuka", joint_names=_KUKA_ARM_JOINTS)}
        )
        humanoid_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("humanoid", joint_names=_HUMANOID_ARM_JOINTS)},
        )
        humanoid_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("humanoid", joint_names=_HUMANOID_ARM_JOINTS)},
        )
        cloth_edges = ObsTerm(func=mdp.cloth_edge_positions)
        cloth_press_profile = ObsTerm(func=mdp.cloth_press_profile)
        kuka_hand = ObsTerm(
            func=base_mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("kuka", body_names="palm_link")}
        )
        humanoid_hand = ObsTerm(
            func=base_mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("humanoid", body_names="right_hand_pitch_link")},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """A minimal dual-hand approach and local cloth-draping objective."""

    dual_hand_proximity = RewTerm(func=mdp.hand_cloth_proximity, params={"std": 0.25}, weight=3.0)
    cloth_press = RewTerm(func=mdp.cloth_press_target, params={"target_asymmetry": 0.012, "std": 0.006}, weight=2.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-3)


@configclass
class EventsCfg:
    """Reset the two articulations and VBD nodal state without teleporting during an episode."""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class TerminationsCfg:
    """Time limit and cloth-drop boundary."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cloth_outside_workspace = DoneTerm(
        func=mdp.cloth_outside_workspace,
        params={"x_bounds": (-0.20, 1.30), "y_bounds": (-0.52, 0.52), "z_bounds": (0.67, 1.70)},
    )


@configclass
class TextileAtelierEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based, PPO-ready textile draping scene with coupled VBD contacts."""

    scene: TextileAtelierSceneCfg = TextileAtelierSceneCfg(num_envs=4, env_spacing=3.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Configure the coupled Newton rigid/VBD simulation and camera."""
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic", ls_iterations=50, integrator="implicitfast", nconmax=128
                        ),
                        bodies=[
                            r"/World/envs/env_[^/]+/Kuka",
                            r"/World/envs/env_[^/]+/Humanoid",
                            r"/World/envs/env_[^/]+/Tabletop",
                            r"/World/envs/env_[^/]+/Support(Neg|Pos)Y",
                        ],
                    ),
                    CouplerEntryCfg(
                        name="soft",
                        solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=2048),
                        all_particles=True,
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="soft",
                        bodies=[
                            r"/World/envs/env_[^/]+/Kuka/.*palm_link",
                            r"/World/envs/env_[^/]+/Kuka/.*(index|middle|thumb)_link_(1|2|3)",
                            r"/World/envs/env_[^/]+/Humanoid/.*right_hand_pitch_link",
                            r"/World/envs/env_[^/]+/Humanoid/.*R_(index|middle|thumb).*_link",
                            r"/World/envs/env_[^/]+/Tabletop",
                            r"/World/envs/env_[^/]+/Support(Neg|Pos)Y",
                        ],
                        collide_interval=1,
                        collision_pipeline=NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=False),
                    )
                ],
                iterations=1,
            ),
            soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=8.0e3, soft_contact_kd=1.0e-2, soft_contact_mu=10.0),
            num_substeps=2,
        )
        self.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            eye=(1.50, -1.65, 1.45), lookat=(0.55, 0.0, 0.90), window_width=1280, window_height=720
        )

    def play_mode(self) -> None:
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 4)
