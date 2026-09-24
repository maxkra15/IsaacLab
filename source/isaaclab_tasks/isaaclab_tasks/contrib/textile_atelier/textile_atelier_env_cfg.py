# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-arm textile atelier with VBD cloth and rigid-body contact."""

from __future__ import annotations

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonSoftContactCfg, VBDSolverCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

from isaaclab_contrib.custom_coupling import CoupledMJWarpVBDSolverCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG

from . import mdp

_KUKA_ARM_JOINTS = ["iiwa7_joint_(1|2|3|4|5|6|7)"]
_HUMANOID_ARM_JOINTS = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
_FRAME_USD = str(Path(__file__).with_name("atelier_frame.usda"))


def _kuka_cfg() -> ArticulationCfg:
    """Place the KUKA on a pedestal with the authored arm facing the cloth."""
    joint_pos = dict(KUKA_ALLEGRO_CFG.init_state.joint_pos)
    joint_pos.pop("iiwa7_joint_(1|2|7)")
    joint_pos["iiwa7_joint_(1|7)"] = 0.0
    joint_pos["iiwa7_joint_2"] = -0.40
    cfg = KUKA_ALLEGRO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Kuka",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.30, -0.25, 0.70),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos=joint_pos,
        ),
    )
    cfg.spawn.fix_root_link = True
    arm = cfg.actuators["kuka_allegro_actuators"]
    arm.stiffness["iiwa7_joint_(1|2|3|4|5|6|7)"] = 800.0
    arm.damping = {
        name: 2.0 * value if name.startswith("iiwa7_joint_") else value for name, value in arm.damping.items()
    }
    return cfg


def _humanoid_cfg() -> ArticulationCfg:
    """Fix GR1T2 at its pelvis and hold its hands during VBD contact."""
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
    # Hold the open fingers against VBD contact; the asset's default hand drives are too weak.
    for hand_name in ("right-hand", "left-hand"):
        hand = cfg.actuators[hand_name]
        hand.stiffness = 100.0
        hand.damping = 10.0
        hand.armature = 0.01
    return cfg


@configclass
class TextileAtelierSceneCfg(InteractiveSceneCfg):
    """A hanging VBD curtain between an arm and a fixed-base humanoid."""

    kuka: ArticulationCfg = _kuka_cfg()
    humanoid: ArticulationCfg = _humanoid_cfg()

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 1.05), rot=(0.70710678, 0.0, 0.0, 0.70710678)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.90, 0.70),
            edge_refinement=20,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                func="isaaclab_tasks.contrib.textile_atelier.cloth_art:spawn_kinetic_tapestry_material",
                diffuse_color=(0.025, 0.045, 0.20),
                roughness=0.85,
            ),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.008,
                tri_ke=5.0e2,
                tri_ka=5.0e2,
                tri_kd=3.0e-3,
                edge_ke=0.5,
                edge_kd=3.0e-2,
            ),
        ),
    )

    kuka_pedestal: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KukaPedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.30, -0.25, 0.175)),
        spawn=sim_utils.CuboidCfg(
            size=(0.32, 0.38, 1.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.09, 0.11, 0.12), roughness=0.5),
        ),
    )
    frame: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AtelierFrame",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
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
        spawn=sim_utils.DomeLightCfg(color=(0.82, 0.88, 1.0), intensity=750.0),
    )
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/AtelierKeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.295, -0.208, -0.065, 0.930)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.87, 0.68), intensity=1650.0, angle=3.0),
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
    """Two position-IK actions used only by the scripted demo."""

    kuka_arm = base_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="kuka",
        joint_names=_KUKA_ARM_JOINTS,
        body_name="palm_link",
        controller=DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.15}
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
        curtain_deflection_profile = ObsTerm(func=mdp.curtain_deflection_profile)
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
    """A minimal dual-hand approach and front-normal curtain-shaping objective."""

    dual_hand_proximity = RewTerm(func=mdp.hand_cloth_proximity, params={"std": 0.25}, weight=3.0)
    curtain_deflection = RewTerm(
        func=mdp.curtain_deflection_target,
        params={"target_left": 0.04, "target_right": -0.04, "std": 0.03},
        weight=2.0,
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-3)


@configclass
class EventsCfg:
    """Reset the robots and pin the curtain's top edge to its visible rail."""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    pin_top_edge = EventTerm(func=mdp.pin_curtain_top_edge, mode="reset")


@configclass
class TerminationsCfg:
    """Time limit and generous curtain workspace boundary."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cloth_outside_workspace = DoneTerm(
        func=mdp.cloth_outside_workspace,
        params={"x_bounds": (-0.20, 1.30), "y_bounds": (-0.52, 0.52), "z_bounds": (0.40, 1.70)},
    )


@configclass
class TextileAtelierEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based, PPO-ready curtain shaping with dynamic VBD fabric."""

    scene: TextileAtelierSceneCfg = TextileAtelierSceneCfg(num_envs=4, env_spacing=9.0, replicate_physics=True)
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
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(
                    use_mujoco_contacts=True,
                    cone="elliptic",
                    ls_iterations=50,
                    integrator="implicitfast",
                    nconmax=128,
                ),
                soft_solver_cfg=VBDSolverCfg(
                    iterations=10,
                    rigid_body_particle_contact_buffer_size=2048,
                    integrate_with_external_rigid_solver=True,
                ),
                coupling_mode="two_way",
            ),
            soft_contact_cfg=NewtonSoftContactCfg(
                soft_contact_ke=8.0e3,
                soft_contact_kd=1.0e-2,
                soft_contact_mu=10.0,
            ),
            num_substeps=2,
        )
        self.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            eye=(2.15, -3.50, 1.72), lookat=(0.55, 0.0, 0.82), window_width=1280, window_height=720
        )

    def play_mode(self) -> None:
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 4)
