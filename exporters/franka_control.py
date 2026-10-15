# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export native Franka kinematics and action contracts for browser teleoperation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def manual_control(model: newton.Model, robot: Articulation, root: int, hand: int, kind: str) -> dict:
    """Describe the actual joint frames, limits and existing action adapter; never edit physics state."""
    parents, children, types, coordinates, dofs, parent_frames, child_frames, axes, lower, upper = (
        getattr(model, name).numpy()
        for name in (
            "joint_parent",
            "joint_child",
            "joint_type",
            "joint_q_start",
            "joint_qd_start",
            "joint_X_p",
            "joint_X_c",
            "joint_axis",
            "joint_limit_lower",
            "joint_limit_upper",
        )
    )
    labels = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(model.joint_label)}
    joints = [labels[name] for name in robot.joint_names]
    indices = coordinates[joints].tolist()
    chain, child = [], hand
    while child != root:
        joint = int(np.flatnonzero(children == child)[0])
        joint_type = int(types[joint])
        if joint_type not in (newton.JointType.REVOLUTE, newton.JointType.FIXED):
            raise ValueError(f"Unreviewed Franka arm joint: {model.joint_label[joint]}")
        chain.append(
            {
                "parent": parent_frames[joint].tolist(),
                "child": child_frames[joint].tolist(),
                "coordinate": int(coordinates[joint]) if joint_type == newton.JointType.REVOLUTE else None,
                "axis": axes[dofs[joint]].tolist() if joint_type == newton.JointType.REVOLUTE else None,
            }
        )
        child = int(parents[joint])
    chain.reverse()
    if [step["coordinate"] for step in chain if step["coordinate"] is not None] != indices[:7]:
        raise ValueError("Franka arm order differs from the control contract")
    mode, scale, gripper = {
        "reach": ("absolute", 0.5, "none"),
        "lift": ("relative", 0.1, "relative"),
        "drawer": ("absolute", 1.0, "binary"),
        "soft": ("relative", 0.03, "normalized"),
        "cloth": ("relative", 0.03, "normalized"),
        "stack": ("position", 1.0, "position"),
    }[kind]
    namespace = getattr(model, "mujoco", None)
    gravcomp = getattr(namespace, "gravcomp", None)
    masses = model.body_mass.numpy() * (1 - gravcomp.numpy() if gravcomp is not None else 1)
    descendants = []
    for joint in joints[:7]:
        bodies = {int(children[joint])}
        for _ in range(len(joints)):
            bodies.update(int(child) for parent, child in zip(parents, children, strict=True) if int(parent) in bodies)
        descendants.append(sorted(bodies))
    stiffness = model.joint_target_ke.numpy()[dofs[joints[:7]]]
    if np.any(stiffness <= 0):
        raise ValueError("Manual gravity compensation requires position-actuator stiffness")
    return {
        "rootBody": root,
        "handBody": hand,
        "offset": [0, 0, 0.1034],
        "chain": chain,
        "qIndices": indices,
        "dofIndices": dofs[joints].tolist(),
        "limits": [[float(lower[dofs[joint]]), float(upper[dofs[joint]])] for joint in joints],
        "defaults": robot.data.default_joint_pos.torch.numpy()[0].tolist(),
        "actionMode": mode,
        "actionScale": scale,
        "gripper": gripper,
        "binding": "manual_target" if kind == "stack" else "action",
        "modeBinding": "manual" if kind == "stack" else None,
        "settleSteps": 90 if kind == "drawer" else 0,
        "homeOffset": [-0.06, 0, 0.02] if kind == "drawer" else [0, 0, 0],
        "motionGain": 0.5 if kind in ("soft", "cloth") else 1.0,
        "velocityDamping": 0.05 if kind in ("soft", "cloth") else 0.0,
        "gravity": model.gravity.numpy()[0].tolist(),
        "masses": masses.tolist(),
        "centers": model.body_com.numpy().tolist(),
        "descendants": descendants,
        "stiffness": stiffness.tolist(),
    }


def main() -> None:
    """Emit metadata for existing physics bundles without rebuilding or changing those bundles."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("reach", "lift", "drawer", "soft", "cloth", "stack"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "stack":
        from stacking import scene_cfg

        cfg = scene_cfg()
    else:
        task = {
            "reach": "Isaac-Reach-Franka",
            "lift": "Isaac-Lift-Franka",
            "drawer": "Isaac-Open-Drawer-Franka",
            "soft": "Isaac-Lift-Soft-Franka",
            "cloth": "Isaac-Lift-Cloth-Franka",
        }[args.kind]
        overrides = [] if args.kind in ("soft", "cloth") else ["physics=newton_mjwarp"]
        if args.kind == "lift":
            overrides.append("presets=cube")
        cfg = parse_env_cfg(task, device="cpu", num_envs=1, overrides=overrides)
        cfg.play_mode()
        cfg.sim.physics.use_cuda_graph = False
        if args.kind == "lift":
            cfg.sim.physics.collision_cfg.rigid_contact_max = 512
        if args.kind == "drawer":
            # The immutable v1 capture predates the task's -1.1 rad reset pose.
            # Its absolute-action adapter uses the original -0.569 rad offset.
            cfg.scene.robot.init_state.joint_pos["panda_joint2"] = -0.569
        if args.kind in ("soft", "cloth"):
            cfg.scene.deformable.spawn.edge_refinement = 8 if args.kind == "soft" else 3
            cfg.sim.physics.solver_cfg.entries[0].solver_cfg.use_mujoco_contacts = True
            for entry in cfg.sim.physics.solver_cfg.entries:
                if hasattr(entry.solver_cfg, "particle_enable_tile_solve"):
                    entry.solver_cfg.particle_enable_tile_solve = False
    cfg.seed = 42
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            env.reset(seed=42)
            model = NewtonManager.get_model()
            root = next(i for i, label in enumerate(model.body_label) if label.endswith("/panda_link0"))
            hand = next(i for i, label in enumerate(model.body_label) if label.endswith("/panda_hand"))
            metadata = manual_control(model, env.scene["robot"], root, hand, args.kind)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
            # Keep native FK evidence beside build metadata, outside the deployed site.
            rng = np.random.default_rng(42)
            state = NewtonManager.get_state_0()
            initial = state.joint_q.numpy()
            samples, poses = [], []
            forward_state = model.state()
            for sample in range(32):
                q = initial.copy()
                for index, (lo, hi) in zip(metadata["qIndices"][:7], metadata["limits"][:7], strict=True):
                    q[index] = np.clip(q[index] + (rng.uniform(-0.4, 0.4) if sample else 0), lo, hi)
                newton.eval_fk(model, wp.array(q, dtype=float, device="cpu"), state.joint_qd, forward_state)
                samples.append(q)
                poses.append(forward_state.body_q.numpy().copy())
            np.savez(args.output.with_suffix(".reference.npz"), joint_q=samples, body_q=poses)
            print("EXPORTED", args.kind, "native arm frames and action adapter", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
