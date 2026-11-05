# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate one adapter's physical and image contracts in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


def validate(config: dict, output: Path) -> dict:
    """Exercise one scene and write measured evidence, including failed checks.

    Args:
        config: A worker configuration with an adapter module and prepared URDF path.
        output: JSON destination; camera samples are written beside this file.

    Returns:
        The validation result. A failed check gives ``status="failed"``.
    """
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    result = {
        "schema_version": 1,
        "kind": "adapter_validation",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "failed",
        "config": config,
        "checks": {},
    }
    checks = result["checks"]
    adapter = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        module = importlib.import_module(config["adapter_module"])
        import torch
        from tasks import ANYMAL_JOINTS, Task

        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        adapter = module.Adapter(config)
        task = Task(adapter, config)
        task.reset()
        adapter.synchronize()
        result["backend_metadata"] = adapter.metadata
        state = adapter.state()
        _record(checks, "finite_state", all(bool(torch.isfinite(value).all()) for value in state.values()))
        if not checks["finite_state"]["passed"]:
            raise FloatingPointError("Initial physics state contains non-finite values")
        reach = config["task"] == "franka"
        if task.walk:
            expected_names = ANYMAL_JOINTS
        elif reach:
            expected_names = [f"panda_joint{index}" for index in range(1, 8)]
        else:
            expected_names = ["slider", "pole"]
        _record(checks, "joint_order", list(adapter.joint_names) == expected_names, actual=list(adapter.joint_names))
        initial = torch.tensor(config["initial_joint_pos"], device=config["device"], dtype=torch.float32)
        _validate_reset(adapter, config, initial, checks)
        if task.walk:
            expected_mass = sum(
                float(item.attrib["value"]) for item in ET.parse(config["asset_path"]).findall("link/inertial/mass")
            )
            adapter.reset()
            adapter.apply_action(initial.expand(task.n, -1))
            for _ in range(16):
                adapter.physics_step()
            adapter.synchronize()
            state = adapter.state()
            upright = 1 - 2 * (state["root_quat"][:, 1].square() + state["root_quat"][:, 2].square())
            joint_deflection = (state["joint_pos"] - initial).abs().max().item()
            _record(
                checks,
                "anymal_short_stability",
                all(bool(torch.isfinite(value).all()) for value in state.values())
                and bool((state["root_pos"][:, 2] > 0.25).all())
                and bool((upright > 0.8).all())
                and joint_deflection < 0.5,
                steps=16,
                duration_s=16 * config["dt"],
                minimum_height_m=state["root_pos"][:, 2].min().item(),
                minimum_upright=upright.min().item(),
                maximum_joint_deflection_rad=joint_deflection,
            )
        elif reach:
            expected_mass = sum(
                float(link.find("inertial/mass").attrib["value"])
                for link in ET.parse(config["asset_path"]).findall("link")
                if link.attrib["name"] != "panda_link0" and link.find("inertial/mass") is not None
            )
            _validate_franka(adapter, config, initial, checks)
        else:
            expected_mass = 1.1
            _validate_cartpole(adapter, config, checks)
        actual_mass = adapter.metadata.get("moving_mass_kg")
        _record(
            checks,
            "moving_mass",
            isinstance(actual_mass, (int, float)) and math.isclose(actual_mass, expected_mass, abs_tol=1e-3),
            expected_kg=expected_mass,
            actual_kg=actual_mass,
            tolerance_kg=1e-3,
        )
        if config["vision"] == "rgb":
            _validate_images(adapter, config, initial, output.parent, checks)
        if adapter.metadata.get("physics_backend") == "Isaac Lab Newton / MuJoCo Warp":
            _validate_mjwarp(adapter, config, initial, output.parent, checks)
            result["backend_diagnostics"] = adapter.diagnostics()
        result["status"] = "ok" if all(check["passed"] for check in checks.values()) else "failed"
        if result["status"] == "failed":
            result["error"] = "Failed: " + ", ".join(name for name, check in checks.items() if not check["passed"])
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Preserve completed evidence even if native teardown crashes the process.
        output.write_text(json.dumps(_json_evidence(result), indent=2, allow_nan=False) + "\n")
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                result.update(status="failed", teardown_error=f"{type(error).__name__}: {error}")
                output.write_text(json.dumps(_json_evidence(result), indent=2, allow_nan=False) + "\n")
    return result


def _record(checks: dict, name: str, passed: bool, **evidence) -> None:
    checks[name] = {"passed": passed, **evidence}


def _json_evidence(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {name: _json_evidence(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_evidence(item) for item in value]
    return value


def _validate_reset(adapter, config: dict, initial, checks: dict) -> None:
    import torch

    count = config["num_envs"]
    positions = initial.expand(count, -1).clone()
    positions[:, 0] += torch.linspace(-0.05, 0.05, count, device=config["device"])
    adapter.reset(joint_pos=positions)
    for _ in range(2):
        adapter.physics_step()
    previous = {name: value.clone() for name, value in adapter.state().items()}
    ids = torch.arange(0, count, 2, device=config["device"])
    others = torch.arange(1, count, 2, device=config["device"])
    adapter.reset(ids, joint_pos=initial.expand(len(ids), -1))
    state = adapter.state()
    position_error = (state["joint_pos"][ids] - initial).abs().max().item()
    speed = state["joint_vel"][ids].abs().max().item()
    untouched_error = (
        max((state[name][others] - previous[name][others]).abs().max().item() for name in state) if len(others) else 0.0
    )
    _record(
        checks,
        "subset_reset",
        position_error < 1e-5 and speed < 1e-5 and untouched_error < 1e-5,
        maximum_joint_position_error=position_error,
        maximum_joint_speed=speed,
        maximum_unselected_state_change=untouched_error,
        selected_count=len(ids),
        unselected_count=len(others),
    )
    adapter.reset()
    quat = adapter.state()["root_quat"]
    # A quaternion and its negation represent the same orientation.
    quaternion_error = torch.cat(((quat[:, :1].abs() - 1).abs(), quat[:, 1:].abs()), -1).max().item()
    _record(checks, "root_quaternion_wxyz", quaternion_error < 1e-5, maximum_error=quaternion_error)


def _validate_cartpole(adapter, config: dict, checks: dict) -> None:
    import torch

    count, device = config["num_envs"], config["device"]
    adapter.reset(joint_pos=torch.tensor([[0.0, 0.1]], device=device).expand(count, -1))
    adapter.apply_action(torch.zeros((count, 1), device=device))
    for _ in range(8):
        adapter.physics_step()
    adapter.synchronize()
    state = adapter.state()
    reference = _cartpole_reference(config["dt"], 8)
    actual = torch.cat((state["joint_pos"], state["joint_vel"]), -1).double().cpu()
    error = (actual - reference).abs().amax(0)
    tolerance = torch.tensor([2e-4, 5e-4, 2e-3, 5e-3], dtype=torch.float64)
    _record(
        checks,
        "cartpole_independent_dynamics",
        bool((error < tolerance).all()) and bool((actual[:, 0] < 0).all()) and bool((actual[:, 1] > 0.1).all()),
        fields=["cart_position_m", "pole_angle_rad", "cart_velocity_m_s", "pole_velocity_rad_s"],
        reference=reference.tolist(),
        actual_first_env=actual[0].tolist(),
        maximum_absolute_errors=error.tolist(),
        absolute_tolerances=tolerance.tolist(),
        duration_s=8 * config["dt"],
        reference_method="Independent Lagrange equations; float64 RK4, 25 substeps per simulator step",
    )
    adapter.reset(joint_pos=torch.zeros((count, 2), device=device))
    adapter.apply_action(torch.full((count, 1), 10.0, device=device))
    for _ in range(16):
        adapter.physics_step()
    adapter.synchronize()
    q = adapter.state()["joint_pos"]
    _record(
        checks,
        "cartpole_force_direction",
        bool((q[:, 0] > 1e-3).all()) and bool((q[:, 1] < 0).all()),
        force_n=10.0,
        duration_s=16 * config["dt"],
        minimum_cart_displacement_m=q[:, 0].min().item(),
        maximum_pole_angle_rad=q[:, 1].max().item(),
    )


def _cartpole_reference(dt: float, steps: int):
    """Integrate the specified cart and uniform rod independently on the CPU."""
    import torch

    state = torch.tensor([0.0, 0.1, 0.0, 0.0], dtype=torch.float64)
    step_size = dt / 25

    def derivative(value):
        _, angle, cart_velocity, pole_velocity = value
        # M = cart + pole mass; J = pole COM inertia + m*l^2, l = half-length.
        total_mass, rotational_inertia, mass_length = 1.1, 0.0083333333 + 0.1 * 0.5**2, 0.1 * 0.5
        coupling = mass_length * angle.cos()
        cart_rhs = mass_length * angle.sin() * pole_velocity.square()
        pole_rhs = mass_length * 9.81 * angle.sin()
        determinant = total_mass * rotational_inertia - coupling.square()
        cart_acceleration = (rotational_inertia * cart_rhs - coupling * pole_rhs) / determinant
        pole_acceleration = (total_mass * pole_rhs - coupling * cart_rhs) / determinant
        return torch.stack((cart_velocity, pole_velocity, cart_acceleration, pole_acceleration))

    for _ in range(steps * 25):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step_size * k1)
        k3 = derivative(state + 0.5 * step_size * k2)
        k4 = derivative(state + step_size * k3)
        state += step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state


def _validate_franka(adapter, config: dict, initial, checks: dict) -> None:
    """Compare native tool positions with independent float64 URDF kinematics."""
    import torch

    joints = {joint.attrib["name"]: joint for joint in ET.parse(config["asset_path"]).findall("joint")}
    by_child = {joint.find("child").attrib["link"]: joint for joint in joints.values()}
    chain = []
    # Use the authored grasp frame, independently of the adapter's merged-body offset.
    link = "panda_grasptarget"
    while link in by_child:
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    chain.reverse()
    lower = torch.tensor(
        [float(joints[name].find("limit").attrib["lower"]) for name in adapter.joint_names], dtype=torch.float64
    )
    upper = torch.tensor(
        [float(joints[name].find("limit").attrib["upper"]) for name in adapter.joint_names], dtype=torch.float64
    )
    columns = torch.arange(len(adapter.joint_names), dtype=torch.float64)
    rows = torch.linspace(-1, 1, config["num_envs"], dtype=torch.float64).unsqueeze(-1)
    errors = []
    for phase in (0.0, 1.0, 2.0):
        positions = initial.double().cpu() + 0.15 * torch.sin(columns + phase) + 0.05 * rows
        positions = positions.clamp(lower + 0.02, upper - 0.02)
        adapter.reset(joint_pos=positions.float().to(config["device"]))
        adapter.synchronize()
        actual = adapter.state()["ee_pos"].double().cpu()
        expected = torch.stack(
            [_franka_reference_position(chain, adapter.joint_names, row, config) for row in positions]
        )
        errors.append((actual - expected).norm(dim=-1))
    maximum_error = torch.stack(errors).max().item()
    _record(
        checks,
        "franka_independent_end_effector_kinematics",
        maximum_error < 2e-4,
        maximum_position_error_m=maximum_error,
        tolerance_m=2e-4,
        pose_count=len(errors),
        environment_count=config["num_envs"],
        reference="Float64 transforms parsed from prepared URDF through the authored panda_grasptarget frame",
    )
    adapter.reset()
    adapter.apply_action(initial.expand(config["num_envs"], -1))
    for _ in range(16):
        adapter.physics_step()
    state = adapter.state()
    deviation = (state["joint_pos"] - initial).abs().max().item()
    base_error = (
        (state["root_pos"] - torch.tensor([0, 0, config["initial_height"]], device=config["device"])).abs().max().item()
    )
    _record(
        checks,
        "franka_short_stability",
        all(bool(torch.isfinite(value).all()) for value in state.values()) and deviation < 0.5 and base_error < 1e-5,
        maximum_joint_deflection_rad=deviation,
        maximum_fixed_base_displacement_m=base_error,
        duration_s=16 * config["dt"],
    )


def _franka_reference_position(chain, names: list[str], positions, config: dict):
    """Compose URDF origin and revolute transforms without backend FK utilities."""
    import torch

    transform = torch.eye(4, dtype=torch.float64)
    transform[2, 3] = config["initial_height"]
    coordinates = dict(zip(names, positions.tolist(), strict=True))
    for joint in chain:
        origin = joint.find("origin")
        xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        roll, pitch, yaw = rpy
        cr, sr, cp, sp, cy, sy = (
            math.cos(roll),
            math.sin(roll),
            math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw),
            math.sin(yaw),
        )
        local = torch.eye(4, dtype=torch.float64)
        local[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
        local[:3, :3] = torch.tensor(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=torch.float64,
        )
        transform = transform @ local
        if joint.attrib["type"] != "fixed":
            if joint.attrib["type"] not in ("revolute", "continuous"):
                raise ValueError(f"Unsupported Franka reference joint type: {joint.attrib['type']}")
            axis_element = joint.find("axis")
            axis = torch.tensor([float(value) for value in axis_element.attrib["xyz"].split()], dtype=torch.float64)
            axis /= axis.norm()
            x, y, z = axis.tolist()
            skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64)
            angle = coordinates[joint.attrib["name"]]
            rotation = torch.eye(4, dtype=torch.float64)
            rotation[:3, :3] += math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
            transform = transform @ rotation
    return transform[:3, 3]


def _validate_images(adapter, config: dict, initial, directory: Path, checks: dict) -> None:
    import torch
    from PIL import Image

    count = config["num_envs"]
    positions = initial.expand(count, -1).clone()
    positions[:, 0] += torch.linspace(-0.2, 0.2, count, device=config["device"])
    positions[:, 1] += torch.linspace(-0.3, 0.3, count, device=config["device"])
    adapter.reset(joint_pos=positions)
    images = adapter.render()
    adapter.synchronize()
    expected = (count, config["height"], config["width"], 3)
    shape_ok = isinstance(images, torch.Tensor) and tuple(images.shape) == expected
    dtype_ok = shape_ok and images.dtype == torch.uint8 and images.device.type == "cuda"
    _record(
        checks,
        "rgb_tensor_contract",
        dtype_ok,
        shape=list(images.shape),
        expected_shape=list(expected),
        dtype=str(images.dtype),
        device=str(images.device),
    )
    if not dtype_ok:
        return
    pixels = images.float()
    minimum_std = pixels.flatten(1).std(dim=1).min().item()
    _record(checks, "rgb_nonblank_every_env", minimum_std > 1.0, minimum_pixel_std=minimum_std)
    difference = (pixels[0] - pixels[-1]).abs().mean().item() if count > 1 else None
    _record(
        checks,
        "rgb_distinct_environments",
        count == 1 or difference > 0.05,
        first_last_mean_absolute_pixel_difference=difference,
        note="A single environment cannot exercise cross-environment camera isolation" if count == 1 else "",
    )
    samples = []
    for index in sorted({0, count - 1}):
        filename = f"validation_camera_env{index}.png"
        Image.fromarray(images[index].cpu().numpy()).save(directory / filename)
        samples.append(filename)
    checks["rgb_tensor_contract"]["samples"] = samples


def _validate_mjwarp(adapter, config: dict, initial, directory: Path, checks: dict) -> None:
    """Verify graph-bound controls using paired trajectories from identical resets."""
    import sys

    import torch

    position_control = config["task"] != "cartpole"
    nominal = (
        initial.expand(config["num_envs"], -1)
        if position_control
        else torch.zeros((config["num_envs"], 1), device=config["device"])
    )
    adapter.reset()
    adapter.apply_action(nominal)
    for _ in range(8):
        adapter.physics_step()
    graph = adapter._sim.backend_manager.backend.graph
    _record(checks, "mjwarp_graph_after_warmup", adapter.diagnostics()["graph_active"])
    trajectories = []
    for action in (nominal, nominal + (0.1 if position_control else 10.0)):
        adapter.reset()
        adapter.apply_action(action)
        for _ in range(16):
            adapter.physics_step()
        adapter.synchronize()
        trajectories.append(adapter.state()["joint_pos"].clone())
    response = trajectories[1] - trajectories[0]
    response_per_env = response.abs().amax(-1) if position_control else response[:, 0]
    _record(
        checks,
        "mjwarp_changed_control_after_capture",
        bool((response_per_env > 1e-3).all()),
        response_per_env=response_per_env.tolist(),
        control_delta=0.1 if position_control else 10.0,
        control_units="rad" if position_control else "N",
        response_units="rad" if position_control else "m",
        comparison="Changed versus nominal controls, identical initial states and 16 physics steps",
    )
    _record(checks, "mjwarp_commands_preserve_graph", graph is adapter._sim.backend_manager.backend.graph)
    diagnostics = adapter.diagnostics()
    _record(
        checks,
        "mjwarp_no_capacity_overflow_at_end",
        all(flag == 0 for flag in diagnostics["fatal_capacity_flags_at_end"]),
        **diagnostics,
    )
    kit_modules = sorted(name for name in sys.modules if name == "omni.kit" or name.startswith("omni.kit."))
    _record(checks, "mjwarp_kitless_runtime", not kit_modules, loaded_kit_modules=kit_modules)
    _record(
        checks,
        "mjwarp_finite_after_controls",
        all(bool(torch.isfinite(value).all()) for value in adapter.state().values()),
    )
    if config["task"] == "anymal" and config["vision"] == "rgb" and config.get("camera_motion") == "root_translation":
        _validate_camera_motion(adapter, config, directory, checks)


def _validate_camera_motion(adapter, config: dict, directory: Path, checks: dict) -> None:
    """Check every camera's world pose and rendered framing after a root translation."""
    import torch
    from PIL import Image

    adapter.reset()
    before = adapter.render().clone()
    root_before = adapter.state()["root_pos"].clone()
    eye_before, quat_before = (value.clone() for value in adapter._camera.get_world_poses())
    delta = torch.zeros_like(root_before)
    delta[:, 0] = 3.0
    pose = adapter._robot.data.root_pose_w.torch.clone()
    pose[:, :3] += delta
    adapter._robot.write_root_pose_to_sim_index(root_pose=pose)
    after = adapter.render().clone()
    adapter.synchronize()
    eye_after, quat_after = adapter._camera.get_world_poses()
    root_error = (adapter.state()["root_pos"] - root_before - delta).abs().max().item()
    eye_error = (eye_after - eye_before - delta).abs().max().item()
    orientation_error = (
        torch.minimum((quat_after - quat_before).norm(dim=-1), (quat_after + quat_before).norm(dim=-1)).max().item()
    )
    _record(checks, "camera_follow_root_translation", root_error < 1e-4, maximum_root_error_m=root_error)
    _record(checks, "camera_follow_every_env", eye_error < 1e-4, maximum_camera_error_m=eye_error)
    _record(checks, "camera_follow_fixed_orientation", orientation_error < 1e-5, maximum_error=orientation_error)
    centroids = []
    for label, frames in (("before", before), ("after", after)):
        pixels = frames.float()
        # The shared ANYmal model has a red shell, which supplies an image-space oracle.
        shell = (
            (pixels[..., 0] > 1.3 * pixels[..., 1]) & (pixels[..., 0] > 1.3 * pixels[..., 2]) & (pixels[..., 0] > 30)
        )
        centers = []
        for index in range(config["num_envs"]):
            y, x = shell[index].nonzero(as_tuple=True)
            if not len(x):
                raise AssertionError(f"No ANYmal shell pixels in {label} frame, environment {index}")
            centers.append(torch.stack((x.float().mean(), y.float().mean())))
            Image.fromarray(frames[index].cpu().numpy()).save(directory / f"movement_{label}_env{index}.png")
        centroids.append(torch.stack(centers))
    displacement = (centroids[1] - centroids[0]).norm(dim=-1)
    minimum_std = min(
        before.float().flatten(1).std(dim=1).min().item(), after.float().flatten(1).std(dim=1).min().item()
    )
    _record(
        checks,
        "camera_follow_stable_image_framing",
        bool((displacement <= 2.0).all()) and minimum_std > 1.0,
        maximum_displacement_pixels=displacement.max().item(),
        tolerance_pixels=2.0,
        minimum_pixel_std=minimum_std,
        before_centroids_xy_pixels=centroids[0].tolist(),
        after_centroids_xy_pixels=centroids[1].tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.config.read_text()), args.output)
    raise SystemExit(0 if result["status"] == "ok" else 1)
