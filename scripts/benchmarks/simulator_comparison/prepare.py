# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare shared URDF inputs without changing either simulator's source assets."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def prepare_franka(genesis_path: Path, output_dir: Path) -> dict:
    """Prepare the pinned Panda with a fixed open gripper and authored arm limits."""
    source = genesis_path / "genesis/assets/urdf/panda_bullet/panda.urdf"
    tree = ET.parse(source)
    assets = {}
    for mesh in tree.iter("mesh"):
        path = (source.parent / mesh.attrib["filename"].removeprefix("package://")).resolve()
        assets[str(path.relative_to(genesis_path.resolve()))] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh.set("filename", str(path))
    names = [f"panda_joint{index}" for index in range(1, 8)]
    joints = {joint.attrib["name"]: joint for joint in tree.iter("joint")}
    limits = [joints[name].find("limit").attrib for name in names]
    chain = []
    for name in names:
        joint = joints[name]
        origin = joint.find("origin")
        chain.append(
            {
                "origin_xyz": [float(value) for value in origin.attrib["xyz"].split()],
                "origin_rpy": [float(value) for value in origin.attrib["rpy"].split()],
                "axis": [float(value) for value in joint.find("axis").attrib["xyz"].split()],
            }
        )
        ET.SubElement(joint, "dynamics", {"damping": "0", "friction": "0"})
    # Bake each authored prismatic displacement into its fixed-joint origin.
    # This removes gripper/mimic control differences from a seven-action reach task.
    fingers = {}
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint = joints[name]
        opening = float(joint.find("limit").attrib["upper"])
        origin = joint.find("origin")
        xyz = [float(value) for value in origin.attrib["xyz"].split()]
        axis = [float(value) for value in joint.find("axis").attrib["xyz"].split()]
        origin.set("xyz", " ".join(str(value + opening * direction) for value, direction in zip(xyz, axis)))
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag in {"limit", "axis", "mimic", "dynamics"}:
                joint.remove(child)
        fingers[name] = opening
    zero_inertia_markers = []
    for link in tree.iter("link"):
        if link.attrib["name"] in {"panda_link8", "panda_grasptarget"}:
            inertial = link.find("inertial")
            if float(inertial.find("mass").attrib["value"]) != 0.0:
                raise ValueError(f"Expected a massless Panda marker: {link.attrib['name']}")
            inertia = inertial.find("inertia")
            for component in inertia.attrib:
                inertia.set(component, "0")
            zero_inertia_markers.append(link.attrib["name"])
    tree.write(output_dir / "franka.urdf", encoding="unicode", xml_declaration=True)
    return {
        "joint_names": names,
        "initial_joint_pos": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, math.pi / 4],
        "joint_lower_limits": [float(limit["lower"]) for limit in limits],
        "joint_upper_limits": [float(limit["upper"]) for limit in limits],
        "kp": 80.0,
        "kd": 4.0,
        "effort_limits": [float(limit["effort"]) for limit in limits],
        "end_effector_link": "panda_link7",
        "end_effector_offset": [0.0, 0.0, 0.212],
        "kinematic_chain": chain,
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "meshes_sha256": assets,
        "normalization": {
            "fixed_finger_positions_m": fingers,
            "zero_mass_marker_inertias_zeroed": zero_inertia_markers,
            "joint_damping": 0.0,
            "joint_friction": 0.0,
        },
    }


def prepare(genesis_path: Path, output_dir: Path) -> dict:
    """Write identical model inputs for both engines and record their hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = genesis_path / "genesis/assets/urdf/anymal_c/urdf/anymal_c.urdf"
    tree = ET.parse(source)
    assets = {}
    for mesh in tree.iter("mesh"):
        path = (source.parent / mesh.attrib["filename"]).resolve()
        assets[str(path.relative_to(genesis_path.resolve()))] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh.set("filename", str(path))
    unbounded_joints = []
    # The bundled ANYmal URDF omits angular bounds on revolute joints.
    # Genesis treats those as unbounded; the Isaac importer otherwise locks
    # them at zero. Make that interpretation explicit in BOTH engine inputs.
    for joint in tree.iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and (
            limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib
        ):
            joint.set("type", "continuous")
            unbounded_joints.append(joint.attrib["name"])
    # Joint friction and damping are explicit in the shared input.
    for joint in tree.iter("joint"):
        if joint.get("type") != "fixed":
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", "0")
            dynamics.set("friction", "0")
    tree.write(output_dir / "anymal.urdf", encoding="unicode", xml_declaration=True)
    # This deliberately simple model has no network asset dependency.
    (output_dir / "cartpole.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="benchmark_cartpole">
  <link name="base"><inertial><mass value="1"/>
    <inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="cart">
    <inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size=".3 .2 .15"/></geometry>
      <material name="blue"><color rgba=".1 .4 .9 1"/></material></visual>
    <collision><geometry><box size=".3 .2 .15"/></geometry></collision>
  </link>
  <joint name="slider" type="prismatic"><parent link="base"/><child link="cart"/><axis xyz="1 0 0"/>
    <origin xyz="0 0 .3"/><limit lower="-3" upper="3" effort="100" velocity="100"/>
    <dynamics damping="0" friction="0"/></joint>
  <link name="pole">
    <inertial><origin xyz="0 0 .5"/><mass value=".1"/>
      <inertia ixx=".0083333333" iyy=".0083333333" izz=".00002" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 .5"/><geometry><cylinder radius=".02" length="1"/></geometry>
      <material name="orange"><color rgba="1 .4 .05 1"/></material></visual>
    <collision><origin xyz="0 0 .5"/><geometry><cylinder radius=".02" length="1"/></geometry></collision>
  </link>
  <joint name="pole" type="continuous"><parent link="cart"/><child link="pole"/><axis xyz="0 1 0"/>
    <dynamics damping="0" friction="0"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    manifest = {
        "meshes_sha256": assets,
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalization": {
            "unbounded_revolute_to_continuous": unbounded_joints,
            "joint_damping": 0.0,
            "joint_friction": 0.0,
        },
    }
    manifest["franka"] = prepare_franka(genesis_path, output_dir)
    # Absolute mesh paths vary between machines; the original model/mesh hashes identify the physical inputs.
    manifest["prepared_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.glob("*.urdf")
    }
    (output_dir / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genesis_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.genesis_path, args.output_dir)
