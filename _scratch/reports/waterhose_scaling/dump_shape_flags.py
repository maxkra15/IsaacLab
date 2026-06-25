# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump the Newton model's shapes with their VISIBLE / COLLIDE_SHAPES / COLLIDE_PARTICLES flags.

Shows how the Newton GL viewer categorizes geometry (Visuals = VISIBLE bit, Collisions = COLLIDE_SHAPES
bit). Run kitless:

    ./isaaclab.sh -p _scratch/reports/waterhose_scaling/dump_shape_flags.py --headless --visualizer none
"""

from __future__ import annotations

import argparse
import collections

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Waterhose-Coupled-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.scene.num_envs = 1

with launch_simulation(env_cfg, args):
    import gymnasium as gym  # noqa: E402
    from newton import ShapeFlags  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    model = NewtonManager.get_model()
    flags = model.shape_flags.numpy()
    shape_body = model.shape_body.numpy()
    shape_color = model.shape_color.numpy() if getattr(model, "shape_color", None) is not None else None
    body_label = list(model.body_label)
    # shape-level label/key (try a few attr names)
    shape_label = None
    for attr in ("shape_key", "shape_label"):
        if hasattr(model, attr) and getattr(model, attr) is not None:
            shape_label = list(getattr(model, attr))
            print(f">>> shape label source: model.{attr}")
            break

    V = int(ShapeFlags.VISIBLE)
    C = int(ShapeFlags.COLLIDE_SHAPES)
    P = int(ShapeFlags.COLLIDE_PARTICLES)

    def categorize(label: str) -> str:
        t = label.lower()
        if "collider" in t:
            return "fridge housing hulls (Cable008_Collider*)"
        if "bodycollision" in t:
            return "fridge body welded mesh"
        if "socketcollision" in t:
            return "fridge socket"
        if "visuals" in t or "/cable008/visuals" in t:
            return "fridge VISUAL mesh"
        if "plug" in t:
            return "plug"
        if "cable" in t:
            return "cable"
        if "anchor" in t:
            return "anchor"
        if "ground" in t:
            return "ground"
        if any(k in t for k in ("torso", "arm", "gripper", "head", "wheel", "base", "link", "rby1", "robot")):
            return "robot"
        return f"OTHER: {label[:60]}"

    def name_of(i: int) -> str:
        if shape_label is not None and i < len(shape_label):
            return str(shape_label[i])
        b = int(shape_body[i])
        return f"(body<0)" if b < 0 else str(body_label[b]) if b < len(body_label) else f"body{b}"

    cats = collections.defaultdict(lambda: {"VISIBLE_only": 0, "COLLIDE_only": 0, "BOTH": 0, "particles": 0, "neither": 0, "total": 0})
    for i in range(len(flags)):
        f = int(flags[i])
        cat = categorize(name_of(i))
        d = cats[cat]
        d["total"] += 1
        if f & P:
            d["particles"] += 1
        vis = bool(f & V)
        col = bool(f & C)
        if vis and col:
            d["BOTH"] += 1
        elif vis:
            d["VISIBLE_only"] += 1
        elif col:
            d["COLLIDE_only"] += 1
        else:
            d["neither"] += 1

    print(f"\n>>> total shapes: {len(flags)}  (ShapeFlags VISIBLE={V} COLLIDE_SHAPES={C} COLLIDE_PARTICLES={P})\n")
    print(f"{'category':42s} {'total':>6} {'VIS-only':>9} {'COL-only':>9} {'BOTH':>6} {'particle':>9} {'neither':>8}")
    for cat in sorted(cats):
        d = cats[cat]
        print(f"{cat:42s} {d['total']:6d} {d['VISIBLE_only']:9d} {d['COLLIDE_only']:9d} {d['BOTH']:6d} {d['particles']:9d} {d['neither']:8d}")
    print("\nLegend: VIS-only -> shows ONLY under Visuals; COL-only -> ONLY under Collisions; BOTH -> under both.")

    if shape_color is not None:
        print("\n>>> per-shape color (the only material info the GL viewer uses -- no texture maps / PBR):")
        by_cat_colors = collections.defaultdict(list)
        for i in range(len(flags)):
            by_cat_colors[categorize(name_of(i))].append(tuple(round(float(c), 3) for c in shape_color[i]))
        for cat in sorted(by_cat_colors):
            cols = by_cat_colors[cat]
            uniq = sorted(set(cols))
            sample = uniq[:3]
            print(f"  {cat:42s} {len(uniq):3d} distinct color(s); e.g. {sample}")
    env.close()
