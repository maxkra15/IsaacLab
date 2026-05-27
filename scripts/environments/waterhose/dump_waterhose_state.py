"""Side-by-side state dumper for the success demo and the IsaacLab env.

Constructs each setup, then prints a comprehensive snapshot of body counts,
masses, flags, shape geometry / materials, contact pair counts, and (after a
small warmup) per-contact statistics so we can compare them directly.

Usage:
    # Inside the newton repo (success demo)
    cd /home/maximiliank/Work/newton
    uv run python /home/maximiliank/Work/IsaacLab-waterhose-demo/scripts/environments/waterhose/dump_waterhose_state.py success
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")


SEP = "=" * 100


def _array_np(value):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _stats(values, precision: int = 4) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "n=0"
    return f"n={arr.size} min={arr.min():.{precision}g} mean={arr.mean():.{precision}g} max={arr.max():.{precision}g}"


def _shape_ids_for_bodies(model, body_ids):
    shape_body = _array_np(getattr(model, "shape_body", None))
    if shape_body is None or not body_ids:
        return []
    body_set = {int(b) for b in body_ids}
    return [int(s) for s, b in enumerate(shape_body) if int(b) in body_set]


def dump_model_group(label, model, body_ids, shape_ids=None, indent="  "):
    """Print body+shape stats for one logical group of bodies."""
    shape_body = _array_np(getattr(model, "shape_body", None))
    if shape_ids is None and shape_body is not None and body_ids:
        body_set = {int(b) for b in body_ids}
        shape_ids = [s for s, b in enumerate(shape_body) if int(b) in body_set]

    print(f"{indent}--- {label} ---")
    if body_ids:
        bm = _array_np(model.body_mass)
        bim = _array_np(model.body_inv_mass)
        bf = _array_np(getattr(model, "body_flags", None))
        bq = _array_np(model.body_q)
        idx = np.asarray(body_ids, dtype=np.int64)
        print(f"{indent}bodies={len(body_ids)} mass({_stats(bm[idx], precision=4)})")
        print(f"{indent}        inv_mass({_stats(bim[idx], precision=4)})")
        if bf is not None and bf.shape[0] > idx.max():
            print(f"{indent}        body_flags={[int(bf[i]) for i in idx[:8]]}{'...' if len(idx)>8 else ''}")
        # first body pose
        if bq is not None and bq.shape[0] > idx[0]:
            q0 = bq[idx[0]]
            print(f"{indent}        first_body_q (id={int(idx[0])}): pos=({q0[0]:+.4f}, {q0[1]:+.4f}, {q0[2]:+.4f}) quat_xyzw=({q0[3]:+.4f}, {q0[4]:+.4f}, {q0[5]:+.4f}, {q0[6]:+.4f})")
        # last body pose
        if bq is not None and bq.shape[0] > idx[-1]:
            qN = bq[idx[-1]]
            print(f"{indent}        last_body_q  (id={int(idx[-1])}): pos=({qN[0]:+.4f}, {qN[1]:+.4f}, {qN[2]:+.4f}) quat_xyzw=({qN[3]:+.4f}, {qN[4]:+.4f}, {qN[5]:+.4f}, {qN[6]:+.4f})")
    else:
        print(f"{indent}bodies=0")

    if shape_ids:
        st = _array_np(getattr(model, "shape_type", None))
        ssc = _array_np(getattr(model, "shape_scale", None))
        sm_ke = _array_np(getattr(model, "shape_material_ke", None))
        sm_kd = _array_np(getattr(model, "shape_material_kd", None))
        sm_mu = _array_np(getattr(model, "shape_material_mu", None))
        sm_margin = _array_np(getattr(model, "shape_margin", None))
        sm_gap = _array_np(getattr(model, "shape_gap", None))
        sf = _array_np(getattr(model, "shape_flags", None))
        idx = np.asarray(shape_ids, dtype=np.int64)
        ss_max = int(idx.max()) if idx.size else -1
        print(f"{indent}shapes={len(shape_ids)}")
        if st is not None and st.shape[0] > ss_max:
            unique, counts = np.unique(st[idx], return_counts=True)
            print(f"{indent}        types={dict(zip(unique.tolist(), counts.tolist()))}")
        if ssc is not None and ssc.shape[0] > ss_max:
            print(f"{indent}        scale_x({_stats(ssc[idx, 0], precision=4)})")
            print(f"{indent}        scale_y({_stats(ssc[idx, 1], precision=4)})")
            print(f"{indent}        scale_z({_stats(ssc[idx, 2], precision=4)})")
        if sm_ke is not None and sm_ke.shape[0] > ss_max:
            print(f"{indent}        material_ke({_stats(sm_ke[idx], precision=4)})")
        if sm_kd is not None and sm_kd.shape[0] > ss_max:
            print(f"{indent}        material_kd({_stats(sm_kd[idx], precision=4)})")
        if sm_mu is not None and sm_mu.shape[0] > ss_max:
            print(f"{indent}        material_mu({_stats(sm_mu[idx], precision=4)})")
        if sm_margin is not None and sm_margin.shape[0] > ss_max:
            print(f"{indent}        margin({_stats(sm_margin[idx], precision=4)})")
        if sm_gap is not None and sm_gap.shape[0] > ss_max:
            print(f"{indent}        gap({_stats(sm_gap[idx], precision=4)})")
        if sf is not None and sf.shape[0] > ss_max:
            print(f"{indent}        shape_flags={[int(sf[i]) for i in idx[:8]]}{'...' if len(idx)>8 else ''}")
    else:
        print(f"{indent}shapes=0")


def dump_contact_pairs(label, model, groups):
    """Print contact pair count and inter-group breakdown."""
    pairs = _array_np(getattr(model, "shape_contact_pairs", None))
    if pairs is None:
        print(f"  {label}: contact_pairs=<unavailable>")
        return
    print(f"  {label}: contact_pairs total={pairs.shape[0]}")
    sets = {name: set(ids) for name, ids in groups.items()}
    counts = {}
    for raw_a, raw_b in pairs:
        a = int(raw_a)
        b = int(raw_b)
        pair = {a, b}
        for ga_name, ga in sets.items():
            for gb_name, gb in sets.items():
                if ga_name > gb_name:
                    continue
                key = f"{ga_name}_vs_{gb_name}"
                hit = False
                if ga_name == gb_name:
                    hit = a in ga and b in ga
                else:
                    hit = (a in ga and b in gb) or (a in gb and b in ga)
                if hit:
                    counts[key] = counts.get(key, 0) + 1
    for key, v in sorted(counts.items()):
        print(f"    {key} = {v}")


def dump_success():
    """Construct the success demo and dump model state."""
    print(SEP)
    print("SUCCESS DEMO STATE DUMP")
    print(SEP)

    # Insert newton repo path and import the example
    newton_repo = Path("/home/maximiliank/Work/newton")
    if str(newton_repo) not in sys.path:
        sys.path.insert(0, str(newton_repo))

    import newton  # noqa: F401
    import warp as wp
    import newton.examples
    from newton.examples.cable_robot.example_waterhose_scene2_insert_extract_success import Example

    # Build a null-viewer args namespace.
    from argparse import Namespace
    args = Namespace(
        device=None,
        viewer="null",
        rerun_address=None,
        output_path="output.usd",
        num_frames=1,
        headless=True,
        test=True,
        quiet=True,
        benchmark=False,
        warp_config=[],
        realtime=False,
        primary_view="mujoco",
        no_twoway=False,
        print_cable_poses=False,
        cable_pose_settle_seconds=0.0,
        print_robot_poses=False,
        broad_phase="explicit",
    )

    viewer = newton.viewer.ViewerNull(num_frames=1, benchmark=False, benchmark_timeout=None)
    ex = Example(viewer, args)

    print()
    print("=== MJC MODEL ===")
    mjc = ex.mujoco_model
    print(f"body_count={mjc.body_count} shape_count={mjc.shape_count} joint_count={mjc.joint_count}")
    dump_model_group("ALL MJC bodies", mjc, list(range(mjc.body_count)))

    print()
    print("=== VBD MODEL ===")
    vbd = ex.vbd_model
    print(f"body_count={vbd.body_count} shape_count={vbd.shape_count} joint_count={vbd.joint_count}")

    scene_body_ids = list(getattr(ex, "scene_body_ids", []) or [])
    cable_body_ids = list(getattr(ex, "cable_body_ids", []) or [])
    head_body_ids = list(getattr(ex, "cable_head_body_ids", []) or [])
    proxy_body_ids = list(getattr(ex, "proxy_body_ids", []) or [])
    # Anchor bodies for insert/pull pinning (mass=0, not in collision much)
    anchor_body_ids = []
    if getattr(ex, "_insert_anchor_body", None) is not None:
        anchor_body_ids.append(ex._insert_anchor_body)
    if getattr(ex, "_plug_anchor_body", None) is not None:
        anchor_body_ids.append(ex._plug_anchor_body)

    cable_only_ids = sorted(set(cable_body_ids) - set(head_body_ids))

    dump_model_group("scene (static, in VBD)", vbd, scene_body_ids)
    dump_model_group("cable capsules (in VBD)", vbd, cable_only_ids)
    dump_model_group("plug/head (in VBD)", vbd, head_body_ids)
    dump_model_group("proxy gripper duplicates (in VBD)", vbd, proxy_body_ids)
    dump_model_group("anchor bodies (in VBD)", vbd, anchor_body_ids)

    # Per-group shape ids using the VBD model.
    scene_shape_ids = _shape_ids_for_bodies(vbd, scene_body_ids)
    cable_shape_ids = _shape_ids_for_bodies(vbd, cable_only_ids)
    head_shape_ids = _shape_ids_for_bodies(vbd, head_body_ids)
    proxy_shape_ids = _shape_ids_for_bodies(vbd, proxy_body_ids)

    print()
    print("=== CONTACT PAIRS (VBD MODEL) ===")
    dump_contact_pairs(
        "vbd_model",
        vbd,
        {
            "cable": set(cable_shape_ids),
            "head": set(head_shape_ids),
            "proxy": set(proxy_shape_ids),
            "scene": set(scene_shape_ids),
        },
    )

    # Run one step to get contacts populated.
    print()
    print("=== AFTER 1 STEP ===")
    ex.step()
    contacts = ex.vbd_contacts
    n = int(_array_np(contacts.rigid_contact_count)[0])
    print(f"runtime rigid_contact_count={n}")
    if n > 0:
        s0 = _array_np(contacts.rigid_contact_shape0)[:n]
        s1 = _array_np(contacts.rigid_contact_shape1)[:n]
        # Distribution
        sets = dict(
            cable=set(cable_shape_ids),
            head=set(head_shape_ids),
            proxy=set(proxy_shape_ids),
            scene=set(scene_shape_ids),
        )
        bins = {}
        for a, b in zip(s0, s1):
            ai = int(a)
            bi = int(b)
            tags_a = [k for k, v in sets.items() if ai in v] or ["unknown"]
            tags_b = [k for k, v in sets.items() if bi in v] or ["unknown"]
            key = "_vs_".join(sorted([tags_a[0], tags_b[0]]))
            bins[key] = bins.get(key, 0) + 1
        for k, v in sorted(bins.items()):
            print(f"  runtime {k} = {v}")

    print()
    print("=== VBD SOLVER OPTIONS ===")
    s = ex.vbd_solver
    fields = [
        "iterations",
        "friction_epsilon",
        "rigid_contact_hard",
        "rigid_contact_history",
        "rigid_avbd_alpha",
        "rigid_avbd_gamma",
        "rigid_avbd_contact_alpha",
        "rigid_avbd_linear_beta",
        "rigid_avbd_angular_beta",
        "rigid_contact_alpha",
        "_coupling_has_rigid_avbd_state",
    ]
    for f in fields:
        v = getattr(s, f, "<missing>")
        try:
            v = float(v)
        except Exception:
            pass
        print(f"  {f} = {v}")

    print()
    print("=== HARVEST KERNEL ===")
    h = getattr(s, "coupling_harvest_proxy_wrenches", None)
    print(f"  bound: {getattr(h, '__qualname__', '<missing>') if h else '<none>'}")

    print()
    print("=== PROXY MAPPING SUMMARY ===")
    print(f"  mj_to_vbd_body_map (truncated): {dict(list(ex.mj_to_vbd_body_map.items())[:20])} ...")
    print(f"  proxy_body_ids (VBD): {ex.proxy_body_ids}")
    print(f"  proxy_mj_body_ids (MJC): {ex.proxy_mj_body_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["success", "isaaclab"], help="Which setup to dump.")
    args = parser.parse_args()
    if args.target == "success":
        dump_success()
    else:
        raise NotImplementedError("isaaclab target requires running inside isaaclab; see comparison wrapper script")
