# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instrumented headless runner for the waterhose coupled-instability investigation.

Runs the scripted RBY1 grasp with NO visualizers and logs, every manager step,
the max |velocity| and max |position| of each Newton body group (cable segments,
plug, anchor, robot, gripper fingers) plus a NaN/explosion watchdog. Prints a
phase-annotated trace and a final summary identifying the first unstable step and
which body group blew up first.

Supports --exp PRESET to patch the env cfg before building, so we can isolate the
instability cause one factor at a time (see EXPERIMENTS dict).

Usage:
  ./isaaclab.sh -p scripts/environments/waterhose/instrument_waterhose.py \
      --task Isaac-Waterhose-Coupled-v0 --num_envs 1 --max_steps 900 \
      --exp baseline --csv /tmp/wh_baseline.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ.setdefault("ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS", "0")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Instrumented waterhose grasp runner.")
parser.add_argument("--task", type=str, default="Isaac-Waterhose-Coupled-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--settle_time", type=float, default=0.1)
parser.add_argument("--exp", type=str, default="baseline", help="Experiment preset name (see EXPERIMENTS).")
parser.add_argument("--csv", type=str, default="", help="Optional CSV output path for per-step metrics.")
parser.add_argument("--explode_vel", type=float, default=50.0, help="Speed [m/s or rad/s] flagged as unstable.")
parser.add_argument("--stop_on_explode", action="store_true", help="Stop a few steps after first explosion.")
parser.add_argument("--debug_script", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless, no visualizers.
args_cli.headless = True
args_cli.headless_explicit = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.contrib.waterhose.scripted_state_machine import WaterhoseDemoState  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_newton.physics import NewtonManager  # noqa: E402


# --------------------------------------------------------------------------------------
# Experiment presets: each takes the env cfg and mutates it in place before env build.
# --------------------------------------------------------------------------------------
def _solver_cfg(env_cfg):
    return env_cfg.sim.physics.solver_cfg


def _vbd_entry_cfg(env_cfg):
    for e in _solver_cfg(env_cfg).entries:
        if e.name == "vbd":
            return e.solver_cfg
    raise RuntimeError("no vbd entry")


def _proxy_cfg(env_cfg):
    return _solver_cfg(env_cfg).proxy_coupling.proxies[0]


def exp_baseline(env_cfg):
    pass


def exp_no_coupling(env_cfg):
    # Remove the proxy entirely: gripper no longer present in the VBD solve.
    _solver_cfg(env_cfg).proxy_coupling.proxies = []


def exp_more_substeps(env_cfg):
    env_cfg.sim.physics.num_substeps = 40


def exp_soft_proxy(env_cfg):
    p = _proxy_cfg(env_cfg)
    p.shape_material_ke = 2.0e4
    p.shape_material_kd = 1.0e1


def exp_proxy_damping(env_cfg):
    p = _proxy_cfg(env_cfg)
    p.shape_material_kd = 5.0e2


def exp_more_vbd_iter(env_cfg):
    _vbd_entry_cfg(env_cfg).iterations = 40


def exp_light_cable(env_cfg):
    env_cfg.scene.cable1.spawn.physics_material.density = 1500.0


def exp_lagged(env_cfg):
    _proxy_cfg(env_cfg).mode = "lagged"


def _gripper_action(env_cfg):
    return env_cfg.actions.gripper_action


def exp_tight_grasp(env_cfg):
    # Close to a ~13 mm gap (was ~17 mm) -> ~0.8 mm interference per side on the
    # 14.6 mm flange, so the kinematic fingers actually clamp the plug.
    g = _gripper_action(env_cfg)
    g.close_command_expr = {
        "right_gripper_finger_joint_1": 0.013,
        "right_gripper_left_finger_joint": -0.0065,
        "right_gripper_right_finger_joint": 0.0065,
    }


def exp_tight_grasp_14(env_cfg):
    g = _gripper_action(env_cfg)
    g.close_command_expr = {
        "right_gripper_finger_joint_1": 0.014,
        "right_gripper_left_finger_joint": -0.007,
        "right_gripper_right_finger_joint": 0.007,
    }


def exp_tight_grasp_12(env_cfg):
    g = _gripper_action(env_cfg)
    g.close_command_expr = {
        "right_gripper_finger_joint_1": 0.012,
        "right_gripper_left_finger_joint": -0.006,
        "right_gripper_right_finger_joint": 0.006,
    }


def exp_insert_firm(env_cfg):
    # Firm grasp (11mm clamp + higher proxy friction) + softer cable axial stiffness, to test
    # whether cable tension during the insertion push is what makes the plug slip / not insert.
    g = _gripper_action(env_cfg)
    g.close_command_expr = {
        "right_gripper_finger_joint_1": 0.011,
        "right_gripper_left_finger_joint": -0.0055,
        "right_gripper_right_finger_joint": 0.0055,
    }
    _proxy_cfg(env_cfg).shape_material_mu = 3.0
    env_cfg.scene.cable1.spawn.physics_material.stretch_stiffness = 1.0e6


def exp_soft_cable_only(env_cfg):
    env_cfg.scene.cable1.spawn.physics_material.stretch_stiffness = 1.0e6


def exp_firm_grip(env_cfg):
    # Firm the clamp + raise friction so the centered plug can't rotate in the grip.
    g = _gripper_action(env_cfg)
    g.close_command_expr = {
        "right_gripper_finger_joint_1": 0.011,
        "right_gripper_left_finger_joint": -0.0055,
        "right_gripper_right_finger_joint": 0.0055,
    }
    p = _proxy_cfg(env_cfg)
    p.shape_material_mu = 3.0
    mc = env_cfg.sim.physics.model_cfg
    mc.soft_contact_mu = 1.0
    mc.shape_material_mu = 2.0


def exp_heavy_proxy(env_cfg):
    # Make the gripper proxy bodies ~immovable in the VBD view (inv_mass -> ~0),
    # i.e. true one-way coupling: the proxy drives the cable but is not pushed back.
    _proxy_cfg(env_cfg).mass_scale = 1.0e6


def exp_heavy_proxy_soft(env_cfg):
    p = _proxy_cfg(env_cfg)
    p.mass_scale = 1.0e6
    p.shape_material_ke = 5.0e4
    p.shape_material_kd = 1.0e1


def exp_proxy_immovable_1e3(env_cfg):
    _proxy_cfg(env_cfg).mass_scale = 1.0e3


def exp_no_plug_weld(env_cfg):
    # Drop the head weld (plug -> cable seg 0). Keep only the tail anchor weld.
    atts = env_cfg.scene.cable1.attachments
    env_cfg.scene.cable1.attachments = [a for a in atts if a.cable_anchor != 0]


def exp_softer_plug_weld(env_cfg):
    v = _vbd_entry_cfg(env_cfg)
    v.rigid_joint_linear_ke = 1.0e4
    v.rigid_joint_angular_ke = 1.0e3


def exp_heavier_plug(env_cfg):
    # 1g -> 50g plug; raises the lightest contact body's mass to damp stiff-contact accel.
    env_cfg.scene.plug1.spawn.mass_props = __import__(
        "isaaclab.sim", fromlist=["MassPropertiesCfg"]
    ).MassPropertiesCfg(mass=0.05)


def exp_resample_cable(env_cfg):
    # Force ~uniform 6mm segments (removes the 44mm head segment discontinuity).
    env_cfg.scene.cable1.resample_segment_length = 0.006


def exp_lower_stretch(env_cfg):
    env_cfg.scene.cable1.spawn.physics_material.stretch_stiffness = 1.0e6


def _admm(env_cfg):
    return _solver_cfg(env_cfg).admm_coupling


def exp_admm_more_iter(env_cfg):
    _admm(env_cfg).iterations = 20


def exp_admm_low_rho(env_cfg):
    _admm(env_cfg).rho = 5.0


def exp_admm_high_rho(env_cfg):
    _admm(env_cfg).rho = 500.0


def exp_admm_more_substeps(env_cfg):
    env_cfg.sim.physics.num_substeps = 40


def exp_admm_combo(env_cfg):
    # candidate stabilization: more substeps + more admm iters + lower rho
    env_cfg.sim.physics.num_substeps = 30
    a = _admm(env_cfg)
    a.iterations = 12
    a.rho = 10.0


EXPERIMENTS = {
    "baseline": exp_baseline,
    "no_coupling": exp_no_coupling,
    "more_substeps": exp_more_substeps,
    "soft_proxy": exp_soft_proxy,
    "proxy_damping": exp_proxy_damping,
    "more_vbd_iter": exp_more_vbd_iter,
    "light_cable": exp_light_cable,
    "lagged": exp_lagged,
    "tight_grasp": exp_tight_grasp,
    "tight_grasp_14": exp_tight_grasp_14,
    "tight_grasp_12": exp_tight_grasp_12,
    "insert_firm": exp_insert_firm,
    "soft_cable_only": exp_soft_cable_only,
    "firm_grip": exp_firm_grip,
    "heavy_proxy": exp_heavy_proxy,
    "heavy_proxy_soft": exp_heavy_proxy_soft,
    "proxy_immovable_1e3": exp_proxy_immovable_1e3,
    "no_plug_weld": exp_no_plug_weld,
    "softer_plug_weld": exp_softer_plug_weld,
    "heavier_plug": exp_heavier_plug,
    "resample_cable": exp_resample_cable,
    "lower_stretch": exp_lower_stretch,
    "admm_more_iter": exp_admm_more_iter,
    "admm_low_rho": exp_admm_low_rho,
    "admm_high_rho": exp_admm_high_rho,
    "admm_more_substeps": exp_admm_more_substeps,
    "admm_combo": exp_admm_combo,
}


# --------------------------------------------------------------------------------------
# Body-group classification from Newton model labels.
# --------------------------------------------------------------------------------------
def classify_bodies(model):
    """Return dict group_name -> np.array of body indices, plus the label list."""
    labels = list(model.body_label)
    groups = {"cable": [], "plug": [], "anchor": [], "gripper": [], "robot": [], "other": []}
    for i, lab in enumerate(labels):
        low = lab.lower()
        if "/cable" in low:
            groups["cable"].append(i)
        elif "plug" in low:
            groups["plug"].append(i)
        elif "anchor" in low:
            groups["anchor"].append(i)
        elif "gripper" in low:
            groups["gripper"].append(i)
            groups["robot"].append(i)
        elif "robot" in low:
            groups["robot"].append(i)
        else:
            groups["other"].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}, labels


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Kill all visualizers / video for a clean headless physics run.
    env_cfg.sim.visualizer_cfgs = []

    patch = EXPERIMENTS.get(args_cli.exp)
    if patch is None:
        raise SystemExit(f"unknown --exp {args_cli.exp}; choices={list(EXPERIMENTS)}")
    patch(env_cfg)
    print(f"[instr] exp={args_cli.exp} num_envs={args_cli.num_envs} dt={env_cfg.sim.dt} "
          f"substeps={env_cfg.sim.physics.num_substeps}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    model = NewtonManager._model
    groups, labels = classify_bodies(model)
    body_world = np.asarray(model.body_world.numpy()) if hasattr(model.body_world, "numpy") else np.asarray(model.body_world)
    n_bodies = len(labels)
    print(f"[instr] n_bodies={n_bodies} group_sizes={{ {', '.join(f'{k}:{len(v)}' for k,v in groups.items())} }}", flush=True)
    # restrict to env 0 bodies (world 0 or -1) so multi-env doesn't pollute group stats
    env0_mask = (body_world == 0) | (body_world == -1)

    def _find_idx(substr):
        for i, lab in enumerate(labels):
            if substr.lower() in lab.lower() and env0_mask[i]:
                return i
        return -1

    plug_idx = _find_idx("Plug1")
    grip_idx = _find_idx("right_gripper_base")
    lf_idx = _find_idx("right_gripper_leftfinger")
    rf_idx = _find_idx("right_gripper_rightfinger")
    print(f"[instr] plug_idx={plug_idx} grip_idx={grip_idx} lf_idx={lf_idx} rf_idx={rf_idx}", flush=True)

    # One-time dump of collision geometry for the right fingers + plug (shape extents + transform
    # relative to their body), to understand the finger gripping surface and the plug flange.
    def _dump_geom():
        m = model
        try:
            sbody = m.shape_body.numpy()
            stype = m.shape_type.numpy() if hasattr(m.shape_type, "numpy") else np.asarray(m.shape_type)
            sxf = m.shape_transform.numpy()
            sscale = m.shape_scale.numpy() if hasattr(m, "shape_scale") and m.shape_scale is not None else None
        except Exception as e:
            print(f"[geom] could not read shapes: {e}", flush=True)
            return
        # mesh source list (vertices) for mesh shapes
        msrc = getattr(m, "shape_source", None)
        def mesh_bounds(s):
            try:
                src = msrc[int(s)] if msrc is not None else None
                if src is None:
                    return None
                V = getattr(src, "vertices", None)
                if V is None:
                    V = getattr(src, "points", None)
                if V is None:
                    return None
                V = np.asarray(V)
                return V.min(0), V.max(0)
            except Exception:
                return None
        for name, bidx in [("leftfinger", lf_idx), ("rightfinger", rf_idx), ("plug", plug_idx)]:
            sh = np.where(sbody == bidx)[0]
            print(f"[geom] {name} (body {bidx}): {len(sh)} shapes", flush=True)
            for s in sh[:6]:
                t = sxf[s]
                mb = mesh_bounds(s)
                if mb is not None:
                    lo, hi = mb
                    print(f"[geom]   shape{int(s)} type={int(stype[s])} shapeT_pos={np.round(t[:3],4).tolist()} "
                          f"mesh_min={np.round(lo,4).tolist()} mesh_max={np.round(hi,4).tolist()} "
                          f"size_mm={np.round((hi-lo)*1000,1).tolist()}", flush=True)
                else:
                    print(f"[geom]   shape{int(s)} type={int(stype[s])} shapeT_pos={np.round(t[:3],4).tolist()} (no mesh src)", flush=True)
    _dump_geom()

    sm = WaterhoseDemoState(env.num_envs, env.step_dt, env.device, args_cli.settle_time, args_cli.debug_script)
    actions = sm.compute(env)

    csv_f = open(args_cli.csv, "w") if args_cli.csv else None
    if csv_f:
        cols = ["step", "phase", "grip"]
        for g in groups:
            cols += [f"{g}_vmax", f"{g}_pmax"]
        cols += ["nan_bodies"]
        # Plug pose (7) + plug spatial velocity (6) + gripper-base pose (7) for grasp-quality analysis.
        cols += ["plug_px", "plug_py", "plug_pz", "plug_qw", "plug_qx", "plug_qy", "plug_qz"]
        cols += ["plug_wd0", "plug_wd1", "plug_wd2", "plug_wd3", "plug_wd4", "plug_wd5"]
        cols += ["grip_px", "grip_py", "grip_pz", "grip_qw", "grip_qx", "grip_qy", "grip_qz"]
        cols += ["lf_px", "lf_py", "lf_pz", "rf_px", "rf_py", "rf_pz"]  # right gripper fingers
        csv_f.write(",".join(cols) + "\n")

    first_explode_step = -1
    first_explode_group = ""
    first_nan_step = -1
    last_phase = -1
    steps_after_explode = 0

    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:
        _, _, terminated, truncated, _ = env.step(actions)

        state = NewtonManager._state_0
        body_qd = state.body_qd.numpy()  # [nbody, 6]
        body_q = state.body_q.numpy()    # [nbody, 7]
        speeds = np.linalg.norm(body_qd, axis=1)
        finite = np.isfinite(body_qd).all(axis=1) & np.isfinite(body_q).all(axis=1)
        nan_idx = np.where(~finite & env0_mask)[0]
        # Global (all-env) watchdog: catch instability in ANY env, not just env 0.
        nan_idx_all = np.where(~finite)[0]
        sp_finite = speeds[np.isfinite(speeds)]
        global_vmax = float(sp_finite.max()) if sp_finite.size else float("nan")

        phase = int(sm.phase[0].item())
        grip = float(actions[0, -1].item()) if actions.numel() else 0.0

        row = {"step": step, "phase": WaterhoseDemoState.PHASE_NAMES[phase], "grip": grip}
        worst_v = 0.0
        worst_g = ""
        for g, idx in groups.items():
            idx0 = idx[np.isin(idx, np.where(env0_mask)[0])] if idx.size else idx
            if idx0.size:
                sp = speeds[idx0]
                sp_f = sp[np.isfinite(sp)]
                vmax = float(sp_f.max()) if sp_f.size else float("nan")
                pos = body_q[idx0, :3]
                pos_f = np.linalg.norm(pos[np.isfinite(pos).all(axis=1)], axis=1)
                pmax = float(pos_f.max()) if pos_f.size else float("nan")
            else:
                vmax = pmax = float("nan")
            row[f"{g}_vmax"] = vmax
            row[f"{g}_pmax"] = pmax
            if np.isfinite(vmax) and vmax > worst_v:
                worst_v = vmax
                worst_g = g

        if csv_f:
            vals = [str(row["step"]), row["phase"], f"{row['grip']:.3f}"]
            for g in groups:
                vals += [f"{row[f'{g}_vmax']:.4g}", f"{row[f'{g}_pmax']:.4g}"]
            vals.append(str(len(nan_idx)))
            pq = body_q[plug_idx] if plug_idx >= 0 else np.zeros(7)
            pqd = body_qd[plug_idx] if plug_idx >= 0 else np.zeros(6)
            gq = body_q[grip_idx] if grip_idx >= 0 else np.zeros(7)
            vals += [f"{v:.6g}" for v in pq.tolist()]
            vals += [f"{v:.5g}" for v in pqd.tolist()]
            vals += [f"{v:.6g}" for v in gq.tolist()]
            lf = body_q[lf_idx, :3] if lf_idx >= 0 else np.zeros(3)
            rf = body_q[rf_idx, :3] if rf_idx >= 0 else np.zeros(3)
            vals += [f"{v:.6g}" for v in lf.tolist()]
            vals += [f"{v:.6g}" for v in rf.tolist()]
            csv_f.write(",".join(vals) + "\n")

        if phase != last_phase:
            print(f"[phase] step={step} -> {row['phase']} grip={grip:.2f}", flush=True)
            last_phase = phase

        # explosion watchdog (global across all envs)
        if first_explode_step < 0 and global_vmax > args_cli.explode_vel:
            first_explode_step = step
            bad = int(np.nanargmax(np.where(np.isfinite(speeds), speeds, -1)))
            first_explode_group = worst_g
            print(f"[EXPLODE] step={step} phase={row['phase']} global_vmax={global_vmax:.3g} "
                  f"worst_body={labels[bad]} (idx={bad}, world={int(body_world[bad])}) grip={grip:.2f}", flush=True)
            snap = " ".join(f"{g}={row[f'{g}_vmax']:.3g}" for g in groups)
            print(f"[EXPLODE] env0 vmax_by_group: {snap}", flush=True)
        if first_nan_step < 0 and len(nan_idx_all) > 0:
            first_nan_step = step
            bad_groups = {}
            for g, idx in groups.items():
                inter = np.intersect1d(idx, nan_idx_all)
                if inter.size:
                    bad_groups[g] = inter.size
            worlds = sorted(set(int(body_world[i]) for i in nan_idx_all))
            print(f"[NAN] step={step} phase={row['phase']} n_nan={len(nan_idx_all)} worlds={worlds} groups={bad_groups} "
                  f"first_labels={[labels[i] for i in nan_idx_all[:5]]}", flush=True)

        if first_explode_step >= 0 or first_nan_step >= 0:
            steps_after_explode += 1
            if args_cli.stop_on_explode and steps_after_explode > 15:
                print("[instr] stopping shortly after explosion", flush=True)
                break

        if bool(torch.any(terminated | truncated).item()):
            print(f"[instr] episode terminated/truncated at step {step}", flush=True)
            sm.reset((terminated | truncated).nonzero(as_tuple=False).squeeze(-1))

        actions = sm.compute(env)
        step += 1

    print("=" * 70, flush=True)
    print(f"[SUMMARY] exp={args_cli.exp} steps_run={step}", flush=True)
    print(f"[SUMMARY] first_explode_step={first_explode_step} group={first_explode_group}", flush=True)
    print(f"[SUMMARY] first_nan_step={first_nan_step}", flush=True)
    if first_explode_step < 0 and first_nan_step < 0:
        print("[SUMMARY] STABLE (no explosion, no NaN within run)", flush=True)
    else:
        print("[SUMMARY] UNSTABLE", flush=True)
    print("=" * 70, flush=True)

    if csv_f:
        csv_f.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
