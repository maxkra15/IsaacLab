"""Per-component GPU profile of the waterhose coupled demo (IK / VBD / MJWarp / collision / coupling)."""
import argparse
import collections
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Waterhose-Coupled-v0")
parser.add_argument("--warmup", type=int, default=150)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
# Disable the CUDA graph so each kernel launches individually and can be timed by name.
env_cfg.sim.physics.use_cuda_graph = False
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

from isaaclab_tasks.contrib.waterhose.scripted_state_machine import create_scripted_policy  # noqa: E402

sm = create_scripted_policy(env, settle_time=0.5, debug=False)

for _ in range(args.warmup):
    env.step(sm.compute(env))
wp.synchronize()

t0 = time.perf_counter()
wp.timing_begin(synchronize=True)
for _ in range(args.steps):
    env.step(sm.compute(env))
results = wp.timing_end(synchronize=True)
t1 = time.perf_counter()

sps = args.steps / (t1 - t0)
print(f"\n>>> PROFILE: {args.steps} steps, graph OFF, wall steps/s = {sps:.1f}")
if results:
    r0 = results[0]
    print(">>> TimingResult attrs:", [a for a in dir(r0) if not a.startswith("_")])


def get_name(r):
    for a in ("name", "func_name", "kernel"):
        v = getattr(r, a, None)
        if isinstance(v, str):
            return v
    return str(r)


def get_ms(r):
    for a in ("elapsed", "elapsed_ms", "duration", "time"):
        v = getattr(r, a, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


by_name = collections.defaultdict(lambda: [0.0, 0])
for r in results:
    by_name[get_name(r)][0] += get_ms(r)
    by_name[get_name(r)][1] += 1
total = sum(v[0] for v in by_name.values()) or 1.0


def categorize(name):
    n = name.lower()
    if "memset" in n or "memcpy" in n:
        return "Overhead (memset/memcpy)"
    if any(k in n for k in ("proxy", "harvest", "scatter", "sync_proxy", "mirror")):
        return "Coupling (proxy)"
    if any(k in n for k in ("_lm_solve", "_pos_jac", "_rot_jac", "_pos_residual", "_rot_residual",
                            "compute_costs", "update_gradient", "cholesky", "jtcj", "_ik", "jacobian")):
        return "Newton IK"
    if any(k in n for k in ("narrow_phase", "broad_phase", "broadphase", "mesh_triangle", "mesh_mesh",
                            "mesh_query", "shape_aabb", "buffered_contacts", "reduced_contacts",
                            "reducer", "sdf", "bvh")):
        return "Collision (broad/narrow phase)"
    if any(k in n for k in ("solve_rigid_body", "update_duals", "accumulate_body_body",
                            "initialize_body_forces", "cosserat", "stretch", "bend", "vbd",
                            "particle", "contact_k")):
        return "VBD / AVBD solve"
    if any(k in n for k in ("_crb", "_cfrc", "_subtree", "articulation_fk", "kinematics", "_qfrc",
                            "mj_", "mjwarp", "mujoco", "rne", "_com_acc", "implicit", "_qpos", "_qvel")):
        return "MJWarp (rigid robot)"
    return "Other"


by_cat = collections.defaultdict(float)
for name, (ms, _) in by_name.items():
    by_cat[categorize(name)] += ms

print(f"\n>>> TOTAL kernel GPU time = {total:.1f} ms / {args.steps} steps = {total / args.steps:.3f} ms/step\n")
print("=== GPU time by COMPONENT ===")
for c, ms in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"  {c:26s} {ms:9.1f} ms  {100 * ms / total:5.1f}%  ({ms / args.steps:.3f} ms/step)")

print("\n=== TOP 30 kernels by total GPU time ===")
for name, (ms, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:30]:
    print(f"  {ms:8.1f} ms {100 * ms / total:5.1f}%  x{cnt // args.steps:4d}/step  [{categorize(name)[:10]:10s}] {name[:64]}")

env.close()
app.app.close()
