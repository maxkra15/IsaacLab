"""Clean path: env.reset() lands at curriculum stage 0 (cup loaded, opening-up over target). Drive a tilt
and check the cup tilts + media is delivered to the target."""
import argparse

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import torch

    from isaaclab_tasks.contrib.franka_scoop.scoop_env import FrankaScoopEnv
    from isaaclab_tasks.contrib.franka_scoop.scoop_env_cfg import FrankaScoopEnvCfg

    cfg = FrankaScoopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args_cli.device)
    cfg.ik_backend = "newton"
    # test the source-side opening-up loaded pose (reachable side) instead of the +y target (joint-limit railed)
    cfg.curriculum_reset_pose = ("home_up", "home_up", "home_up", "pile", "pile")
    with launch_simulation(cfg, args_cli):
        env = FrankaScoopEnv(cfg)
        env.reset()  # full manager reset -> curriculum stage 0 (loaded, opening-up over target)
        dev = env.device
        bp = [round(float(v), 3) for v in env.bowl_pos_e()[0]]
        print(f"[DUMP] reset: bowl_e={bp} in_bowl={int(env.count_in_bowl()[0])} in_target={int(env.count_in_target()[0])} "
              f"pitch={float(env._pitch[0]):+.2f}", flush=True)
        # introspect the action term after one step with full +x action
        term = env.action_manager.get_term("scoop")
        env.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))
        jt0 = term._joint_targets.clone()
        env.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))
        print(f"[DUMP] action term: raw={[round(float(v),2) for v in term._raw[0]]} "
              f"proc={[round(float(v),3) for v in term._proc[0]]} "
              f"target_bowl_e={[round(float(v),3) for v in env._target_bowl_e[0]]} "
              f"joint_targets_changed={float((term._joint_targets - jt0).abs().max()):.5f} "
              f"arm_q={[round(float(v),2) for v in env.arm_joint_q()[0]]}", flush=True)
        print(f"[DUMP] arm limits lo={[round(float(v),2) for v in env._arm_lo[0]]} hi={[round(float(v),2) for v in env._arm_hi[0]]}", flush=True)
        # localize: does ANY action move the cup at this stage-0 pose? test +x, +z, tilt separately
        for tag, act in (("move+x", [1.0, 0, 0, 0]), ("move+z", [0, 0, 1.0, 0]), ("tilt+", [0, 0, 0, 1.0])):
            env.reset()
            b0 = [round(float(v), 3) for v in env.bowl_pos_e()[0]]
            for _ in range(40):
                env.step(torch.tensor([act], device=dev))
            b1 = [round(float(v), 3) for v in env.bowl_pos_e()[0]]
            _, bq = env._bowl_pose_w()
            up = float(1.0 - 2.0 * (bq[0, 0] ** 2 + bq[0, 1] ** 2))
            print(f"[DUMP] {tag}: bowl_e {b0} -> {b1}  pitch={float(env._pitch[0]):+.2f} opening_up={up:+.2f} "
                  f"in_target={int(env.count_in_target()[0])}", flush=True)
        env.close()


main()
