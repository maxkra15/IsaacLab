# Sources and comparison rules

Sources checked on 2026-11-04. External results below are references, not measurements made by this suite.

## What the “6×” claim measures

[GPUSimBench, Zhang et al., July 2026](https://arxiv.org/html/2607.13059v1), Table III, reports
11.124 million Franka environment-steps/s for Genesis at 131,072 environments and 1.851 million
for Isaac Lab at 32,768 environments on an RTX 5070. Their ratio is approximately 6.0, computed
from the table. These are different batch sizes at each simulator’s maximum tested capacity.
The paper used Genesis 0.3.11 and Isaac Lab 2.2.1. Rendering and learning were excluded.
Its free-falling cube result favors Isaac Lab instead. Neither result establishes policy-training speed.

The [published benchmark implementation](https://github.com/zhanghuzhenyu/gpu_sim_bench/tree/7d39254e6642bef924166ac74e7c6940c713a471)
adds further qualifications:

- [Genesis Franka](https://github.com/zhanghuzhenyu/gpu_sim_bench/blob/7d39254e6642bef924166ac74e7c6940c713a471/Genesis/exp3/benchmark_genesis.py)
  sends joint position targets and calls `scene.step()` on a fixed-base Franka scene.
- [Its scene](https://github.com/zhanghuzhenyu/gpu_sim_bench/blob/7d39254e6642bef924166ac74e7c6940c713a471/Genesis/exp3/scene_unified_single_franka_fixed.xml)
  explicitly specifies no ground or objects and includes an external `panda.xml`.
- [Isaac Lab Franka](https://github.com/zhanghuzhenyu/gpu_sim_bench/blob/7d39254e6642bef924166ac74e7c6940c713a471/IsaacLab/exp3/isaaclab_franka_random_benchmark.py)
  defaults to `Isaac-Lift-Cube-Franka-v0` and calls `env.step()`, including task processing and the task’s
  configured physics decimation. It does not set the paper’s stated 0.01 s physics step explicitly.

These observable code differences prevent treating the public scripts as a matched pure-physics
comparison. They do not prove which code produced the paper’s table; the paper’s exact executable
revision is not identified by those files.

## Rules used for new measurements

| Measurement | Required interpretation |
| --- | --- |
| Physics throughput | Environment × physics steps / elapsed time; identical physics timestep, substeps, assets, contacts and controllers where supported. Synchronize before and after timing. |
| Environment throughput | Include actions, observations, rewards and resets. State control decimation separately; one policy step can contain several physics steps. |
| Scaling | Compare the same environment counts first. Maximum capacity is a separate result, with failures and memory limits retained. |
| Startup | Measure a fresh process through first usable state; record imports, scene build, first step and compilation separately. Identify cache state; a fresh process does not imply an empty disk cache. |
| Vision | Match resolution, channels, frame rate, camera count and transfer destination. Prove frames are produced. Report renderer and any device-to-host transfer. |
| Learning | Match task definition, observations, actions, reward, termination, policy, optimizer and evaluation. Show quality against both transitions and wall time across fixed seeds. |
| Reliability | Retain failed runs, exact source revisions, commands, dependencies, hardware, seeds and raw timing samples. Label unsupported combinations explicitly. |

Equal numerical solver settings do not guarantee equal physical accuracy. Asset or renderer differences
must accompany results, and a shorter timed rollout alone must not be called faster learning.

The main `native` profile follows the pinned Genesis `RigidOptions` maximum of 25 iterations
and the pinned Isaac ANYmal/cartpole articulation settings of TGS 4 position / 0 velocity iterations.
Genesis exits early on convergence; its precision-dependent tolerance is recorded after scene construction.
These are backend-specific practical profiles, not an equality of solver work or numerical accuracy.

The shared trainer uses clipped [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
with generalized advantage estimation. It is one fixed recipe for both engines; it is not either project's
individually tuned training recipe. A target missed within the fixed budget is recorded as censored.

## Upstream projects and publication

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis): upstream source and examples.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): upstream source and task definitions.
- [mjlab](https://github.com/mujocolab/mjlab): potential future backend; not measured by this report.
- [Existing report host](https://github.com/maxkra15/maxkra15.github.io): `site/reports/YYYY-MM-DD-topic/`
  holds standalone HTML, measurements, figures and provenance. Its existing Pages workflow publishes
  `site/` after merging to `main`; add a homepage card without changing older report URLs.
