# Waterhose Teleop Scaling Report

NVIDIA | Isaac Lab internal technical report

Date: 2026-05-29

Branch: `waterhose-demo`
Commit: `d54af20f1`
Task: `Isaac-Waterhose-Robot-Demo-v0`
Mode: built-in teleop path, idle SpaceMouse command broadcast to all environments
GPU: `0, NVIDIA GeForce RTX 5090, 32607 MiB`

## Executive Summary

The current default waterhose task does not scale like a batched Isaac Lab environment. It creates independent split Newton runtimes per environment. In teleop mode, aggregate env-step throughput stays nearly flat at about 73 env steps/s from 2 to 32 envs, while live control rate falls almost exactly inversely with `num_envs`.

For live teleop, 1 env is the only comfortable operating point on this machine: 67.0 Hz. Two envs is borderline at 36.4 Hz. By 8 envs, teleop drops to 9.1 Hz. 32 envs completes, but at only 2.3 Hz.

The 64-env run did not finish within the 300 s timeout. Larger requested points through 8192 envs were not run after that timeout because the measured setup slope is roughly 4.15 s per additional environment at the larger measured points, and peak GPU memory was already about 14.9 GiB before the 64-env run completed.

## Methodology

Each point launched:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Robot-Demo-v0 \
  --mode teleop --teleop_device spacemouse \
  --vis none --num_envs N --max_steps 60 --profile --device cuda:0
```

The SpaceMouse device was initialized normally. The benchmark used idle input, so the same zero/idle teleop command was broadcast to all environments. This measures the teleop control path and environment stepping throughput, not human reaction quality or rendering latency.

Metrics:

- Live Hz: manager steps per wall-clock second, equivalent to teleop control update rate.
- Env steps/s: `live Hz * num_envs`, useful for aggregate simulation throughput.
- Setup time: time before rollout starts.
- GPU delta: peak `nvidia-smi` memory above the pre-run baseline.

## Scaling Table

| num_envs | status | setup (s) | rollout (s) | live Hz | env steps/s | step time (ms) | GPU delta (GiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | completed | 5.5 | 0.90 | 67.0 | 67.0 | 14.9 | 0.98 |
| 2 | completed | 8.8 | 1.65 | 36.4 | 72.8 | 27.5 | 1.23 |
| 4 | completed | 15.3 | 3.26 | 18.4 | 73.6 | 54.3 | 1.71 |
| 8 | completed | 28.9 | 6.58 | 9.1 | 72.8 | 109.9 | 2.69 |
| 16 | completed | 58.2 | 12.98 | 4.6 | 73.6 | 217.4 | 4.64 |
| 32 | completed | 127.8 | 25.79 | 2.3 | 73.6 | 434.8 | 8.53 |
| 64 | timed out at 300s | - | - | - | - | - | 14.87 |
| 128 | not run after 64-env timeout | - | - | - | - | - | - |
| 256 | not run after 64-env timeout | - | - | - | - | - | - |
| 512 | not run after 64-env timeout | - | - | - | - | - | - |
| 1,024 | not run after 64-env timeout | - | - | - | - | - | - |
| 2,048 | not run after 64-env timeout | - | - | - | - | - | - |
| 4,096 | not run after 64-env timeout | - | - | - | - | - | - |
| 8,192 | not run after 64-env timeout | - | - | - | - | - | - |

## Interpretation

The measured scaling is serial-runtime scaling: total work increases almost linearly with `num_envs`, but the implementation does not recover that cost through batching. This is why env steps/s stays flat instead of rising, and why live Hz degrades from 67.0 Hz at 1 env to 2.3 Hz at 32 envs.

Recommended operating points:

| Use case | Recommended num_envs | Reason |
| --- | ---: | --- |
| Live SpaceMouse / client demo | 1 | Highest control rate and shortest startup |
| Side-by-side visual smoke | 2 | Still usable for comparison, but below 40 Hz |
| Offline robustness smoke | 4-8 | Aggregate throughput is flat, useful only to expose multi-env bugs |
| Data collection / training | Not this architecture | Needs a true batched Newton model or a different collection strategy |

## Caveat

This report profiles the stable default one-way task, not the experimental coupled-manager task. The result is dominated by the current N-independent-runtime architecture, not by SpaceMouse polling.
