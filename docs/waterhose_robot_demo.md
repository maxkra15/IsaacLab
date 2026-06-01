# Waterhose Robot Demo

Last verified: 2026-05-31.

The scene-authored waterhose task lives in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose/
```

The scene is authored with regular Isaac Lab assets:

- `ArticulationCfg` for the RBY1DF robot USD.
- `RigidObjectCfg` for the plugs and kinematic cable anchors.
- `CableObjectCfg` for the two USD cable curves.
- `AssetBaseCfg` for the fridge and static scene assets.
- `CoupledNewtonCfg` with `CoupledAdmmSolverCfg` or `CoupledProxySolverCfg` for MJWarp robot + VBD cable/plug coupling.

The older task-local Newton builder under `waterhose_robot_demo` is still present for the existing demo and Newton visualizer flows. Kit/USD rendering work should prefer the scene-authored task because it renders the authored robot, fridge, plug, and cable USD assets directly.

## Task IDs

Canonical task IDs:

| Task ID | Purpose |
| --- | --- |
| `Isaac-Waterhose-v0` | Manager/RL task with joint-position actions. |
| `Isaac-Waterhose-Proxy-v0` | Manager/RL task with Newton proxy coupling. |
| `Isaac-Waterhose-RBY1-IK-Abs-v0` | Scripted demo task with absolute IK action space. |

Existing robot-demo task IDs:

| Task ID | Current target |
| --- | --- |
| `Isaac-Waterhose-Robot-Demo-v0` | Stable task-local one-way demo; single-env on upstream Newton PR 2848. |
| `Isaac-Waterhose-Robot-Demo-Coupled-v0` | Alias to the scene-authored ADMM task (`WaterhoseEnvCfg`). |
| `Isaac-Waterhose-Robot-Demo-Proxy-Coupled-v0` | Alias to the scene-authored proxy task (`WaterhoseProxyEnvCfg`). |

## Visuals

Use Kit for the proper USD-authored view:

```bash
./isaaclab.sh -p scripts/environments/state_machine/waterhose_rby1_ik.py \
  --task Isaac-Waterhose-RBY1-IK-Abs-v0 \
  --num_envs 1 \
  --visualizer kit
```

The legacy runner still supports both visualizer families:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis kit
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis newton
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --vis none --max_steps 200
```

Newton visualizer support is intentionally kept. The Kit path should render authored USD assets, not synthetic Newton-derived display meshes.

## Assets

By default the task looks for assets next to `waterhose_env_cfg.py`:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose/assets/
```

You can override this with:

```bash
WATERHOSE_ASSETS_DIR=/path/to/waterhose/assets
```

Expected files include:

```text
fridge/fridge.usda
fridge/cable/cable001.usda
fridge/cable/cable002.usda
fridge/cable/plug.usda
rby1df/rby1df.usda
```

## Training

The standard training entry point works with the canonical task:

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Waterhose-v0 \
  --num_envs 1 \
  --max_iterations 1000 \
  --video --video_length 20
```

For multi-env coupled smoke tests through the demo runner:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Robot-Demo-Proxy-Coupled-v0 \
  --num_envs 4 \
  --vis none \
  --max_steps 100 \
  --profile
```

## Notes

The old synthetic Kit display authored Newton-derived meshes under `/World/WaterhoseDemo/Dynamic`; this is the path that made Kit visuals look unlike the real assets. If Kit visuals look wrong, inspect the USD assets, `CableObjectCfg` registration, or Fabric curve sync rather than falling back to synthetic display meshes.
