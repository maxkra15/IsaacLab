# Genesis / Isaac Lab MJWarp comparison

Genesis versus **Isaac Lab with Newton/MuJoCo Warp**, using shared cartpole balance,
ANYmal-C forward walking and Franka reaching tasks. The primary `isaaclab` adapter runs without Kit.
Each task uses proprioception, or proprioception plus 64×64 RGB in the same CNN
policy. This measures vision processing cost; these tasks do not establish vision-only navigation.

| Measurement | Included work |
| --- | --- |
| Startup | Worker clock: runtime imports through first usable observation; excludes Python/site initialization |
| First step | First control step, separately from warmed throughput |
| Physics | One 5 ms physics step; fixed targets, resets outside timed windows, rendering disabled |
| Rollout | Four physics steps, actions, observations, reward, termination/reset, optional RGB |
| Learning | Shared PPO, fixed transitions, held-out evaluation, training time and first observed target |

## Shared benchmark baseline

The default `shared` solver profile uses the following explicit settings:

| Setting | Genesis | Isaac Lab MJWarp |
| --- | --- | --- |
| Solver / integrator | Newton / approximate implicitfast | Newton / implicitfast |
| Iteration limits | 25; native convergence tolerance | 100 solver, 50 line search; tolerance 1e-6 |
| Contacts / friction | Genesis contacts; coefficient 0.8 | MuJoCo contacts; pyramidal cone; impratio 1; coefficient 0.8 |
| Substeps / execution | 1 / native GPU execution | 1 / CUDA graphs |
| Capacity | Native allocation | njmax 300; nconmax native allocation |
| RGB renderer | Madrona rasterizer | Newton Warp ray tracer; textures/ambient light enabled; shadows disabled |

Iteration counts and rendered image quality are engine specific. This baseline does
not establish equal numerical accuracy. Stock ANYmal MJWarp training presets use
two substeps, Newton contacts and elliptic friction; they are a different configuration.
`--solver_profile fixed_cap --solver_iterations 50` selects a separate diagnostic profile.

Both engines receive the same URDF, masses, controller, reward, observation and reset
code. Passive joint damping/friction and added armature are zero; self-collision is
disabled. ANYmal uses PD gains 80/2 and an 80 N·m effort limit. Missing joint bounds in
the bundled ANYmal model are explicitly represented as continuous joints in both inputs.
Franka uses a fixed-base seven-joint Panda, fixed open fingers, PD gains 80/4 and
authored effort limits (87 N·m for joints 1–4, 12 N·m for joints 5–7). Seeded FK poses
near home define reachable targets. The 5 s evaluation target requires mean tool-position
error ≤0.05 m and survival ≥0.95; survival means valid joint state within limits plus
a 0.1 rad margin. It does not establish grasping or contact manipulation.

RGB environments are separated by 100 m. ANYmal cameras follow root translation;
cartpole uses a wider fixed view. Native camera-update costs remain timed, including
Genesis's host-side follow update. Displacement bounds check camera isolation.
MJWarp graph activity and fatal capacity flags are checked at the final solver state.
Convergence-limit flags are recorded separately and do not reject a run. Final-state
flags do not establish that no transient overflow occurred earlier.
Throughput means **batched environment steps/s**, not one robot's refresh rate.

## Source and environments

Sources were selected from upstream heads on 2026-11-04; Newton follows Isaac Lab's pin:

| Source | Revision |
| --- | --- |
| Isaac Lab | `efa240ec39794f5d80430cbe38620626954c2c4d` |
| Genesis | `969c3bd1449d0f0c79effd7b6a72614ed46ba6e7` |
| Newton | `42f7def118c2c4299ce493457bafd4e8a22a696e` |

Place the harness in the pinned Isaac Lab checkout. Its
[installation guide](../../../docs/source/isaaclab/installation.rst) documents the
kitless environment. The `importers` extra selects the isolated asset importer
6.2.0.1; the simulator and Newton Warp renderer do not launch Kit.

```bash
mkdir -p .benchmark_sources
git clone https://github.com/newton-physics/newton.git .benchmark_sources/newton
git -C .benchmark_sources/newton checkout 42f7def118c2c4299ce493457bafd4e8a22a696e
UV_PROJECT_ENVIRONMENT="$PWD/.benchmark_sources/mjwarp_env" uv sync --extra importers

git clone https://github.com/Genesis-Embodied-AI/Genesis.git .benchmark_sources/genesis
git -C .benchmark_sources/genesis checkout 969c3bd1449d0f0c79effd7b6a72614ed46ba6e7
uv venv --python 3.12 .benchmark_sources/genesis_env
export ISAACLAB_PYTHON="$PWD/.benchmark_sources/mjwarp_env/bin/python"
export GENESIS_PYTHON="$PWD/.benchmark_sources/genesis_env/bin/python"
uv pip install --python "$GENESIS_PYTHON" torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$GENESIS_PYTHON" -e .benchmark_sources/genesis
export PYGLET_HEADLESS=1 PYOPENGL_PLATFORM=egl ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
```

Use the same Torch/CUDA build and training flags in both environments. Engine-specific
packages differ; retained package freezes record actual versions. Local `.pth` files
that borrow an existing environment are not the portable setup contract.

## Run and validate

Run a small matrix first. Repeat the independent validator for both backends,
all three tasks and both vision modes, with at least two environments (four is sufficient):

```bash
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/suite.py \
  --genesis_path .benchmark_sources/genesis --newton_path .benchmark_sources/newton \
  --genesis_python "$GENESIS_PYTHON" --isaaclab_python "$ISAACLAB_PYTHON" \
  --output_dir benchmarks/simulator_smoke --solver_profile shared \
  --modes physics rollout --num_envs 4 --vision_num_envs 4 --seeds 0 --steps 32 --warmup_steps 8

uv run --no-project --python "$GENESIS_PYTHON" python scripts/benchmarks/simulator_comparison/validate_adapter.py \
  --config benchmarks/simulator_smoke/GENESIS_CASE/config.json --output benchmarks/validation/genesis.json
```

Replace `GENESIS_CASE` with the generated directory; use the Isaac interpreter for
Isaac cases. The validator checks independent cartpole Lagrange/RK4 dynamics, force
direction, imported masses, canonical/subset resets, ANYmal/Franka short stability,
independent URDF-to-tool Franka kinematics and
batched RGB isolation. Preserve its JSON and screenshots; failed checks exit nonzero.

The complete matrix contains 126 runs across three seeds (42 per task). Collect the
84 ANYmal/cartpole cases with the original baseline source:

```bash
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/suite.py \
  --genesis_path .benchmark_sources/genesis --newton_path .benchmark_sources/newton \
  --genesis_python "$GENESIS_PYTHON" --isaaclab_python "$ISAACLAB_PYTHON" \
  --output_dir benchmarks/simulator_full --solver_profile shared --tasks anymal cartpole \
  --num_envs 256 4096 --vision_num_envs 64 --learning_num_envs 512 --vision_learning_num_envs 512 \
  --seeds 0 1 2 --steps 500 --iterations 500 --rollout_steps 24 --evaluation_interval 50 --evaluation_steps 250

uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/report.py \
  --input_dir benchmarks/simulator_full --output_dir benchmarks/simulator_report
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/export_figures.py \
  --input_dir benchmarks/simulator_full --output_dir benchmarks/simulator_report/figures
```

Repeat that suite command at the Franka source revision with `--tasks franka` and
`--output_dir benchmarks/simulator_franka` for its 42 cases. For publication, use an
empty output directory and the committed publication-helper revision:

```bash
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/assemble_report.py \
  --cohort_manifest benchmarks/cohorts.launch.json --output_dir benchmarks/simulator_publication \
  --harness_dir scripts/benchmarks/simulator_comparison --source_commit "$(git rev-parse HEAD)" \
  --include_run_archive
```

For the measured two-stage study, keep the original ANYmal/cartpole source and 84 runs
intact; collect Franka's 42 cases with `--tasks franka` using its separately committed
runtime. A launch manifest declares both sources and local evidence directories:

```json
{"cohorts": [
  {"name": "baseline", "tasks": ["anymal", "cartpole"], "source_commit": "FULL_BASELINE_SHA",
   "input_dir": "simulator_full", "validation_dir": "validation_baseline"},
  {"name": "franka", "tasks": ["franka"], "source_commit": "FULL_FRANKA_SHA",
   "input_dir": "simulator_franka", "validation_dir": "validation_franka"}
]}
```

Manifest paths resolve relative to that file. `--source_commit` identifies the current publication helpers.
Each cohort requires all matched runs and four independent validators per task using
its exact runtime, validator source and physical inputs. Backends share one physical GPU
within each cohort; cohorts may use separate L40s on the same recorded CPU host.
Use a separate output directory for each RGB validator so its frames remain intact. Old/failed attempts
are archived separately. The bundle includes source plus license, CSV, figures, validation
frames, provenance and checksums; the optional run archive includes policies and logs.
The exported `data/cohorts.json` uses relative record/validation/archive paths and can be
passed back to the same assembler to rebuild from the downloaded bundle. Each measured
cohort has its own source ZIP; `source.zip` contains the exact publication helpers.

Use an idle GPU. Jobs run serially in shuffled order and resume completed cases;
`--rerun` repeats them. Workers start fresh processes while disk/JIT caches are reused.
Retain configs, logs, freezes, source/harness hashes, asset manifests and checkpoints.
The first observed target uses the training clock, including evaluation and excluding
startup; checks every 50 iterations limit its time resolution. Missed targets remain
censored. Target-time medians include reached seeds only. Complete process wall time
is recorded separately. Final task quality accompanies speed.

Publish the complete generated directory under `site/reports/YYYY-MM-DD-genesis-isaaclab/`
in `maxkra15/maxkra15.github.io`. [REFERENCES.md](REFERENCES.md) explains why this MJWarp
comparison is a different backend/version comparison from the older RTX 5070 headline.

PhysX is an optional separate reference for Cartpole and ANYmal: explicitly select
`--backends genesis isaaclab_physx --tasks cartpole anymal`
and `--isaaclab_physx_python /path/to/physx/bin/python`. Its default interpreter path is
`.benchmark_sources/isaaclab_env/bin/python`; those results are excluded from the main MJWarp report.

## Another engine

Add an adapter module and interpreter selection to `suite.py`; shared task/PPO code stays unchanged.
Use `--backends mjlab --backend_python mjlab=/path/to/python` once its adapter exists.
An adapter owns `joint_names`, `metadata`, `reset(env_ids=None, joint_pos=None)`, `state()`,
`apply_action()`, `physics_step()`, `physics_only_step()`, `refresh_state()`, `render()`,
`synchronize()` and `close()`. `state()` returns ordered CUDA joint/root tensors in SI units,
world-frame root velocities and **wxyz** quaternions. `render()` returns CUDA uint8 `[N,H,W,3]`.
Franka additionally requires `ee_pos` with shape `[N,3]`: environment-local world grasp-center coordinates [m].
Run the physical validator before including another engine, and report unsupported capabilities explicitly.
