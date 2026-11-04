# Genesis / Isaac Lab comparison

Genesis versus Isaac Lab's PhysX backend with Isaac Sim, using two shared tasks:
cartpole balance and ANYmal-C forward walking on a flat floor.
Each runs with proprioception, or proprioception plus 64×64 RGB. Cameras feed the
same CNN policy; this does not establish that vision is necessary to solve either task.

| Measurement | Included work |
| --- | --- |
| Startup | Runtime imports, simulator launch, asset import, scene construction, first reset/observation |
| First step | First control step, reported separately from warmed throughput |
| Physics | One 5 ms physics step; fixed targets, resets outside timed windows, rendering disabled |
| Rollout | Four physics steps, actions, observations, reward, termination/reset, optional RGB |
| Learning | Shared PPO, fixed transition budget, held-out evaluation, wall time and time to threshold |

These are matched task inputs, not identical numerical solvers or image quality.
PhysX uses TGS/implicit drives; Genesis uses its Newton solver/implicitfast integrator.
The default `native` solver profile uses Genesis's 25-iteration maximum with native
convergence tolerance and the maintained Isaac robot settings of 4 position / 0
velocity iterations. This matches practical settings, not numerical work or accuracy.
Use `--solver_profile fixed_cap --solver_iterations 50` for the separate diagnostic
profile; it is not the native task profile.
The native RGB renderers differ. Inspect camera samples and backend metadata before interpreting a ratio.
Cartpole uses USD clones in Isaac because native replication misplaced its fixed-base
anchors; the independent dynamics check covers every copy. ANYmal uses native replication.
RGB environments are spaced 100 m apart. The walking camera follows root translation;
cartpole uses a wider fixed view. Recorded displacement bounds check that other robot
copies stay outside the cameras' far planes. Native camera update costs remain timed.
Throughput is **batched environment steps/s**, not one robot's refresh rate.

## Source and environment

The initial source pins were current upstream heads on 2026-11-04:

| Source | Revision |
| --- | --- |
| Isaac Lab | `efa240ec39794f5d80430cbe38620626954c2c4d` |
| Genesis | `969c3bd1449d0f0c79effd7b6a72614ed46ba6e7` |

Place this archived harness in the pinned Isaac Lab checkout. Follow its maintained
[installation guide](../../../docs/source/isaaclab/installation.rst) for an isolated uv environment
with Isaac Sim/PhysX and RTX rendering. The latest checkout uses `uv run`; older checkouts
can use `./isaaclab.sh -p` when that wrapper exists. Do not install Genesis into the Isaac environment.

```bash
mkdir -p .benchmark_sources
git clone https://github.com/Genesis-Embodied-AI/Genesis.git .benchmark_sources/genesis
git -C .benchmark_sources/genesis checkout 969c3bd1449d0f0c79effd7b6a72614ed46ba6e7
uv venv --python 3.12 .benchmark_sources/genesis_env
```

Install a compatible CUDA PyTorch build into that environment using the pinned
[Genesis instructions](https://github.com/Genesis-Embodied-AI/Genesis/blob/969c3bd1449d0f0c79effd7b6a72614ed46ba6e7/README.md),
then `uv pip install --python .benchmark_sources/genesis_env/bin/python -e .benchmark_sources/genesis`.
Use the same Torch version/build in both environments. Retained package freezes describe the actual run.
Local `.pth` files that borrow an existing installation are a convenience, not the portable setup contract.

```bash
export ISAACLAB_PYTHON=/absolute/path/to/isaaclab/.venv/bin/python
export GENESIS_PYTHON="$PWD/.benchmark_sources/genesis_env/bin/python"
export PYGLET_HEADLESS=1 PYOPENGL_PLATFORM=egl ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
```

`prepare.py` generates identical URDF inputs and records original model/mesh hashes.
The bundled ANYmal model lacks bounds on twelve revolute joints: both inputs make
those joints explicitly continuous. Passive joint damping/friction and added armature
are zero; self-collision is disabled. ANYmal PD gains are 80/2 with an 80 N·m effort cap.

## Run and validate

From the Isaac Lab root, first run a small matrix, then validate each backend/task
configuration with `validate_adapter.py`. Use at least two environments for camera
isolation and subset-reset checks; four is sufficient. Validation runs are separate
from throughput measurements and must pass before publishing comparisons.

```bash
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/suite.py \
  --genesis_path .benchmark_sources/genesis --genesis_python "$GENESIS_PYTHON" \
  --isaaclab_python "$ISAACLAB_PYTHON" --output_dir benchmarks/simulator_smoke \
  --modes physics rollout --num_envs 4 --vision_num_envs 4 --seeds 0 --steps 32 --warmup_steps 8

uv run --no-project --python "$GENESIS_PYTHON" python scripts/benchmarks/simulator_comparison/validate_adapter.py \
  --config benchmarks/simulator_smoke/GENESIS_CASE/config.json --output benchmarks/validation/genesis.json
```

Replace `GENESIS_CASE` with the generated case directory; repeat with the Isaac interpreter
and an Isaac case. The validator checks cartpole against independent Lagrange/RK4 dynamics,
force direction, imported moving mass, canonical resets, ANYmal short stability, and actual batched RGB.
JSON records numerical errors and screenshots. A failed check exits nonzero.

```bash
uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/suite.py \
  --genesis_path .benchmark_sources/genesis --genesis_python "$GENESIS_PYTHON" \
  --isaaclab_python "$ISAACLAB_PYTHON" --output_dir benchmarks/simulator_full \
  --seeds 0 1 2 --steps 1000 --iterations 1000

uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/report.py \
  --input_dir benchmarks/simulator_full --output_dir benchmarks/simulator_report

uv run --no-project --python "$ISAACLAB_PYTHON" python scripts/benchmarks/simulator_comparison/export_figures.py \
  --input_dir benchmarks/simulator_full --output_dir benchmarks/simulator_report/figures
```

Use an idle GPU for publishable timings. Keep hardware, batch sizes, image settings,
seeds and budgets matched; report failures/OOMs rather than silently reducing one engine's batch.
Jobs run serially in shuffled order and resume completed cases; `--rerun` repeats them.
Each worker starts a new process; disk, JIT and shader caches are reused. This is not a
cold-machine startup measurement. Retain config, logs, package freezes, source/harness hashes,
asset manifest, validation output and checkpoints. Learning budgets that miss the threshold
remain censored; throughput alone does not establish learned task quality.

Publish the generated report directory under
`site/reports/YYYY-MM-DD-genesis-isaaclab/` in `maxkra15/maxkra15.github.io` using its existing
Pages workflow, with raw evidence alongside it. [REFERENCES.md](REFERENCES.md) audits the quoted benchmark claim.

## Another engine

Add an adapter module and interpreter selection to `suite.py`; shared task/PPO code stays unchanged.
Select another interpreter with `--backends mjlab --backend_python mjlab=/path/to/python` once its adapter exists.
An adapter owns `joint_names`, `metadata`, `reset(env_ids=None, joint_pos=None)`, `state()`,
`apply_action()`, `physics_step()`, `physics_only_step()`, `refresh_state()`, `render()`,
`synchronize()` and `close()`. `state()` returns ordered CUDA joint/root tensors in SI units,
world-frame root velocities and **wxyz** quaternions. `render()` returns CUDA uint8 `[N,H,W,3]`.
Run the physical validator before adding mjlab results. Report unsupported capabilities explicitly.
