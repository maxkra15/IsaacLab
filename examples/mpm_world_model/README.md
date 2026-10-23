# Coupled world-model prototype

This experimental training example learns short task-observation rollouts from
Newton MPM trajectories. PPO can pretrain or adapt a policy inside those rollouts,
then load the same policy into the real MPM task for evaluation and fine-tuning.
It does not replace Newton's material solver, predict a complete particle cloud,
or establish accurate contact forces.

The architecture takes inspiration from SuperMimic
([paper](https://arxiv.org/abs/2610.28480)). This implementation uses a recurrent
ensemble and ordinary RSL-RL PPO, rather than SuperMimic's analytical policy
gradients. It contains no copied proprietary browser-demo code or weights.

## Shared dynamics, explicit task adapters

The shared model contains masked particle and rigid-body set encoders and a GRU
transition core. Each task has its own observation, action and reward heads.
Material parameters, numerical resolution, timestep, geometry bounds and coupling
semantics enter an ordered context vector. A real initial snapshot also provides
controller state and material-history summaries. Every imagined transition after
that snapshot uses the model's own recurrent state and observations, with policy
actions; it cannot read future reference geometry or observations.

| Task | Actor / critic | Actions | Native coupling |
| --- | --- | --- | --- |
| Franka Pour | 84 / 86 | Existing 8 actions | Two-way staggered proxy coupling |
| Rizon Sharpa held pour | 121 / 121 | 6 bounded pose residuals through the existing IK | Measured rigid boundaries drive fluid; no fluid reaction on the robot |

Franka keeps its original observation groups and ordering, action clipping and
RSL checkpoint format, including the official iteration-7450 policy. Actor and
critic predictions share their identical observation prefix exactly. Known
previous-action and episode-clock updates are computed exactly. Large finite
material-history aggregates use a versioned signed-log encoding for numerical
conditioning; the original physical recordings remain unchanged.

The optional `--physical-decoder typed` uses task-declared physical fields. A
row-stochastic transport matrix updates mass fractions as `f_next = f @ P`,
preserving their sum and nonnegativity for admissible initial fractions. Local
rotation increments update unit quaternions, with a sign-invariant geodesic loss.
Seeds must already contain unit quaternions and a valid mass partition. Franka's
actor quaternion fields use XYZW; recorded entity quaternions and the teapot
actor use WXYZ. The decoder handles these orders explicitly and copies declared
constant, action-history and clock fields. The default `none` preserves the
earlier unconstrained architecture and its checkpoint behavior.

These constraints remove particular representation errors; they do not establish
correct material flow, contact, forces, volume preservation or general geometric
equivariance. Evaluation reports each ensemble member's violations separately
from the ensemble mean, because an arithmetic mean of unit quaternions need not
have unit norm.

The teapot adapter reuses the scripted contact-acquired grasp. After two seconds,
the policy adjusts desired translation and rotation through the original physical
IK controller; it does not teleport the pot. This is a new residual-control task,
not a learned full grasp/lift/pour policy. Its reward measures newly delivered
mass minus newly spilled mass and control effort. The learned environment derives
that reward from predicted mass fractions instead of the independent reward head.

Supporting a new task means implementing an adapter and collecting its physics
data. Context conditioning and a shared core provide an experiment for transfer;
they do not make a Franka-trained model a zero-shot simulator for liquid, new mesh
shapes or other constitutive laws. Mesh bounds do not describe arbitrary geometry.
Use the recorded mesh hashes to track which geometry was actually trained.

## Data and validity boundaries

Collectors save atomic episode shards with `T+1` states and `T` actions, rewards
and termination labels. Franka's recorder captures the physical endpoint before
automatic reset. A storage-budget cut is separate from a physical task timeout.
Reset-row or initial-cloud groups determine the train/validation/test split, so
action variants of the same reset cannot leak between splits. File and source
hashes accompany the manifest and checkpoints.

Coordinates are environment-local, Z-up; rigid quaternions use WXYZ. Particle
weights represent kilograms. Teapot stores a fixed spatial sample for the entity
encoder, but its mass partition, reward, finite checks and material aggregates
use the entire cloud. Its cavity classification is a documented mesh/ray
heuristic, not an exact fluid-volume integral.

Nonfinite states remain visible in raw recordings. Invalid transitions cannot
be training seeds or targets. Normalization uses training episodes only. PPO
seeds must belong to the model's declared training reset groups. Validation
selects checkpoints; final test groups are not used for selection.

Teapot also monitors rigid contact/constraint capacity at every native physics
step, including the grasp prefix. `finite_mask` records numerical validity;
`rigid_capacity_valid` records whether capacity remained valid. Their conjunction,
`valid_mask`, determines usable learning windows. A finite trajectory with dropped
constraints is not accepted as valid physics. Native iteration-limit flags are
reported separately from contact/constraint overflow.

Imagination is deliberately short. Independent ensemble disagreement, nonfinite
predictions and task-declared mass-admissibility checks end a rollout. Artificial
cuts bootstrap the value at the predicted endpoint, never the unrelated replay
snapshot used for reset. Model rewards are bounded to the observed training range.
These safeguards limit obvious exploitation; they do not establish fidelity.
Teapot's known 16-second control timeout also ends imagination before replay reset.
`train_policy --action-limit` can keep a new task's executed actions within its
collected range. The exported TorchScript `policy.pt` embeds the same clipping;
the raw RSL checkpoint does not. Franka requires the original limit of one to
preserve its existing native checkpoint contract.

## Run an experiment

Run from this repository with a validated Isaac Lab/Newton environment. The
commands below use `uv run`; a frozen dependency interpreter can instead be reused
with `uv run --no-project --python /path/to/python`, without changing that environment.
Use a fresh output directory for every run. Keep generated datasets/checkpoints
outside the repository.

Collect a reference dataset with the saved policy:

```bash
uv run python -m examples.mpm_world_model.collect_franka_pour \
  --checkpoint /path/to/model_7450.pt --reset-dataset /path/to/reset_dataset.pt \
  --output /results/franka-reference --num-envs 64 --steps 360 --warmup-steps 16 \
  --global-substeps 2 --media-substeps 2 --collision-decimation 0 \
  --perturb-std 0 --perturb-fraction 0 --headless
```

The explicit `2 / 2 / 0` settings reproduce the historical policy reference.
The collector's default `8 / 1 / 1` settings define different coupled dynamics;
changing them requires an independent physics and policy check. A smaller
timestep does not automatically preserve a trained controller's behavior.

Collect a second dataset with bounded action perturbations for coverage. Preserve
the same physics flags; use a different output and seed, and for example
`--perturb-std 0.15 --perturb-fraction 1`. Data only from deterministic expert
actions can encourage a model to memorize trajectory progress instead of learning
how actions change the outcome.

Fit and independently evaluate the dynamics:

```bash
uv run python -m examples.mpm_world_model.train_world_model \
  --data /results/franka-reference/episodes /results/franka-exploration/episodes \
  --output /results/franka-model --device cuda:0 --ensemble 2 \
  --steps 2000 --horizon 8 --burn-in 0 --batch-size 64 \
  --physical-decoder typed

uv run python -m examples.mpm_world_model.evaluate_world_model \
  --data /results/franka-reference/episodes /results/franka-exploration/episodes \
  --checkpoint /results/franka-model/best.pt --output /results/model-test.json \
  --split test --horizons 1 4 8 16 --device cuda:0
```

The evaluator reports raw and normalized rollout/endpoint errors, persistence and
shuffled-action controls, ensemble disagreement and physical-observable violations.
Low one-step error alone is insufficient. A shuffled-action result close to the
correct-action result is a warning about policy-training suitability.

Adapt the saved policy inside the model, then evaluate it in native MPM:

```bash
uv run python -m examples.mpm_world_model.train_policy \
  --data /results/franka-reference/episodes /results/franka-exploration/episodes \
  --model /results/franka-model/best.pt --output /results/model-ppo \
  --runner-config /results/franka-reference/runner_cfg.json \
  --checkpoint /path/to/model_7450.pt --task franka_pour \
  --device cuda:0 --num-envs 1024 --horizon 8 --iterations 20

uv run python -m examples.mpm_world_model.evaluate_franka_pour \
  --checkpoint /results/model-ppo/model_final.pt \
  --reset-dataset /path/to/reset_dataset.pt --output /results/native-evaluation \
  --num-envs 64 --steps 360 --warmup-steps 16 \
  --global-substeps 2 --media-substeps 2 --collision-decimation 0 --headless
```

For a native fine-tuning pilot, use `collect_franka_pour` with
`--train-only --fine-tune-iterations 20 --fine-tune-rollout-steps 16
--fine-tune-epochs 2 --fine-tune-learning-rate 0.00001` and the learned-policy
checkpoint. This runs actual MPM PPO and saves `model_mpm_finetuned.pt`. The
learning-rate override is applied after loading; the resolved rate is recorded.
Loop timing includes rollout collection and PPO updates and excludes process
startup and the final export. Evaluate that checkpoint in a fresh process.
Compare unchanged 7450, model-adapted, and MPM-fine-tuned policies using the same
physics, reset distribution and full episode horizon. Report numerical failures
separately from physical task failures.

## Teapot transfer experiment

```bash
uv run python -m examples.mpm_world_model.collect_teapot \
  --output /results/teapot-seed7 --fill-seed 7 --seed 0 \
  --episodes 6 --split-plan train train train train validation test \
  --steps 1600 --particle-samples 1024 --collider-basis pic356 \
  --mpm-iterations 256 --rigid-njmax 1024 --rigid-nconmax 512 \
  --headless -- --voxel_size 0.004 --grid_type sparse
```

Native demo arguments follow `--`. This pilot explicitly uses 4-mm voxels,
particle-based contact with a 356-point cap and 256 iterations; it does not claim
identical settings to the default demo. The larger rigid buffers must still pass
the per-step capacity monitor. The split plan rejection-samples safe
initial jitter before actions or outcomes, using the recorder's unchanged hash
buckets. Repeated episodes restore physical, material, actuator and controller
state without rebuilding the scene. These held-out resets share geometry,
material and topology: this tests local generalization, not unseen objects.
Broader transfer requires additional fill levels, geometries and material data.

Train on both tasks to test a shared model, or pass `--initialize-from` to transfer
the Franka shared core into a teapot model. Compare transfer and training from
scratch with identical teapot data and optimization budgets. `--freeze-core`
provides a stricter adapter-only transfer control. Each task keeps its own policy.

The teapot policy task ID is `rizon_sharpa_teapot_held_pour`; `train_policy` creates
a small new actor when no checkpoint/config is supplied and exports `policy.pt`.
Replay it through `collect_teapot --policy-checkpoint /results/teapot-ppo/policy.pt`.
For a narrow-action pilot, set the new actor's exploration standard deviation
through its runner JSON and use the same action limit in imagined training and
native replay. Reducing collection amplitude alone does not constrain a policy's
exploration or make out-of-distribution predictions reliable.
Measure delivery, spills, grasp retention and full-horizon finite behavior in real
MPM before claiming the policy learned to pour. The collector reports maximum
sampled palm drift relative to the pot; success additionally requires drift
below a provisional 0.1-m threshold. This geometry proxy tolerates compliance
and does not prove a particular contact force or wrench.

## Remaining scope

This is an experimental policy-training pipeline. It has no full particle decoder
and cannot produce physically meaningful imagined particle videos. It does not
learn calibrated forces or implement SuperMimic's BPTT policy optimizer. A stronger
general material model would need richer geometry, local interactions and broader
material/contact data. The typed decoder supplies only a few necessary output
constraints. Learned
warm-starts inside the native solver are a separate path when retaining MPM's
physical constraints is more important than eliminating simulation cost.

Run CPU contract tests with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=examples uv run python -m pytest examples/mpm_world_model/tests -q
```

CPU analytical dynamics tests verify learning and interfaces. They are not
evidence of material accuracy, real-MPM transfer or GPU throughput.
