# Coupled world-model prototype

This experimental training example learns short task-observation rollouts from
Newton MPM trajectories. PPO can pretrain or adapt a policy inside those rollouts,
then load the same policy into the real MPM task for evaluation and fine-tuning.
It does not replace Newton's material solver, predict a complete particle cloud,
or establish accurate contact forces.

During learned PPO, the network predicts the next policy observations; it does
not run particle-to-grid transfer, a constitutive/contact solve or grid-to-particle
transfer. During data collection, native evaluation and native fine-tuning,
Newton still performs real MPM. Videos from the capture helpers show that native
replay, not particles imagined by the observation model.

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

The default `2 / 2 / 0` settings reproduce the historical policy reference.
Using `8 / 1 / 1` instead defines different coupled dynamics;
changing these settings requires an independent physics and policy check. A smaller
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

For an illustrative native video, install the optional MoviePy/FFmpeg recording
dependencies in a compatible rendering environment and set `CUDA_VISIBLE_DEVICES`
to one physical GPU UUID. `--egl-device-index` is the host EGL ordinal, independent
of the masked logical CUDA index:

```bash
uv run python -m examples.mpm_world_model.capture_franka_pour \
  --checkpoint /results/model-ppo/model_final.pt \
  --reset-dataset /path/to/reset_dataset.pt --output /results/franka-video \
  --egl-device-index 0 --steps 360 --warmup-steps 16
```

This records one world at 1x playback with physical particles. `capture.json`
retains every frame's policy clock, reset row and episode-reset boundaries.
These clips illustrate the saved policy; use the separate batched evaluations
for success statistics. The native task starts from its reset-dataset states,
which can already be partway through a skill.

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
Add `--video-path /results/teapot-video.mp4 --video-fps 25` to record the full
physical grasp/pour loop. Its sidecar labels scripted versus learned residual
control and retains physical outcomes even when the task fails. For explicit
headless EGL selection, set `PYGLET_HEADLESS_DEVICE` to the host device ordinal
before launching; `CUDA_VISIBLE_DEVICES` separately selects physics.
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

## Measured Franka pilot, 2026-10-24

The initial unconstrained-model experiment used the official 7450 policy,
47,104 real transitions and two recurrent ensemble members. Its model fit took
528 seconds on an L40. PPO used at most four imagined steps before resetting to
a training replay seed; these are not independent complete physical episodes.
The starting policy was already strong and received only 20 learned PPO
iterations at learning rate 1e-5. Preserving its success rate demonstrates
interface compatibility and limited adaptation, not learning pouring from
scratch inside the model.

| Actual MPM replay | First completed successes | Aggregate successes | Recorded nonfinite material-state events |
| --- | --- | --- | --- |
| Unchanged official 7450 | 63 / 64 | 563 / 569 | 1 |
| After learned-model PPO | 62 / 64 | 484 / 490 | 0 |
| Then real-MPM PPO fine-tuning | 64 / 64 | 538 / 545 | 0 |

The replays share the historical coupling settings, seed 42 and the fixed reset
dataset, with 16 warmup and 360 evaluation actions. Initial-cohort metrics include
warmup; aggregate metrics exclude warmup completions. Subsequent resets diverge
with policy timing, so aggregate episodes are not paired. Fifty initial reset
rows belong to model-training groups; this is a policy-interface and stability
check, not a held-out generalization score or evidence of improved success.

| Recorded PPO workload | Transitions | Loop seconds | Transitions / second |
| --- | --- | --- | --- |
| Native MPM, 64 environments | 20,480 | 50.177 | 408 |
| Learned model, 64 environments | 20,480 | 6.132 | 3,340 |
| Learned model, 8,192 environments | 2,621,440 | 6.229 | 420,873 |

Both 64-environment runs use 20 iterations, 16 rollout steps, two epochs, four
minibatches and learning rate 1e-5. Their recorded loop times differ by 8.18x;
the starting checkpoints differ and native timing can include initial graph
capture and automatic saves. Data generation, fitting and process startup are
excluded. The 8,192-world result is a separate batch-capacity measurement, not
an equal-batch MPM speedup. This short pilot does not establish an end-to-end
training speedup after amortizing the model's fitting cost.

The optional typed model fit took 831 seconds and subsequently reached 464,666
PPO transitions per second at 8,192 worlds, with no nonfinite predictions or
mass-admissibility cuts. These model checks alone do not validate physical flow
or policy success.

The teapot pilot selected four intact monitored episodes (6,400 transitions)
from two training reset groups, one validation group and one test group. The
failed sparse-grid-capacity episode remains in the source record and is excluded;
selection does not filter by task success. With equal 800-update budgets, core
transfer changes best validation loss from 0.56215 to 0.56108, a marginal 0.2%.
Both models are worse than static persistence on aggregate normalized material
error. Shuffling actions changes material error by only about 0.2-1.7% at four
or eight steps. These results do not establish reliable control-dependent fluid
dynamics or broad transfer across objects and materials.

A fresh scratch-model teapot actor completed 6.55 million imagined transitions
in 28 seconds, with no nonfinite predictions or mass-admissibility cuts. Its
imagined reward did not improve. Executed actions were restricted to 0.1, giving
at most 2-mm translation and 0.02-radian orientation residuals around the existing
motion. This is fast training-loop execution, not evidence that the full teapot
skill has been learned. Native PPO fine-tuning for this new task still needs its
own vector-environment adapter; the existing Franka native fine-tuning path is
already exercised.

Full-horizon native teapot videos were recorded for the nominal script and both
new residual actors. Every replay uses actual MPM particles, includes the
two-second physical grasp and sixteen-second pour, remains finite at recorded
endpoints, and passes contact-capacity checks at all 14,400 native steps. Each
retains its grasp, with maximum palm translation drift relative to the pot below
2.3 cm.

| Native teapot replay | Delivered mass | Spilled mass | Reached provisional 50% delivery goal |
| --- | --- | --- | --- |
| Nominal script, zero residual | 15.13% | 0% | No |
| Actor trained in scratch world model | 10.20% | 0% | No |
| Actor trained in transferred world model | 10.96% | 0% | No |

These three episodes have different initial-cloud reset groups. They illustrate
valid native replay and unsuccessful task completion, not a paired causal
comparison of policies. The clips are not evidence that the learned model can
simulate particles or that transfer improves control.

Run CPU contract tests with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=examples uv run python -m pytest examples/mpm_world_model/tests -q
```

CPU analytical dynamics tests verify learning and interfaces. They are not
evidence of material accuracy, real-MPM transfer or GPU throughput.
