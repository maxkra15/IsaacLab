# Franka Pour Scene-Owned Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Newton Franka Pour task to current Isaac Lab and make its visible cups standard scene-owned rigid assets with isolated multi-world MPM, outer CUDA graph capture, correct reset state, and correct Kit/Newton visualization.

**Architecture:** Merge current `upstream/develop` into the isolated integration branch, retaining upstream ClonePlan/Fabric/USD behavior and only the feature branch's unique coupled/multi-world contracts. Author the two visible bowls as task-local visual USD mesh spawners used by `RigidObjectCfg`; add the particle-only hollow colliders to those scene-owned bodies in the narrow Newton hook and keep the remaining invisible solver proxies there. Resolve all task state through public scene asset APIs and select scene-owned bodies through `SceneEntityCfg`.

**Tech Stack:** Python 3.12, Isaac Lab ManagerBasedRLEnv/InteractiveScene, OpenUSD, Newton, Warp, PyTorch, pytest, pre-commit.

---

### Task 1: Integrate the current upstream baseline

**Files:**
- Merge: `upstream/develop` into `max/franka-pour-scene-assets`
- Resolve: `source/isaaclab_newton/isaaclab_newton/cloner/replicate.py`
- Resolve: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- Resolve: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py`
- Resolve: `source/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py`
- Resolve: `source/isaaclab_newton/isaaclab_newton/physics/mpm_manager_cfg.py`
- Resolve: `source/isaaclab_newton/isaaclab_newton/assets/mpm_object/`
- Resolve: `source/isaaclab_newton/isaaclab_newton/ik/`
- Resolve: `source/isaaclab_visualizers/isaaclab_visualizers/newton/`
- Resolve: package `pyproject.toml` and stub exports reported by the merge

- [ ] **Step 1: Record the integration boundary**

Run:

```bash
git status --short
git rev-parse HEAD upstream/develop
git rev-list --left-right --count HEAD...upstream/develop
```

Expected: clean worktree and the feature/upstream divergence is recorded before the merge.

- [ ] **Step 2: Merge without rewriting the shared branch history**

Run:

```bash
git merge --no-ff upstream/develop
```

Expected: the merge stops on the known MPM/IK/Newton conflicts; `contrib/franka_pour` applies without textual conflicts.

- [ ] **Step 3: Resolve upstream-owned infrastructure from upstream**

Use upstream implementations for ClonePlan/Fabric bindings, Newton/USD asset replication, Kit scene partitioning, Newton visualizer lifecycle, official Newton IK, official MPM demos, dependency pins, and package versions. Remove the feature branch's legacy `_sync_transforms_to_usd_xform_ops` and `_usd_xform_ops` paths.

The resolved asset constructors must contain the guarded dual replication behavior:

```python
if has_kit():
    queue_usd_replication(cfg)
queue_newton_physics_replication(cfg)
```

The resolved Kit visualizer must leave per-environment RTX scene partitioning disabled unless `ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1` is explicitly set.

- [ ] **Step 4: Reapply only the unique feature contracts**

Retain these APIs and behaviors on top of upstream:

```python
MPMSolverCfg.separate_worlds
NewtonManager.register_builder_world_hook
NewtonManager.unregister_builder_world_hook
NewtonManager.reset_solver_state
MPMObject.particle_offsets
CoupledSolverCfg
CoupledSolverEntryCfg
CoupledProxyCfg
ProxyCouplingCfg
```

Retain outer-capture preparation/status checks, canonical state swapping, sparse total-capacity handling, per-world hook invocation, coupled solver exports, and nested solver custom-attribute registration. Use upstream `rename_builder_labels()` and Fabric binding outputs rather than restoring the old clone loop.

- [ ] **Step 5: Remove semantically duplicated pre-upstream files**

Use upstream `isaaclab.utils.warp.particle_mesh.ParticleMeshCounter`, update any feature-branch imports, and remove the duplicate `isaaclab_newton/utils/particle_mesh.py`, its duplicate test, and changelog fragments whose features are now in upstream. Keep changelog fragments for coupled solving, multi-world capture, and Franka Pour.

- [ ] **Step 6: Run import and focused infrastructure tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/cloner/test_rename_builder_labels.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  source/isaaclab_newton/test/assets/test_mpm_object.py \
  source/isaaclab_visualizers/test/test_newton_adapter.py -q
```

Expected: collection succeeds and all selected tests pass.

- [ ] **Step 7: Commit the upstream integration**

Run file-scoped pre-commit on every manually resolved file, then:

```bash
git add -A
git commit
```

Expected: the pending merge commit is created with both parents and no conflict markers remain.

### Task 2: Add a rigid cube-bowl USD spawner

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/cube_bowl_spawner_cfg.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/cube_bowl_spawner.py`
- Create: `source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_spawner.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/__init__.py`

- [ ] **Step 1: Write a failing stage-authoring test**

The test launches a stage, calls the spawner at `/World/Cup`, and verifies:

```python
root = stage.GetPrimAtPath("/World/Cup")
mesh = stage.GetPrimAtPath("/World/Cup/geometry/mesh")
assert root.HasAPI(UsdPhysics.RigidBodyAPI)
assert root.HasAPI(UsdPhysics.MassAPI)
assert mesh.IsA(UsdGeom.Mesh)
assert not mesh.HasAPI(UsdPhysics.CollisionAPI)
assert len(UsdGeom.Mesh(mesh).GetPointsAttr().Get()) == 16
```

For a source config with a grasp proxy, assert `/World/Cup/geometry/grasp_proxy` has
`CollisionAPI`, is invisible, and shares the root rigid body. Add a second case for a kinematic
target with no grasp proxy and verify the authored display color and physics material binding.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_spawner.py -q
```

Expected: collection fails because `CubeBowlSpawnerCfg` and `spawn_cube_bowl` do not exist.

- [ ] **Step 3: Implement the config and spawner**

Define a `RigidObjectSpawnerCfg` subclass with fixed tuple types and task geometry:

```python
@configclass
class CubeBowlSpawnerCfg(RigidObjectSpawnerCfg):
    func: Callable | str = "{DIR}.cube_bowl_spawner:spawn_cube_bowl"
    inner_width: float = MISSING
    inner_depth: float = MISSING
    cavity_depth: float = MISSING
    wall_thickness: float = MISSING
    bottom_thickness: float = MISSING
    display_color: tuple[float, float, float] = (0.95, 0.82, 0.16)
    physics_material_path: str = "material"
    physics_material: RigidBodyMaterialCfg | None = None
```

Implement `spawn_cube_bowl()` with `@clone`, `make_cube_bowl_mesh()`, a visual-only `UsdGeom.Mesh`
with exact triangle topology, an optional invisible `UsdGeom.Cube` grasp collider, normal rigid/mass
schemas, and normal visual/physics material binding. Reject an existing prim and return the root prim.

- [ ] **Step 4: Run the spawner and existing mesh tests to verify GREEN**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_spawner.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py -q
```

Expected: both files pass.

- [ ] **Step 5: Commit the spawner**

Run file-scoped pre-commit, stage the four files, and commit:

```bash
git commit -m "feat: Add rigid cube-bowl scene spawner"
```

### Task 3: Make derived task configuration deterministic and scene-owned

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/cup_media.py`
- Modify: `source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests with these behavioral assertions:

```python
original = FrankaPourEnvCfg()
original.scene.num_envs = 8
resolved = original.finalize()

assert original.scene.source_cup is None
assert original.scene.target_cup is None
assert original.scene.media is None
assert isinstance(resolved.scene.source_cup, RigidObjectCfg)
assert isinstance(resolved.scene.target_cup, RigidObjectCfg)
assert isinstance(resolved.scene.media, MPMObjectCfg)
assert _mpm_solver_cfg(resolved).max_active_cell_count == 8 * resolved.mpm_min_cells_per_env
```

Override one cup dimension before `finalize()` and assert the resolved spawner receives that override. Finalize twice and assert the first result is not mutated by the second.

Assert coupled selectors contain:

```python
SceneEntityCfg("source_cup")
SceneEntityCfg("target_cup")
```

and the proxy selects `SceneEntityCfg("source_cup")`.

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py -q
```

Expected: failures show that `finalize()`, `source_cup`, and `target_cup` are absent and selectors still use cup label patterns.

- [ ] **Step 3: Implement finalization and declarative cup configs**

Add optional typed scene fields:

```python
source_cup: RigidObjectCfg | None = None
target_cup: RigidObjectCfg | None = None
media: MPMObjectCfg | None = None
```

Implement `FrankaPourEnvCfg.finalize()` using `deepcopy(self)`. Always rebuild both cup configs and media config on the copy, then assign `_resolve_mpm_cell_cap(resolved)` to the copied solver. The source uses dynamic rigid properties and explicit mass; the target uses kinematic rigid properties. Their init poses come from `cup_reset_pos` and `target_cup_reset_pos`.

Use `MujocoRigidBodyPropertiesCfg(gravcomp=1.0)` and the matching joint-drive schema on the Franka spawn config instead of relying on task-side builder mutation for gravity compensation.

Update coupled entries so source and target scene bodies are selected through `body_entities`; keep label patterns only for `TargetCupRigid` and `SpillFloor`.

- [ ] **Step 4: Add fixed tuple annotations**

Replace unspecific public tuple types with:

```python
arm_home: tuple[float, float, float, float, float, float, float]
cup_grasp_box_half: tuple[float, float, float]
cup_reset_pos: tuple[float, float, float]
target_cup_reset_pos: tuple[float, float, float]
```

- [ ] **Step 5: Run config and media tests to verify GREEN**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit deterministic scene configuration**

Run file-scoped pre-commit, stage the modified files, and commit:

```bash
git commit -m "refactor: Own Franka Pour cups in the scene"
```

### Task 4: Move cup runtime state to public RigidObject APIs

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/events.py`
- Modify: `source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py`

- [ ] **Step 1: Write failing runtime ownership and reset tests**

For two environments, assert:

```python
assert env.scene["source_cup"].num_instances == 2
assert env.scene["target_cup"].num_instances == 2
assert torch.allclose(env.cup_pose_e(), env.scene["source_cup"].data.root_pose_w.torch - env_origin_pose)
```

Perturb only environment one through the source cup public writer, call `reset_pour_scene(env_ids=[1])`, and assert environment zero is unchanged while environment one returns to its default scene pose with zero velocity and refilled particles.

Add structural assertions that the task no longer creates `_cup_joint_q`, `_cup_joint_qd`, `_cup_body_ids`, or `_target_body_ids` runtime buffers.

- [ ] **Step 2: Run the focused runtime case to verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py \
  -k "scene_owned or selective_reset" -q
```

Expected: scene lookup or public-state assertions fail because the cups are still builder-only.

- [ ] **Step 3: Narrow the builder hook to solver-only objects**

Delete `_build_custom_proto`, `_add_cup_body`, generic label-start/rewrite helpers, manual source/target ID lists, manual free-joint coordinate lists, and direct cup FK bookkeeping.

In the per-world hook:

1. resolve the imported source and target cup bodies by full environment path;
2. clear particle collision from the imported source grasp proxy;
3. attach invisible particle-only hollow meshes to both scene-owned bodies;
4. create the invisible rigid-only `TargetCupRigid` body; and
5. create the invisible particle-only `SpillFloor` plane.

Keep finger contact material configuration in this hook until the same per-shape contact stiffness can be expressed through a public scene config. Every lookup must raise on zero or multiple matches.

- [ ] **Step 4: Read and reset through public scene assets**

Resolve:

```python
self._source_cup = self.scene["source_cup"]
self._target_cup = self.scene["target_cup"]
```

Use their `data.root_pose_w` for observations and containment. Reset the source cup with `write_root_pose_to_sim_index()` and `write_root_velocity_to_sim_index()`. Reset the robot with its existing public joint writers and reset particles with the MPM object's public writers. Do not access Newton `body_q`, `joint_q`, or `joint_qd`, and do not call `newton.eval_fk()`.

Remove the redundant `NewtonManager.set_decimation(self.cfg.decimation)` call from `load_managers()`.

- [ ] **Step 5: Run the complete multi-world runtime test to verify GREEN**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py -q
```

Expected: all eager, captured, parity, isolation, and selective-reset cases pass.

- [ ] **Step 6: Commit the public-state refactor**

Run file-scoped pre-commit, stage the modified files, and commit:

```bash
git commit -m "refactor: Reset Pour cups through asset APIs"
```

### Task 5: Correct and lock down the pre-grasp reset pose

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py`
- Modify: `source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py`
- Create or modify diagnostic: `_scratch/calibrate_franka_pour_pregrasp.py`

- [ ] **Step 1: Write a failing pose invariant**

After reset, compute the actual DiffIK TCP and cup grasp point and require:

```python
distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
assert torch.all(distance < 0.08)
assert torch.all(env.tcp_pos_e()[:, 2] > env.cup_grasp_point_e()[:, 2])
```

- [ ] **Step 2: Run the invariant to verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py \
  -k pregrasp -q
```

Expected: the current pose fails at approximately 0.89 m.

- [ ] **Step 3: Calibrate against the live Newton articulation**

Use a one-environment no-MPM diagnostic that drives the existing DiffIK action from a collision-free Franka seed to the desired TCP pose immediately above the grasp point, records the converged seven arm joints, resets from those joints, and remeasures the invariant. Keep the chosen pose within Panda joint limits and verify the target receiver has positive clearance from both fingers.

- [ ] **Step 4: Update the reset config and rerun GREEN**

Replace `arm_home` with the measured seven-joint tuple and update its comment with the measured TCP-to-grasp distance. Run the pre-grasp invariant twice from a fresh process and require both runs to pass.

- [ ] **Step 5: Commit the pose correction**

Run file-scoped pre-commit, stage the config/test and any retained diagnostic, and commit:

```bash
git commit -m "fix: Start Franka Pour at the cup pre-grasp"
```

### Task 6: Add Kit and Newton multi-environment visualization regressions

**Files:**
- Create: `source/isaaclab_tasks/test/contrib/test_franka_pour_visualization.py`
- Modify if required: `source/isaaclab_newton/test/assets/test_articulation.py`
- Modify if required: `source/isaaclab_newton/test/assets/test_rigid_object.py`
- Modify if required: `source/isaaclab_visualizers/test/test_newton_adapter.py`

- [ ] **Step 1: Add the failing Kit scene test**

Launch two environments with origins at `+1.25` and `-1.25` and Kit visualization. Assert:

```python
for env_id in (0, 1):
    assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot").IsValid()
    assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/source_cup").IsValid()
    assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/target_cup").IsValid()
```

Compare a representative robot body and source-cup Fabric/USD world transform with Newton state and assert no extra `env_origin` is present. Assert the viewport camera has no `omni:scenePartition` value by default.

- [ ] **Step 2: Verify the test catches the pre-fix behavior**

Run the test against the pre-merge parent or temporarily revert the relevant implementation commit. Expected: environment one's robot is missing and both cup prims are missing.

- [ ] **Step 3: Add the Newton viewer all-world test**

Initialize the native visualizer headlessly and assert `model.world_count == 2`, no visible-world filter is active by default, world offsets are zero, and visible robot/cup/particle shape world IDs cover `{0, 1}`.

- [ ] **Step 4: Run both visualization tests to verify GREEN**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/contrib/test_franka_pour_visualization.py -q
```

Expected: Kit and Newton cases pass.

- [ ] **Step 5: Commit visualization coverage**

Run file-scoped pre-commit, stage the tests and any required infrastructure correction, and commit:

```bash
git commit -m "test: Cover Pour multi-environment rendering"
```

### Task 7: Document the task-facing change and clean lifecycle debt

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py`
- Modify: `source/isaaclab_tasks/changelog.d/max-franka-pour-scene-assets.rst`
- Modify if present: Franka Pour run documentation and benchmark help text

- [ ] **Step 1: Update module documentation**

Describe the source/target cups as scene-owned rigid objects, the grasp proxy and spill floor as solver-only, and the source cup proxy mapping as the only cross-solver mapping. Remove statements saying the cup is built exclusively through a Newton prototype.

- [ ] **Step 2: Add the changelog fragment**

Create:

```rst
Fixed
^^^^^

* Fixed the Newton Franka Pour task to expose its visible cups as scene-owned rigid objects,
  reset them through public asset APIs, and render every cloned environment consistently.
```

- [ ] **Step 3: Run static task tests and file-scoped hooks**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_spawner.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit documentation and cleanup**

```bash
git commit -m "docs: Explain scene-owned Pour assets"
```

### Task 8: End-to-end verification and review

**Files:**
- Verify all changed files
- Do not create a PR or push a branch

- [ ] **Step 1: Run the complete focused test matrix**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/cloner/test_rename_builder_labels.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  source/isaaclab_newton/test/assets/test_mpm_object.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_spawner.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_multiworld_runtime.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_visualization.py -q
```

Expected: zero failures.

- [ ] **Step 2: Re-run the user-visible zero-agent reproduction**

Run with the compatible source checkouts:

```bash
export NEWTON_SOURCE_DIR=/home/maximiliank/Work/newton-worktrees/implicit-mpm-coupled-sparse
export WARP_SOURCE_DIR=/home/maximiliank/Work/warp-max
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Pour-Franka-v0 --num_envs 2 --viz kit
```

Expected: both robots, both visible cups, and both particle sets appear at their matching environment origins without CUDA copy errors or tunnelling during the reset smoke interval.

- [ ] **Step 3: Verify captured stepping and one training iteration**

Run the captured/eager parity test, then:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Isaac-Pour-Franka-v0 \
  --num_envs 2 --headless --max_iterations 1
```

Expected: environment construction, reset, one rollout, and one optimization iteration complete.

- [ ] **Step 4: Run repository hooks and separate baseline failures**

Run:

```bash
./isaaclab.sh -f
```

Expected for completion: all changed files pass. If unrelated tracked scratch/demo files still fail, rerun pre-commit on the complete changed-file list and record the unrelated repository-wide failures verbatim; do not modify those unrelated files.

- [ ] **Step 5: Perform spec and code-quality review**

Review the final diff against every design requirement, confirm there are no conflict markers or stale direct-state paths, and run:

```bash
git diff --check
git status --short
git log --oneline --decorate -12
```

Expected: clean diff checks and only intentional branch commits. Leave the branch local and do not open a PR or push.
