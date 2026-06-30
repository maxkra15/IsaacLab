# Franka Pour Scene-Owned Assets Design

## Objective

Bring the Newton Franka Pour task onto the current Isaac Lab `develop` baseline and make its visible
rigid objects participate in the normal `ManagerBasedRLEnv`/`InteractiveScene` lifecycle. The task
must retain isolated per-world implicit MPM grids and outer CUDA graph capture while rendering and
resetting every environment consistently in Kit and the Newton visualizer.

## Baseline integration

The public feature branch is already shared, so integrate `upstream/develop` with a merge commit
rather than rewriting its history. Conflict resolution must retain the feature branch's coupled
solver, sparse-grid capture, and MPM object extensions while taking upstream's current cloning,
Fabric binding, USD replication, and Kit scene-partition behavior. In particular, the result must
include the behavior introduced by upstream changes #6073, #6119, #6204, and #6238.

## Scene ownership

`PourSceneCfg` will own these visible task entities:

- `robot`: the existing Franka `ArticulationCfg`.
- `source_cup`: a dynamic `RigidObjectCfg`.
- `target_cup`: a kinematic `RigidObjectCfg`.
- `media`: the existing `MPMObjectCfg`.

The two cup configs use a task-local rigid mesh spawner. The spawner authors a root rigid body and a
watertight open-top bowl mesh from `make_cube_bowl_mesh()`, including collision, mass, rigid-body,
visual-material, and physics-material USD schemas. This makes the cups cloneable, renderable,
recordable, and accessible through the standard scene dictionary without introducing a general
Isaac Lab API for one task's geometry.

The dynamic source cup still requires two solver representations on the same Newton body:

- its visible hollow mesh is particle-collision-only; and
- an invisible solid box is rigid-shape-collision-only for the Panda grasp.

The Newton per-world builder hook will configure the imported visible mesh's flags and attach the
invisible grasp proxy to the scene-owned source-cup body. The hook will also create the invisible
rigid-only receiver duplicate and spill plane required for disjoint coupled-solver ownership. Those
objects have no user-facing state and remain solver implementation details.

The hook will no longer build, clone, or relabel visible cups. Generic label-rewrite code and manual
source-cup body/joint bookkeeping will be deleted.

## Declarative backend configuration

Robot gravity compensation will be authored through Newton's MuJoCo USD schema configuration where
the current APIs support it. Backend-specific shape flags and contact stiffness that cannot yet be
expressed by an Isaac Lab asset config remain in one narrowly scoped builder hook. The hook must use
full environment paths and validate that every expected body or shape resolves exactly once.

Coupled solver ownership will use `SceneEntityCfg("source_cup")` and
`SceneEntityCfg("target_cup")`. Label patterns remain only for the hidden receiver proxy and spill
plane. The proxy mapping will select `SceneEntityCfg("source_cup")`.

## Derived configuration

The final environment count and command-line/Hydra overrides are known only when the environment is
constructed. `FrankaPourEnvCfg` will therefore provide an explicit finalization method that:

1. deep-copies the caller's config;
2. deterministically rebuilds the source cup, target cup, and media scene configs from final values;
3. resolves total sparse MPM capacity from the final environment count; and
4. returns the resolved copy.

`FrankaPourEnv` will pass only this resolved copy to `ManagerBasedRLEnv`. Reusing or modifying the
caller's config cannot retain stale generated media or cup geometry.

## Runtime state flow

After physics initialization the environment resolves:

- `self.scene["robot"]` as an articulation;
- `self.scene["source_cup"]` and `self.scene["target_cup"]` as rigid objects; and
- `self.scene["media"]` as an MPM object.

Observations, rewards, finite-state checks, and containment calculations read cup poses from the
rigid objects' public data views. Reset uses the rigid object's public indexed pose and velocity
writers, the articulation's public joint writers, and the MPM object's public particle writers.
The task no longer writes Newton `joint_q`, `joint_qd`, or `body_q` directly and no longer calls
`newton.eval_fk()` for cup reset.

The existing public `NewtonManager.reset_solver_state(world_mask=...)` call remains the boundary for
clearing solver-private state after selected environments are reset.

## Initial pose and cameras

The Franka reset joint pose must place the DiffIK TCP at a documented pre-grasp offset from the
source-cup grasp point. A runtime invariant will bound this distance and prevent comments/config from
drifting apart again. The pose will be calibrated against the actual Newton articulation rather than
copied from an unverified trajectory.

Kit's legacy viewer camera will use environment-relative framing for environment zero. Native
visualizers continue to use their own explicit `VisualizerCfg`; no viewer-side world spacing is
added because Newton body poses are already in world coordinates.

## Lifecycle cleanup

The base environment already forwards decimation to the physics manager before `load_managers()`, so
the task's duplicate call will be removed. Public tuple configuration fields will receive fixed,
element-specific type annotations. The custom `load_managers()` override remains only if task MDP
terms need post-physics handles before manager construction.

## Verification

Tests will cover four layers:

1. Pure configuration and geometry tests: deterministic finalization, caller-config immutability,
   typed scene entities, solver selectors, and bowl-spawner USD schemas.
2. Newton runtime tests: scene-owned cup views, reset isolation, public-state agreement, per-world
   solver ownership, particle isolation, and outer CUDA graph capture.
3. Pose tests: reset TCP-to-grasp distance and source/target cup world positions in every environment.
4. Kit/visualizer tests: both environment robot and cup prims exist, scene partitions do not hide all
   but environment zero by default, and Fabric/USD transforms agree with Newton state without an
   extra environment-origin offset.

Every regression test must be observed failing against the relevant pre-fix state and passing after
the implementation. Focused package tests and repository formatting hooks are required before the
branch is considered ready.

## Non-goals

- No changes to Warp or Newton are required for this scene-ownership refactor.
- No NanoVDB particle-grid construction refactor is included.
- No replacement of the coupled solver or the validated contact model is included.
- No pull request or remote push is performed as part of this work.
