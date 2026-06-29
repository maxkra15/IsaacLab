# Waterhose Physics Correctness and Performance Design

## Goal

Apply a focused set of evidence-backed corrections to the waterhose task without redesigning its coupled-solver architecture. The result must preserve the existing MuJoCo/VBD staggered proxy coupling, feed each solver the contacts it owns, restore clean episode boundaries, and expose validated lower-cost solver settings without reducing the high-fidelity defaults.

## Current behavior

The task advances one environment physics step at `dt=0.01` and lets `CoupledNewtonCfg` divide it into internal substeps. The MuJoCo entry generates contacts from its entry-local collision pipeline. The VBD entry receives contacts from the proxy-local pipeline, which also contains the VBD world's non-proxy contact pairs. Staggered synchronization transfers the right-finger poses into VBD and returns proxy contact wrench feedback to MuJoCo on the next coupled substep. This topology and stepping order are internally consistent and will remain unchanged.

Three defects are in scope:

1. `WATERHOSE_SUBSTEPS` and `WATERHOSE_VBD_ITERS` are documented but ignored; live configuration is hard-coded to 10 substeps and 20 VBD iterations.
2. The robot collision-flag initialization that limits Newton collisions to the two right fingers was removed, leaving 30 unintended robot collision shapes active in a pre-finalization model inspection.
3. Episode reset only resets robot joints, allowing cable, plug, anchor, and other scene state to carry across episodes.

## Design

### Validated performance settings

Read `WATERHOSE_SUBSTEPS` and `WATERHOSE_VBD_ITERS` when constructing `WaterhoseEnvCfg`. Defaults remain 10 substeps and 20 VBD iterations for high-fidelity operation. The existing performance report validates 8 substeps and 16 VBD iterations as an optional lower-cost mode. Values must be base-10 positive integers. Missing variables use defaults; malformed, zero, or negative values raise `ValueError` naming the offending variable.

The values configure only `CoupledNewtonCfg.num_substeps` and the waterhose VBD entry's `VBDSolverCfg.iterations`. No timestep, decimation, graph-capture, contact-refresh, or coupling-mode behavior changes.

### Robot collision filtering

Restore a model-initialization callback for the task's robot asset. It identifies robot bodies by the existing `rby1df` body-label prefix and clears `ShapeFlags.COLLIDE_SHAPES` and `ShapeFlags.COLLIDE_PARTICLES` on every robot shape except the two right gripper finger shapes. It does not alter cable, plug, anchor, socket, housing, ground, or other non-robot shapes.

The existing proxy collision pair filter remains in place. The restored flag filter prevents unintended robot geometry from participating in the MuJoCo entry collision pipeline and supplies a defensive, task-level collision invariant before coupled model views are finalized.

### Full episodic reset

Use Isaac Lab's standard `mdp.reset_scene_to_default` reset event instead of a robot-only joint reset. Reset joint targets along with physical state so stale commands do not immediately perturb the restored pose. This restores all scene assets for selected environment IDs through their normal asset write APIs and lets the coupled manager's existing teleport/reset handling synchronize solver state.

The gripper material reset remains unchanged because it is independent of transform/state reset and is already routed through the physics-manager notification path.

## Error handling

Configuration errors are rejected during environment configuration, before simulation construction. Collision filtering is label-based and leaves unknown or unrelated shapes unchanged. If the expected right-finger labels change, those shapes will no longer be exempted, making the failure conservative rather than silently activating the full robot.

## Tests and verification

Regression tests will be added before implementation and observed failing against the current code. They will verify:

- default and environment-overridden substep/iteration values;
- clear rejection of malformed and non-positive tuning values;
- collision flags are cleared only for non-finger robot shapes;
- the reset event uses full scene reset and resets joint targets.

After implementation, run the focused regression file and existing waterhose contrib tests. Also run syntax/import checks and inspect the diff for unrelated changes. CUDA is unavailable in the current workspace, so GPU rollout and throughput validation cannot be performed locally; performance expectations for the optional 8/16 mode are based on the repository's existing benchmark report and must be stated as such rather than claimed as newly measured.

## Out of scope

- Replacing staggered proxy coupling or changing its one-substep feedback latency.
- Moving contacts into a manager-global collision buffer.
- Changing collision buffer capacities, VBD contact matching, timestep, or decimation.
- Broad coupled-manager or Newton solver refactors.
- Publishing, pushing, or otherwise modifying any remote repository.
