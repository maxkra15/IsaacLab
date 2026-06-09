# Isaac Lab Newton Local Setup

Last verified: 2026-06-04.

This workspace uses sibling checkouts under `/home/maximiliank/Work`.

## Active Branches

- Isaac Lab feature reference: `/home/maximiliank/Work/IsaacLab` on `feat/newton-implicit-mpm`.
- Isaac Lab MPM base: `/home/maximiliank/Work/IsaacLab-mpm` on `max/newton-mpm-manager`.
- Isaac Lab coupling worktree: `/home/maximiliank/Work/IsaacLab-coupling` on `max/newton-coupling-manager`.
- Newton PR 2848 reference: `/home/maximiliank/Work/newton-coupled` on `newton-coupled-rebased-main`.

The verified local Newton PR 2848 commit, rebased onto current `origin/main`, is:

```text
f825e84e Refactor coupling hooks
```

This local branch has not been pushed.

## Runtime Wiring

`isaaclab.sh` prepends this repo's `source/*` packages and the adjacent Newton checkout by default. Override with `NEWTON_SOURCE_DIR=/path/to/newton`, or set `NEWTON_SOURCE_DIR=` to disable the local Newton checkout.

```bash
NEWTON_SOURCE_DIR=/home/maximiliank/Work/newton-coupled ./isaaclab.sh -p ...
```

The coupling manager expects Newton PR 2848 APIs, especially:

- `newton.solvers.experimental.coupled.SolverCoupled`
- `newton.solvers.experimental.coupled.SolverCoupledProxy`
- `newton.solvers.experimental.coupled.SolverCoupledADMM`
- `SolverCoupled.Entry(..., configure_view=..., substeps=..., in_place=...)`
- `SolverCoupledProxy.Proxy(..., bodies=..., particles=..., collision_pipeline=..., collide_interval=...)`
- `SolverCoupledADMM.ContactPair(...)`
- `SolverCoupledADMM.add_body_particle_attachment(...)`

## Worktree Rule

Do not push to upstream IsaacLab. The active writable target for this task is the fork branch in `/home/maximiliank/Work/IsaacLab-coupling`.
