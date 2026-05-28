# Isaac Lab Newton Local Setup

Last verified: 2026-05-28.

This workspace uses sibling checkouts under `/home/maximiliank/Work`.

## Active Branches

- Isaac Lab waterhose demo worktree: `/home/maximiliank/Work/IsaacLab-waterhose-demo` on `waterhose-demo`.
- Newton PR 2848 reference: `/home/maximiliank/Work/newton` on `pr-2848-coupled-solver-framework-latest`, tracking `origin/pr-2848-head`.

The verified Newton PR 2848 commit is:

```text
c2f21df3acc0f06d207812810b2e27ca7c4da08c
```

`origin/pr-2848-head` was refreshed directly from `refs/pull/2848/head` and points at that same commit.

## Runtime Wiring

The waterhose demo uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from `newton.solvers.coupled_experimental`, and the local Newton package dependencies pin `newton[sim]` to the verified PR commit above.

There is no vendored coupled solver fallback in this worktree. If PR 2848 changes, update the Newton pins and the local `/home/maximiliank/Work/newton` checkout together.

## Worktree Rule

Do not push to upstream IsaacLab. The active writable target for this task is the fork branch `waterhose-demo` in `/home/maximiliank/Work/IsaacLab-waterhose-demo`.

The local Newton checkout currently has an untracked `newton/examples/cable_robot/` directory. The Isaac Lab waterhose task is self-contained and does not depend on that untracked reference demo code.
