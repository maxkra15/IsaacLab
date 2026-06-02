# Isaac Lab Newton Dependency Setup

Last verified: 2026-06-02.

The waterhose demo depends on Newton PR 2848 until the coupled-solver APIs land in a released Newton version.
For this worktree, use the local rebased checkout so current Newton main changes and PR 2848 are both present.

## Active Branches

- Isaac Lab fork branch: `waterhose-demo`.
- Newton latest-main checkout: `/home/maximiliank/Work/newton-latest`, branch `newton-latest-main`, tracking `origin/main`.
- Newton coupled checkout: `/home/maximiliank/Work/newton-coupled`, branch `newton-coupled-rebased-main`, PR 2848 rebased onto `origin/main`.

The verified Newton dependency for this demo is:

```text
newton[sim] @ file:///home/maximiliank/Work/newton-coupled
```

This installs the local PR 2848 rebase. The PR currently requires a Warp development build,
so Isaac Lab's core Warp requirement must allow `warp-lang>=1.14.0.dev20260514`.

## Runtime Wiring

The waterhose demo uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from
`newton.solvers.experimental.coupled`, and the local Newton package dependencies pin `newton[sim]` to the
local checkout above.

There is no vendored coupled solver fallback in this worktree. If PR 2848 or main changes, update
`/home/maximiliank/Work/newton-coupled` and the local dependency path together.

## Repository Rule

Do not push this work directly to upstream Isaac Lab. Use the fork branch `waterhose-demo`.

The Isaac Lab waterhose task is self-contained and does not depend on Newton example scripts.
