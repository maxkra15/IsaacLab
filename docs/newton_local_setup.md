# Isaac Lab Newton Dependency Setup

Last verified: 2026-05-28.

The waterhose demo depends on Newton PR 2848 until the coupled-solver APIs land in a released Newton version.

## Active Branches

- Isaac Lab fork branch: `waterhose-demo`.
- Newton PR 2848 branch: `pr-2848-coupled-solver-framework-latest`, tracking `origin/pr-2848-head`.

The verified Newton PR 2848 commit is:

```text
c2f21df3acc0f06d207812810b2e27ca7c4da08c
```

`origin/pr-2848-head` was refreshed directly from `refs/pull/2848/head` and points at that same commit.

## Runtime Wiring

The waterhose demo uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from `newton.solvers.coupled_experimental`, and the local Newton package dependencies pin `newton[sim]` to the verified PR commit above.

There is no vendored coupled solver fallback in this worktree. If PR 2848 changes, update the Newton pins and any separate Newton checkout together.

## Repository Rule

Do not push this work directly to upstream Isaac Lab. Use the fork branch `waterhose-demo`.

The Isaac Lab waterhose task is self-contained and does not depend on Newton example scripts.
