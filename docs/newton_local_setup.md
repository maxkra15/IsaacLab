# Isaac Lab Newton Dependency Setup

Last verified: 2026-05-29.

The waterhose demo depends on Newton PR 2848 until the coupled-solver APIs land in a released Newton version.

## Active Branches

- Isaac Lab fork branch: `waterhose-demo`.
- Newton PR 2848 branch: `pr-2848-coupled-solver-framework-latest`, tracking `origin/pr-2848-head`.

The verified Newton PR 2848 dependency is:

```text
git+https://github.com/newton-physics/newton.git@refs/pull/2848/head
```

This installs the current PR head from `refs/pull/2848/head`. The PR currently
requires a Warp development build, so Isaac Lab's core Warp requirement must
allow `warp-lang>=1.14.0.dev20260514`.

## Runtime Wiring

The waterhose demo uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from
`newton.solvers.experimental.coupled`, and the local Newton package dependencies pin `newton[sim]` to the
verified PR head above.

There is no vendored coupled solver fallback in this worktree. If PR 2848 changes, update the Newton pins and any separate Newton checkout together.

## Repository Rule

Do not push this work directly to upstream Isaac Lab. Use the fork branch `waterhose-demo`.

The Isaac Lab waterhose task is self-contained and does not depend on Newton example scripts.
