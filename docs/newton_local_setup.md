# Isaac Lab Newton Dependency Setup

Last verified: 2026-06-10.

The waterhose demo depends on Newton PR 2848 until the coupled-solver APIs land in a released Newton version.
The repository pins directly to the upstream PR 2848 commit verified for this branch. No developer-local
Newton checkout or uncommitted Newton edits are required for setup.

## Active Branches

- Isaac Lab fork branch: `max/waterhose-coupled-experimental`.
- Newton dependency: upstream `newton-physics/newton` PR 2848, verified at commit
  `97d745063ff5556032a09a7c7f5699032f2de053`.

The verified Newton dependency for this demo is:

```text
newton[sim] @ git+https://github.com/newton-physics/newton.git@97d745063ff5556032a09a7c7f5699032f2de053
```

This installs the verified PR 2848 source directly from GitHub. To refresh the pin, resolve the current
PR head with:

```bash
git ls-remote https://github.com/newton-physics/newton.git refs/pull/2848/head
```

Then update every Newton direct URL in the repo to the same commit and rerun the smoke/profile checks.
The PR currently requires a Warp development build, so Isaac Lab's core Warp requirement must allow
`warp-lang>=1.14.0.dev20260514`.

## Runtime Wiring

The waterhose demo uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from
`newton.solvers.experimental.coupled`, and every Newton dependency declaration in this repo uses the same
`newton[sim]` direct URL above so pip/uv does not resolve a bare Newton package without the `[sim]` extra.

There is no vendored coupled solver fallback in this worktree. If PR 2848 changes, verify the new
PR head commit, keep all direct URL refs consistent, and re-run the smoke test.

## Repository Rule

Do not push this work directly to upstream Isaac Lab. Use the fork branch
`max/waterhose-coupled-experimental`.

The Isaac Lab waterhose task is self-contained and does not depend on Newton example scripts.
