<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Isaac Lab interactive demos

Static browser simulations and exporter source published from the reusable
`interactive-releases` branch. Physics and policy inference run locally in the browser.

- [Hosted gallery](https://isaac-sim.github.io/IsaacLab/demos/interactive/)
- [Published archives and source](https://github.com/isaac-sim/IsaacLab/releases/tag/interactive-demos)
- [Scene details and rebuilding](site/interactive/README.md)
- [Exporters and toolchain](exporters/README.md)

Generated from source commit `d7a784f32b22288a7442dbe11796fe46463cc411`. The internal Newton Web compiler is
required for new physics captures; it is not included here. Serving reviewed output
requires only static files.

## Publish

In the source project, commit the reviewed changes and run:

```bash
python tools/publish.py --output /path/to/interactive-releases-checkout
```

Review, commit and push that checkout to the same `interactive-releases` branch.
The branch currently lives in `maxkra15/IsaacLab`; organization rules require an
administrator to create its official counterpart. Runtime hosting uses Isaac Lab.

From the committed publishing checkout, append a deterministic snapshot to the
single official `interactive-demos` release:

```bash
publishing_commit=$(git rev-parse HEAD)
bundle=/tmp/isaaclab-browser-demos-$publishing_commit.tar.gz
git archive --format=tar --prefix="IsaacLab-$publishing_commit/" "$publishing_commit" | gzip -n > "$bundle"
sha256sum "$bundle"
gh release upload interactive-demos "$bundle" --repo isaac-sim/IsaacLab
```

Preserve existing assets and compatibility directories. Reuse the branch, release
and gallery directory; each new snapshot has its publishing commit in its filename.
The docs PR pins the official release asset URL and SHA256. Its build extracts
`site/` and serves the stable `/demos/interactive/` path. The archive also includes
these exporter sources and rebuilding instructions.
