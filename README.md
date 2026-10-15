<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Isaac Lab interactive demos

Static browser simulations and exporter source published from the reusable
`interactive-releases` branch. Physics and policy inference run locally in the browser.

- [Hosted gallery](https://isaac-sim.github.io/IsaacLab/demos/interactive/)
- [Scene details and rebuilding](site/interactive/README.md)
- [Exporters and toolchain](exporters/README.md)

Generated from source commit `2404de0e9dbf97b479823e409def8194cde208eb`. The internal Newton Web compiler is
required for new physics captures; it is not included here. Serving reviewed output
requires only static files.

## Update

In the source project, commit the reviewed changes and run:

```bash
python tools/publish.py --output /path/to/interactive-releases-checkout
```

Review, commit and push that checkout to the same `interactive-releases` branch.
Keep existing compatibility directories unchanged. No new release branch, tag or
numbered gallery directory is required. The docs PR pins the resulting official
commit archive and its SHA256; documentation serves the stable `/demos/interactive/` path.

Archive URL: `https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/<publishing-commit>`.
