# Local Newton and Warp source validation

Use explicit source overrides when testing unpublished Newton and Warp changes.
`isaaclab.sh` never discovers adjacent checkouts implicitly.

The reviewed feature pair is:

- Newton: `2c08bfec9fc01710a203ded7e185334ded3f1ca0`
- Warp: `87370fac45bfb90701ee5f390cb7f26fbfab86ef`

Set checkout paths without committing workstation-specific locations:

```bash
export NEWTON_SOURCE_DIR=/path/to/newton
export WARP_SOURCE_DIR=/path/to/warp
```

Validate that both checkouts are clean, have the reviewed revisions, and supply
the imported Python packages and Warp native library:

```bash
./isaaclab.sh -p tools/validate_newton_sources.py \
  --newton_revision 2c08bfec9fc01710a203ded7e185334ded3f1ca0 \
  --warp_revision 87370fac45bfb90701ee5f390cb7f26fbfab86ef
```

These overrides are only a local-development mechanism. A reproducible clean
installation requires an immutable published Newton Git revision and a
content-addressed Warp wheel built from the reviewed Warp revision. A Warp Git
dependency is insufficient because the repository does not contain its native
libraries. Never commit a local `file://` requirement or a path to a sibling
checkout.
