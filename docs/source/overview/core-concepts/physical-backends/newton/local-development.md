# Local Newton and Warp source validation

Use explicit source overrides when testing unpublished Newton and Warp changes.
`isaaclab.sh` never discovers adjacent checkouts implicitly.

The reviewed feature pair is:

- Newton: `918f901193ca45b87b46ff4122c3a45b6c05424e`
- Warp: `78bcb2617d23b145749c1955474b098e713f99dd`

Set checkout paths without committing workstation-specific locations:

```bash
export NEWTON_SOURCE_DIR=/path/to/newton
export WARP_SOURCE_DIR=/path/to/warp
```

Validate that both checkouts are clean, have the reviewed revisions, and supply
the imported Python packages and Warp native library:

```bash
./isaaclab.sh -p tools/validate_newton_sources.py \
  --newton_revision 918f901193ca45b87b46ff4122c3a45b6c05424e \
  --warp_revision 78bcb2617d23b145749c1955474b098e713f99dd
```

These overrides are only a local-development mechanism. A reproducible clean
installation requires an immutable published Newton Git revision and a
content-addressed Warp wheel built from the reviewed Warp revision. A Warp Git
dependency is insufficient because the repository does not contain its native
libraries. Never commit a local `file://` requirement or a path to a sibling
checkout.
