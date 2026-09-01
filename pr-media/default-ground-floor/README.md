# Default ground plane PR media

Generated from IsaacLab PR #7399 with the same fixed camera and bundled ground-plane asset.

- Kit uses PhysX.
- Newton RTX and Newton GL use MJWarp.
- Each clip contains 300 frames at 60 FPS (5 seconds).
- Each renderer panel is captured at 1280 × 720; the comparison MP4s are 3840 × 720.
- Presentation lighting is calibrated to keep the warm-white floor below clipping: Kit uses the
  environment's neutral light at 40% of its normal intensity, Newton RTX uses a neutral dome at
  intensity 400, and Newton GL uses the established neutral ambient profile at exposure 0.40.

The renderer-specific calibration affects these comparison captures only; it is not part of the
runtime ground-plane change. The GIFs are reduced-size previews for the pull-request description,
while the MP4 files retain the full panel resolution.
