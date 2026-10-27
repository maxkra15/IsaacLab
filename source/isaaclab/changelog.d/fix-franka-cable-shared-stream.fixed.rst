* Fixed visual material initialization and updates unregistering a shared Warp CUDA stream when
  PyTorch used the same stream, preventing subsequent CUDA graph capture from failing.
