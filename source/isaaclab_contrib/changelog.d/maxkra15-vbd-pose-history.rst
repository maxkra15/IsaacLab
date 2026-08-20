Added
^^^^^

* Added public named-coupler APIs to capture both VBD rigid-pose histories and
  queue selected, graph-safe one-shot restores after normal coupled-state
  distribution without exposing nested solver internals.
* Added a deferred named-VBD input-pose projection API that preserves authored
  post-solver body poses across coupled input distribution and iteration
  restarts while retaining standalone VBD pose-history semantics.
