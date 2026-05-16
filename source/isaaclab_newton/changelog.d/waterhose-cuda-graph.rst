Fixed
^^^^^

* Fixed Newton CUDA graph capture to use the selected simulation device so
  runtime control target updates are respected on non-default CUDA devices.
* Fixed coupled Newton manager construction for Newton's solver-factory
  ``SolverCoupled.Entry`` and coupled-solver ``Config`` APIs.
