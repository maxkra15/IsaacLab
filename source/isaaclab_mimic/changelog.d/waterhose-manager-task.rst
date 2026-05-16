Added
^^^^^

* Added the Mimic wrapper and data-generation configuration for the RBY1
  waterhose task.

Fixed
^^^^^

* Fixed RBY1 waterhose Mimic environment initialization so it builds the Newton
  task state before Mimic APIs are used.
* Fixed RBY1 waterhose Mimic wrapper imports so Newton dependencies are not
  loaded before the launch mode is resolved.
