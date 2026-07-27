Added
^^^^^

* Added support for replaying a task-configured action stream during Mimic
  annotation while preserving ``actions`` as the canonical output field.

Fixed
^^^^^

* Fixed Mimic annotation and data generation for configurations that contain
  only final subtasks and therefore require no intermediate termination signals.
