Added
^^^^^

* Added support for replaying a task-configured action stream during Mimic
  annotation while preserving ``actions`` as the canonical output field.
* Added a task capability preflight for explicit SkillGen opt-outs before the
  environment and cuRobo planner are initialized.

Fixed
^^^^^

* Fixed Mimic annotation and data generation for configurations that contain
  only final subtasks and therefore require no intermediate termination signals.
* Fixed Mimic annotation to honor a task's simulation-buffer reset policy while
  retaining the existing hard-reset behavior by default.
