Added
^^^^^

* Added the standard ``IsaacContrib-Juggle-Ball-KukaAllegro-RL`` task for
  continuous one-metre toss-and-catch learning with Newton physics, compact
  palm-translation and hand-aperture actions, repeated physical cycles,
  reset-aware progress, and continuously parameterized adaptive resets.
* Added FK-consistent pre-throw pose and ball randomization, catchable
  flight and return resets, uniform coverage, and online sampling near 50%
  predicted success without demonstrations or trajectory targets.
* Added continuous catch-and-rethrow episodes with phase-local success used as
  adaptive-reset evidence, plus full-resume persistence for curriculum state.
