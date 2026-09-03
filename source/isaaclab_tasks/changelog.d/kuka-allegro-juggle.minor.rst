Added
^^^^^

* Added the standard ``IsaacContrib-Juggle-Ball-KukaAllegro-RL`` task for
  continuous one-metre toss-and-catch learning with Newton physics, compact
  palm-translation and hand-aperture actions, repeated physical cycles,
  reset-aware progress, and continuously parameterized adaptive resets.
* Added broad FK-consistent pre-throw pose and ball randomization, catchable
  flight/return resets, exact uniform coverage, and online sampling near 50%
  predicted success without demonstrations or trajectory targets.
* Added full-resume persistence for reset-curriculum evidence with physical and
  outcome compatibility fingerprints and rank-distinct DDP sampling streams.

Changed
^^^^^^^

* Consolidated the former LOW, one-cycle one-metre, continuous-training, and
  continuous-play task IDs into the standard Juggle task. Use
  ``IsaacContrib-Juggle-Ball-KukaAllegro-RL`` for both training and playback;
  playback automatically runs continuous rallies and resets only on physical
  failure or numerical-safety termination. Old 23-action and variant-specific
  checkpoints are intentionally incompatible with the 108-observation,
  four-action policy.
* Kept phase-local success as adaptive-reset evidence without rewarding or
  terminating it. Training now continued across catch-and-rethrow transitions
  with 35% protected randomized held starts, 15% uniform full-domain coverage,
  and 50% success-boundary sampling.
