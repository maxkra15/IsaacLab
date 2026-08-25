Changed
^^^^^^^

* Expanded the Franka RJ45 pick-and-insert training reset-bank contract to
  20,004 balanced rows and made the six phase-start fractions explicit and
  independent of stored row counts.
* Added a phase-0 reverse curriculum with 1.0--1.6 mm immediate, 1.6--3.5 mm
  quick, and 3.5--12 mm boundary pre-seat bands. Rows remain outside the exact
  reset-time success predicate so the policy must complete and dwell in the
  insertion instead of receiving a no-op terminal reward.
* Added scalable zero-step reset admission using batched IK, analytic cable
  bounds, and exact outer/proxy Newton collision queries. This mode certifies
  only the authored initial state and deliberately does not claim dynamics or
  scripted-recovery validation.
* Existing version-7 reset banks and validation reports are incompatible with
  task contract and pick-insert semantics version 8. Regenerate the bank and
  republish its matching fast-reset report before training. Generate a fresh
  version-8 canonical-goal certificate first; version-7 certificates do not
  satisfy the new source/task binding.
* Existing configuration overrides of ``full_pick_start_fraction`` must also
  set the matching phase-5 entry in ``reset_dataset_phase_fractions``. The two
  fields are intentionally validated as aliases so logged curricula cannot
  disagree about their full-pick share.
* Bound every fast admission record to its final artifact row and restricted
  canonical promotion to the exact 20,004-row reference profile. Legacy
  96-row physical-oracle banks now require an explicit noncanonical output
  path and cannot overwrite the training bank.
