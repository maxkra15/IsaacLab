Added
^^^^^

* Added :meth:`~isaaclab_contrib.coupling.NewtonCouplerManager.get_proxy_contact_data`
  for diagnostics that need a proxy-local contact buffer and its matching
  destination model/state layout.
* Added :meth:`~isaaclab_contrib.coupling.NewtonCouplerManager.refresh_proxy_collision_contacts`
  for deterministic collision-only admission checks that synchronize a proxy
  view and refresh its contacts without advancing a solver or consuming the
  runtime collision cadence.
