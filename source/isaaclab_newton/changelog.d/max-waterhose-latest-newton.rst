Added
^^^^^

* Added named Newton pre-render callbacks for procedural visuals such as cable
  curves that cannot use rigid-body or particle transform synchronization.
* Added one-shot post-solver initialization callbacks for assets whose setup
  depends on solver-owned data.

Fixed
^^^^^

* Preserved coupled-solver manager compatibility with the current derived-state
  refresh boundaries after merging the latest ``develop`` branch.
* Seeded explicit rigid-contact capacity before solver construction so VBD
  contact-history allocation matches the collision pipeline.
* Preserved solver ownership of articulation poses across generic forward-
  kinematics refreshes.
* Fixed Kit rendering of Newton-driven articulated assets with instanced
  visual meshes by mirroring body poses into authored USD xforms and Fabric
  world matrices, then using Cubric to propagate instance-proxy transforms and
  notify Kit's render delegate.
* Restored static render-only USD meshes in Newton visualization models while
  keeping those meshes detached from collision and particle contact.
