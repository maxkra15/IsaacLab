* Migrated Newton cloning and deformable range ownership to the native surface
  and volume metadata, and removed duplicate group recording now performed by
  Newton. Kept unset solver options at their native defaults. These changes
  required the pinned Newton development revision;
  use the upstream-compatible branch with Newton 1.6.1.
* Bound cable segment state to the native articulation's explicit free root
  and rod links, retaining standalone topology validation. The cable's
  ``root_view`` now included the free-root joint; use ``num_segments`` rather
  than adding one to ``root_view.link_count`` to size segment buffers.
