* Migrated cloning and deformable ranges to native Newton surface/volume metadata, removing duplicate group records
  and preserving unset solver defaults. These changes required the pinned Newton development revision.
* Bound cable state to native free-root/rod links. ``root_view`` included the free-root joint;
  use ``num_segments`` rather than ``root_view.link_count + 1`` to size segment buffers.
