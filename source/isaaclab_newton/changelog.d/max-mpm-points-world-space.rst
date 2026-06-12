Fixed
^^^^^

* Fixed MPM particle visualization :class:`~pxr.UsdGeom.Points` prims inheriting
  ancestor transforms: the point positions are written in world space, so the
  prims now reset their xform stack and render correctly under translated
  parent prims.
