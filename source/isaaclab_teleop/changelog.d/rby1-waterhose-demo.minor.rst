Fixed
^^^^^

* Fixed teleoperation actions being emitted in the world frame while a
  configured target frame is temporarily unavailable; actions now resume only
  after the frame resolves while Play, Stop, and Reset events continue to be
  processed.
