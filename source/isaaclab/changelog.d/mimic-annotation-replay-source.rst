Added
^^^^^

* Added :attr:`~isaaclab.envs.MimicEnvCfg.annotation_replay_action_key` to select
  a recorded action stream for Mimic annotation replay.
* Added
  :attr:`~isaaclab.envs.MimicEnvCfg.annotation_reset_sim_buffer_each_episode`
  so tasks with persistent solver resources can opt out of the annotation
  script's per-episode simulation-buffer reset.
