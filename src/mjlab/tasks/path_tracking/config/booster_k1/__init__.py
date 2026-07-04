from mjlab.tasks.path_tracking.rl import PathTrackingOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  booster_k1_path_flat_env_cfg,
  booster_k1_path_rough_env_cfg,
)
from .rl_cfg import booster_k1_path_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-PathTracking-Rough-Booster-K1",
  env_cfg=booster_k1_path_rough_env_cfg(),
  play_env_cfg=booster_k1_path_rough_env_cfg(play=True),
  rl_cfg=booster_k1_path_ppo_runner_cfg(),
  runner_cls=PathTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-PathTracking-Flat-Booster-K1",
  env_cfg=booster_k1_path_flat_env_cfg(),
  play_env_cfg=booster_k1_path_flat_env_cfg(play=True),
  rl_cfg=booster_k1_path_ppo_runner_cfg(),
  runner_cls=PathTrackingOnPolicyRunner,
)
