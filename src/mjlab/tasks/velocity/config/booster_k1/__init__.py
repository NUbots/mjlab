from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  booster_k1_flat_env_cfg,
  booster_k1_rough_env_cfg,
)
from .rl_cfg import booster_k1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-K1",
  env_cfg=booster_k1_rough_env_cfg(),
  play_env_cfg=booster_k1_rough_env_cfg(play=True),
  rl_cfg=booster_k1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-K1",
  env_cfg=booster_k1_flat_env_cfg(),
  play_env_cfg=booster_k1_flat_env_cfg(play=True),
  rl_cfg=booster_k1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
