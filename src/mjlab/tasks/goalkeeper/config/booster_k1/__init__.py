from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_k1_block_env_cfg
from .rl_cfg import booster_k1_block_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Block-Booster-K1",
  env_cfg=booster_k1_block_env_cfg(),
  play_env_cfg=booster_k1_block_env_cfg(play=True),
  rl_cfg=booster_k1_block_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
