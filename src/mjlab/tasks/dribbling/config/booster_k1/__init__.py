from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_k1_dribble_env_cfg
from .kick_env_cfgs import booster_k1_kick_env_cfg
from .rl_cfg import booster_k1_dribble_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Dribble-Flat-Booster-K1",
  env_cfg=booster_k1_dribble_env_cfg(),
  play_env_cfg=booster_k1_dribble_env_cfg(play=True),
  rl_cfg=booster_k1_dribble_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Kick-Flat-Booster-K1",
  env_cfg=booster_k1_kick_env_cfg(),
  play_env_cfg=booster_k1_kick_env_cfg(play=True),
  rl_cfg=booster_k1_dribble_ppo_runner_cfg(
    experiment_name="k1_kick", max_iterations=10_000
  ),
  runner_cls=VelocityOnPolicyRunner,
)
