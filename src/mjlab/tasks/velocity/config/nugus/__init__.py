from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  nubots_nugus_flat_env_cfg,
  nubots_nugus_rough_env_cfg,
)
from .rl_cfg import nubots_nugus_ppo_runner_cfg
from .runner import NugusOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Nubots-Nugus",
  env_cfg=nubots_nugus_rough_env_cfg(),
  play_env_cfg=nubots_nugus_rough_env_cfg(play=True),
  rl_cfg=nubots_nugus_ppo_runner_cfg(),
  runner_cls=NugusOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Nubots-Nugus",
  env_cfg=nubots_nugus_flat_env_cfg(),
  play_env_cfg=nubots_nugus_flat_env_cfg(play=True),
  rl_cfg=nubots_nugus_ppo_runner_cfg(),
  runner_cls=NugusOnPolicyRunner,
)
