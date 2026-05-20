from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import nubots_nugus_arm_tracking_env_cfg
from .rl_cfg import nubots_nugus_arm_tracking_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-ArmTracking-Nubots-Nugus",
  env_cfg=nubots_nugus_arm_tracking_env_cfg(),
  play_env_cfg=nubots_nugus_arm_tracking_env_cfg(play=True),
  rl_cfg=nubots_nugus_arm_tracking_ppo_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
