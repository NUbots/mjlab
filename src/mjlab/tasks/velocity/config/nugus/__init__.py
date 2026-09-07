from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  nubots_nugus_flat_env_cfg,
  nubots_nugus_flat_history_env_cfg,
  nubots_nugus_flat_v57_env_cfg,
  nubots_nugus_rough_env_cfg,
  nubots_nugus_rough_history_env_cfg,
)
from .rl_cfg import (
  nubots_nugus_history_ppo_runner_cfg,
  nubots_nugus_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Nubots-Nugus",
  env_cfg=nubots_nugus_rough_env_cfg(),
  play_env_cfg=nubots_nugus_rough_env_cfg(play=True),
  rl_cfg=nubots_nugus_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Nubots-Nugus",
  env_cfg=nubots_nugus_flat_env_cfg(),
  play_env_cfg=nubots_nugus_flat_env_cfg(play=True),
  rl_cfg=nubots_nugus_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# The observation-history variants (see ``mjlab.rl.obs_history``). Registered
# alongside the plain tasks rather than replacing them: a policy only loads
# against the task that builds its observation layout, so evaluating a plain
# policy and a history policy in one comparison needs both to stay registered.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Nubots-Nugus-History",
  env_cfg=nubots_nugus_rough_history_env_cfg(),
  play_env_cfg=nubots_nugus_rough_history_env_cfg(play=True),
  rl_cfg=nubots_nugus_history_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Nubots-Nugus-History",
  env_cfg=nubots_nugus_flat_history_env_cfg(),
  play_env_cfg=nubots_nugus_flat_history_env_cfg(play=True),
  rl_cfg=nubots_nugus_history_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# The v57 generation from the add-phase-clock branch: a policy-owned gait
# clock, per-actuator current and shared-bus voltage observations, and a
# scripted head. Registered like the history variants and for the same
# reason -- a checkpoint only loads against the task that builds its
# observation layout, and this one is 112 actor dims against the plain
# task's 71. Uses the history runner config: v57 trained with RMA_E2E=1,
# whose inference path is exactly this TCN-over-the-window actor.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Nubots-Nugus-V57",
  env_cfg=nubots_nugus_flat_v57_env_cfg(),
  play_env_cfg=nubots_nugus_flat_v57_env_cfg(play=True),
  rl_cfg=nubots_nugus_history_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
