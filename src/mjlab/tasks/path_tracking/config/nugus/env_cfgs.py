"""NUbots Nugus path tracking environment configurations.

Built on top of the tuned Nugus velocity configurations: the reward
structure, sensors, events, and domain randomization are reused unchanged.
The differences are the command term (a generated walk path instead of a
sampled twist) and the actor's command observation (relative path
waypoints instead of the twist — matching what the deployed walk path
planner can provide from odometry).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.path_tracking import mdp
from mjlab.tasks.path_tracking.mdp import PathCommandCfg
from mjlab.tasks.velocity.config.nugus.env_cfgs import (
  nubots_nugus_flat_env_cfg,
  nubots_nugus_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise


def _convert_to_path_tracking(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap the twist command for a path command, in place.

  The path command's ``command`` property is a desired twist derived from
  the path, so every reward, gate, and curriculum that consumed the
  ``"twist"`` command keeps working — they are only retargeted to the new
  term name. The actor's observation is replaced with the deployment-safe
  relative-waypoint view of the path.
  """
  twist_cmd = cfg.commands.pop("twist")
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  cfg.commands["path"] = PathCommandCfg(
    entity_name="robot",
    # Mid-episode resamples emulate the planner replanning the path on the
    # fly (e.g. an obstruction appears): a fresh path is generated from
    # the robot's current pose while the episode keeps running.
    resampling_time_range=(4.0, 8.0),
    rel_standing_envs=0.1,
    rel_standing_segments=0.1,
    segment_duration_range=(2.0, 4.0),
    lookahead_times=(0.2, 0.5, 1.0, 2.0),
    pos_gain=1.0,
    heading_gain=0.5,
    debug_vis=True,
    # Inherit the velocity task's (possibly play-adjusted) twist ranges as
    # the path segment sampling ranges.
    ranges=PathCommandCfg.Ranges(
      lin_vel_x=twist_cmd.ranges.lin_vel_x,
      lin_vel_y=twist_cmd.ranges.lin_vel_y,
      ang_vel_z=twist_cmd.ranges.ang_vel_z,
    ),
  )

  # Retarget every term that referenced the twist command.
  for reward in cfg.rewards.values():
    if reward.params.get("command_name") == "twist":
      reward.params["command_name"] = "path"
  for group in cfg.observations.values():
    for obs_term in group.terms.values():
      if obs_term.params.get("command_name") == "twist":
        obs_term.params["command_name"] = "path"
  for curriculum in cfg.curriculum.values():
    if curriculum.params.get("command_name") == "twist":
      curriculum.params["command_name"] = "path"

  # Actor: observe the path as relative waypoints, never the twist. Noise
  # models odometry error in the planner-provided relative path; the delay
  # models planner/network latency.
  actor_command = ObservationTermCfg(
    func=mdp.path_waypoints,
    params={"command_name": "path"},
    noise=Gnoise(mean=0.0, std=0.02),
    delay_min_lag=0,
    delay_max_lag=3,  # 0-60ms
  )
  cfg.observations["actor"].terms["command"] = actor_command

  # Critic: clean waypoints plus the privileged desired twist (critic-only
  # inputs are fine — the critic is not deployed).
  cfg.observations["critic"].terms["command"] = ObservationTermCfg(
    func=mdp.path_waypoints,
    params={"command_name": "path"},
  )
  cfg.observations["critic"].terms["target_twist"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "path"},
  )


def nubots_nugus_path_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus rough terrain path tracking configuration."""
  cfg = nubots_nugus_rough_env_cfg(play=play)
  _convert_to_path_tracking(cfg)
  return cfg


def nubots_nugus_path_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus flat terrain path tracking configuration."""
  cfg = nubots_nugus_flat_env_cfg(play=play)
  _convert_to_path_tracking(cfg)
  return cfg
