"""Booster K1 standing-start kick environment.

The pass / shoot skill from JOGA (Table V, passing / shooting column), built
on the K1 dribbling environment so both skills share one scene, one action
space and one actor observation layout. That shared layout is what lets a
kick later be finetuned from states sampled mid-dribble (JOGA Sec. II-C).

The robot starts standing with a resting ball a short way ahead and is given
one commanded ball exit velocity for the episode. Rewards are phased around
the strike: get to the ball and face it, hit it at the commanded velocity,
then stand still rather than chase it.

One policy covers what JOGA splits into passing and shooting; their only
difference there is whether lofting the ball is penalized, and here it is.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import flat_orientation_l2
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.dribbling import mdp
from mjlab.tasks.dribbling.config.booster_k1.env_cfgs import (
  BALL_RADIUS,
  COMMAND_NAME,
  booster_k1_dribble_env_cfg,
)
from mjlab.tasks.dribbling.mdp import BallCommandStage, KickCommandCfg

_STEPS_PER_ITER = 24

# JOGA Table VII ramps passing from 1.5-2.5 to 3-5 m/s over 6k iterations on
# a G1. The K1 is smaller and its ball lighter; these are a first guess at
# its range, starting where a clumsy touch already earns partial credit.
KICK_SPEED_STAGES: list[BallCommandStage] = [
  {"step": 0, "speed_range": (1.0, 2.0)},
  {"step": 2_000 * _STEPS_PER_ITER, "speed_range": (1.5, 3.0)},
  {"step": 4_000 * _STEPS_PER_ITER, "speed_range": (2.0, 4.0)},
  {"step": 6_000 * _STEPS_PER_ITER, "speed_range": (2.5, 5.0)},
]
KICK_PLAY_SPEED_RANGE = (2.0, 4.0)

STRIKE_ZONE = (0.15, 0.0)
EPISODE_LENGTH_S = 6.0


def booster_k1_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster K1 standing-start kick configuration."""
  cfg = booster_k1_dribble_env_cfg(play=play)

  # A kick is short: walk up, strike, settle. The ball is meant to leave.
  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.terminations.pop("ball_out_of_reach", None)

  cfg.commands = {
    COMMAND_NAME: KickCommandCfg(
      # One command per episode.
      resampling_time_range=(1.0e6, 1.0e6),
      speed_range=KICK_PLAY_SPEED_RANGE if play else (1.0, 2.0),
      initial_heading_range=(-math.pi / 4, math.pi / 4),
      rel_standing_envs=0.0,
      debug_vis=True,
    )
  }

  reset = cfg.events["reset_robot_and_ball"]
  reset.params["ball_distance_range"] = (0.3, 0.8)
  reset.params["ball_bearing_range"] = (-0.4, 0.4)
  reset.params["ball_radius"] = BALL_RADIUS

  # Shins and knees touching the ball, for the leg-ball penalty.
  leg_ball = ContactSensorCfg(
    name="leg_ball_contact",
    primary=ContactMatch(
      mode="geom", pattern=r".*(shin|knee)_collision$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (leg_ball,)

  ##
  # Rewards (JOGA Table V, passing / shooting column).
  ##

  for name in (
    "ball_in_zone",
    "ball_velocity_match",
    "ball_velocity_along_command",
    "face_ball",
    "drive_ball_along_command",
    "feet_too_close",
  ):
    cfg.rewards.pop(name, None)

  cmd = {"command_name": COMMAND_NAME}
  cfg.rewards["ball_exit_velocity"] = RewardTermCfg(
    func=mdp.ball_exit_velocity, weight=15.0, params={**cmd, "loft_penalty": 0.2}
  )
  cfg.rewards["ball_in_strike_zone"] = RewardTermCfg(
    func=mdp.ball_in_strike_zone, weight=1.5, params={**cmd, "zone": STRIKE_ZONE}
  )
  cfg.rewards["face_ball"] = RewardTermCfg(
    func=mdp.face_ball_before_kick, weight=1.0, params=dict(cmd)
  )
  cfg.rewards["stay_still_after_kick"] = RewardTermCfg(
    func=mdp.stay_still_after_kick, weight=2.0, params=dict(cmd)
  )
  cfg.rewards["time_to_kick"] = RewardTermCfg(
    func=mdp.time_to_kick, weight=-0.25, params=dict(cmd)
  )
  cfg.rewards["flat_orientation"] = RewardTermCfg(func=flat_orientation_l2, weight=-1.5)
  cfg.rewards["leg_ball_contact"] = RewardTermCfg(
    func=mdp.leg_ball_contact,
    weight=-1.0,
    params={"sensor_name": "leg_ball_contact", "force_threshold": 5.0},
  )

  ##
  # Curriculum (JOGA Table VII): the approach reward fades so the policy
  # commits to the kick, and the commanded speed climbs.
  ##

  cfg.curriculum.pop("ball_command", None)
  if not play:
    cfg.curriculum["kick_speed"] = CurriculumTermCfg(
      func=mdp.ball_command_stages,
      params={"command_name": COMMAND_NAME, "stages": KICK_SPEED_STAGES},
    )
    cfg.curriculum["strike_zone_weight"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "ball_in_strike_zone",
        "stages": [
          {"step": 0, "weight": 1.5},
          {"step": 400 * _STEPS_PER_ITER, "weight": 1.0},
          {"step": 1_200 * _STEPS_PER_ITER, "weight": 0.5},
        ],
      },
    )

  return cfg
