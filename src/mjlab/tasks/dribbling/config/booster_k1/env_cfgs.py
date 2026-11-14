"""Booster K1 dribbling environment.

A port of the dribbling skill from JOGA ("Learning Agile Humanoid Soccer
Skills with Active Vision", Narula, Nguyen, Fey et al., MIT 2026) to the K1.
JOGA trains each soccer skill from scratch with PPO on a task reward in mjlab,
"building on the mjlab base velocity environment" (Appendix B): there is no
reference motion or mocap dataset anywhere in the pipeline. This file starts
from the K1 flat velocity task and applies their recipe:

- Command: a planar ball velocity (Appendix B.1), not a base velocity.
- Observations: proprioception, the command, and the ball as the head camera
  sees it, ``(x, y, z, visible)`` with a field-of-view gate and last-value
  hold. 3-step history for both actor and critic.
- Actions: every joint including the 2-DoF head, so gaze is learned jointly
  with the body under the task reward (their "active vision"; the K1's head
  stands in for their custom neck).
- Rewards: Table V, dribbling column. Curriculum: Table VII. Ball contact:
  Table VIII. PPO: Table IV (see ``rl_cfg.py``).

What is *not* reproduced, because it needs hardware data: the perception
alignment model (replaced by range-proportional noise, their "DR" baseline)
and the WB-UAN actuator residual used for the final sim-to-real finetune.
"""

import math
from dataclasses import replace

import mujoco

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.dribbling import mdp
from mjlab.tasks.dribbling.mdp import BallCommandStage, BallVelocityCommandCfg
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import (
  booster_k1_flat_env_cfg,
)
from mjlab.utils.spec_config import CollisionCfg

COMMAND_NAME = "ball_velocity"

# Size 3 ball, as on tumminello/k1-goalkeeper-task (Humanoid League).
BALL_RADIUS = 0.095
BALL_MASS = 0.33

# JOGA Table VIII, ball-ground. Rolling friction is a length in MuJoCo, so
# their turf value (0.02 for a 0.11 m ball) is scaled down for this ball and
# bracketed between carpet (0.007, measured on the goalkeeper branch) and turf.
BALL_FRICTION = (0.75, 0.05, 0.012)
BALL_SOLREF = (0.055, 0.18)
BALL_SOLIMP = (0.85, 0.95, 0.001, 0.5, 2.0)
FOOT_BALL_FRICTION = 0.3
FOOT_BALL_SOLREF = (0.02, 0.14)

# Head camera model. The optical frame is taken as the Head_2 body frame
# (x forward) pushed out to the front of the head.
CAMERA_BODY = "Head_2"
CAMERA_OFFSET = (0.06, 0.0, 0.07)
CAMERA_HFOV_DEG = 90.0
CAMERA_VFOV_DEG = 60.0
HEAD_PITCH_DEFAULT = 0.5

HISTORY_LENGTH = 3

BALL_VELOCITY_STD = 0.4
BALL_ZONE_STD = 0.3

_STEPS_PER_ITER = 24

# JOGA Table VII ramps turn frequency at 10k/14k/18k iterations with 16,384
# envs. With a quarter of the envs per iteration on one GPU the stages are
# kept at the same iteration counts; the speed ramp is ours (the K1 is about
# two thirds the height of a G1 and tops out near 1.5 m/s).
BALL_COMMAND_STAGES: list[BallCommandStage] = [
  {"step": 0, "speed_range": (0.4, 0.9), "resampling_time_range": (7.0, 7.0)},
  {"step": 3_000 * _STEPS_PER_ITER, "speed_range": (0.4, 1.3)},
  {"step": 4_500 * _STEPS_PER_ITER, "speed_range": (0.4, 1.7)},
  {"step": 6_500 * _STEPS_PER_ITER, "speed_range": (0.5, 2.0)},
  {"step": 10_000 * _STEPS_PER_ITER, "resampling_time_range": (5.0, 7.0)},
  {"step": 14_000 * _STEPS_PER_ITER, "resampling_time_range": (3.0, 7.0)},
  {"step": 18_000 * _STEPS_PER_ITER, "resampling_time_range": (2.0, 8.0)},
]

MAX_BALL_SPEED = 2.0
"""Top of the command ramp. 2.0 m/s is the K1 velocity task's own ceiling,
which that task's notes say the robot does not quite reach while walking."""

_FOOT_REGEX = r".*foot_collision$"
_LEG_REGEX = r".*(shin|knee)_collision$"
_ARM_REGEX = r".*(hand|upper_arm)_collision$"
_TRUNK_REGEX = r".*(torso|pelvis|hip|thigh)_collision$"

# Collision groups, by contype / conaffinity bit:
# - Feet (1 / 1): the ground, each other and the ball, as in the velocity task.
# - Shins and knees (2 / 0): the ball only, via the ball's conaffinity, so the
#   ball cannot pass through a leg but the legs add no ground contacts.
# - Arms (4 / 0) against torso, pelvis, hips and thighs (0 / 4): each other
#   only, so the arms cannot swing through the body. The shoulder geoms sit
#   against the torso by construction and are left out.
DRIBBLE_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX, _LEG_REGEX, _ARM_REGEX, _TRUNK_REGEX),
  contype={_FOOT_REGEX: 1, _LEG_REGEX: 2, _ARM_REGEX: 4, _TRUNK_REGEX: 0},
  conaffinity={_FOOT_REGEX: 1, _LEG_REGEX: 0, _ARM_REGEX: 0, _TRUNK_REGEX: 4},
  condim=3,
  priority={_FOOT_REGEX: 1, _LEG_REGEX: 0, _ARM_REGEX: 0, _TRUNK_REGEX: 0},
  friction={
    _FOOT_REGEX: (1.0,),
    _LEG_REGEX: (0.6,),
    _ARM_REGEX: (0.6,),
    _TRUNK_REGEX: (0.6,),
  },
)


def get_ball_spec() -> mujoco.MjSpec:
  """A football: a sphere on a free joint with rolling resistance."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, BALL_RADIUS))
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS, 0.0, 0.0),
    mass=BALL_MASS,
    rgba=(0.95, 0.55, 0.1, 1.0),
    contype=1,
    conaffinity=3,
    # condim 6 so torsional and rolling friction apply.
    condim=6,
    # The ball takes contact priority, so its parameters are not mixed with
    # the ground's or a leg's (JOGA Appendix D.3).
    priority=2,
    friction=BALL_FRICTION,
    solref=BALL_SOLREF,
    solimp=BALL_SOLIMP,
  )
  return spec


def _add_foot_ball_pairs(spec: mujoco.MjSpec) -> None:
  """Explicit foot-ball pairs with lower sliding friction.

  JOGA Appendix D.3: with the ball's own (ground) friction a stepped-on ball
  grips the sole; lowering it lets the ball roll out from under the foot as
  it does on hardware.
  """
  for side in ("left", "right"):
    spec.add_pair(
      name=f"{side}_foot_ball",
      geomname1=f"robot/{side}_foot_collision",
      geomname2="ball/ball_geom",
      condim=3,
      friction=(FOOT_BALL_FRICTION, FOOT_BALL_FRICTION, 0.005, 1e-4, 1e-4),
      solref=FOOT_BALL_SOLREF,
    )


def _camera_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=(CAMERA_BODY,))


def _ball_camera_term(corrupt: bool) -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.ball_camera_observation,
    params={
      "camera_cfg": _camera_cfg(),
      "camera_offset": CAMERA_OFFSET,
      "hfov_deg": CAMERA_HFOV_DEG,
      "vfov_deg": CAMERA_VFOV_DEG,
      "corrupt": corrupt,
    },
    # Detection latency of 0-2 policy steps (JOGA Table VI).
    delay_min_lag=0,
    delay_max_lag=2 if corrupt else 0,
  )


def booster_k1_dribble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster K1 flat-ground dribbling configuration."""
  cfg = booster_k1_flat_env_cfg(play=play)

  ##
  # Scene: robot with leg-ball collisions, plus the ball.
  ##

  robot_cfg = get_k1_robot_cfg()
  robot_cfg.collisions = (DRIBBLE_COLLISION,)
  # Default gaze tilted down at the pitch. With the head level, a ball within
  # a metre of the feet sits below the camera's view, and the policy would
  # start every episode blind. Action zero and the pose reward both centre
  # on this pose.
  robot_cfg.init_state = replace(
    robot_cfg.init_state,
    joint_pos={
      **(robot_cfg.init_state.joint_pos or {}),
      "Head_pitch": HEAD_PITCH_DEFAULT,
    },
  )
  cfg.scene.entities = {
    "robot": robot_cfg,
    "ball": EntityCfg(
      spec_fn=get_ball_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.6, 0.0, BALL_RADIUS)),
    ),
  }
  cfg.scene.spec_fn = _add_foot_ball_pairs
  cfg.sim.njmax = 500

  # Every robot-robot contact (arm on body, foot on foot), for the penalty.
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision,)

  ##
  # Command.
  ##

  cfg.commands = {
    COMMAND_NAME: BallVelocityCommandCfg(
      resampling_time_range=(7.0, 7.0),
      speed_range=(0.4, 0.9),
      # Episodes open with a command within +/-60 deg of the robot's heading;
      # every later command is uniform over the circle.
      initial_heading_range=(-math.pi / 3, math.pi / 3),
      rel_standing_envs=0.05,
      debug_vis=True,
    )
  }

  ##
  # Observations.
  ##

  all_joints = SceneEntityCfg("robot")
  cfg.observations.pop("history", None)
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  for group in (actor, critic):
    # The policy drives the head, so it observes it too.
    for term_name in ("joint_pos", "joint_vel"):
      group.terms[term_name].params["asset_cfg"] = all_joints
    group.terms["command"] = ObservationTermCfg(
      func=mdp.ball_command_b, params={"command_name": COMMAND_NAME}
    )
    group.history_length = HISTORY_LENGTH
    group.flatten_history_dim = True
  actor.terms["ball"] = _ball_camera_term(corrupt=True)
  critic.terms["ball"] = _ball_camera_term(corrupt=False)
  critic.terms["ball_pos"] = ObservationTermCfg(func=mdp.ball_pos_yaw_frame)
  critic.terms["ball_vel"] = ObservationTermCfg(func=mdp.ball_vel_yaw_frame)

  ##
  # Actions: all 22 joints, head included.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (".*",)
  joint_pos_action.scale = dict(K1_ACTION_SCALE)

  ##
  # Events.
  ##

  cfg.events.pop("reset_base", None)
  cfg.events.pop("command_drop", None)
  cfg.events = {
    "reset_robot_and_ball": EventTermCfg(
      func=mdp.reset_robot_and_ball,
      mode="reset",
      params={
        "ball_distance_range": (0.4, 1.0),
        "ball_bearing_range": (-0.4, 0.4),
        "ball_radius": BALL_RADIUS,
      },
    ),
    **cfg.events,
  }
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*",)
  )
  # JOGA Table VIII.
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.2)
  cfg.events["ball_friction"] = EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
      "operation": "abs",
      "axes": [0, 1, 2],
      "ranges": {0: (0.5, 0.9), 1: (0.01, 0.1), 2: (0.007, 0.02)},
    },
  )

  ##
  # Rewards (JOGA Table V, dribbling column).
  ##

  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "track_linear_velocity_attainment",
    "track_angular_velocity_attainment",
    "air_time",
    "gait_phase_regularity",
    "foot_flat",
    "feet_distance",
    "action_acc_l2",
  ):
    cfg.rewards.pop(name, None)
  for term in cfg.rewards.values():
    if term.params.get("command_name") is not None:
      term.params["command_name"] = COMMAND_NAME

  cfg.rewards["ball_in_zone"] = RewardTermCfg(
    func=mdp.ball_in_zone,
    weight=1.5,
    # JOGA's kernel is 1 m wide. With it the second run parked 0.42 m from
    # the ball at 97% of this term and never made contact, so never found the
    # ball-velocity rewards. At 0.3 m, stopping short costs a quarter of it.
    params={"zone": (0.25, 0.0), "std": BALL_ZONE_STD},
  )
  cfg.rewards["ball_velocity_match"] = RewardTermCfg(
    func=mdp.ball_velocity_match,
    weight=2.0,
    # JOGA's kernel has std 1 m/s, sized for G1 commands of 1.5-2.5 m/s. At K1
    # speeds a ball left standing still scored ~72% of this term, and the
    # first run converged on standing beside the ball. Scaled to K1 commands,
    # a resting ball earns under 10% at the mean command speed.
    params={"command_name": COMMAND_NAME, "std": BALL_VELOCITY_STD},
  )
  cfg.rewards["ball_velocity_along_command"] = RewardTermCfg(
    func=mdp.ball_velocity_along_command,
    weight=1.5,
    params={"command_name": COMMAND_NAME, "max_speed": MAX_BALL_SPEED},
  )
  cfg.rewards["face_ball"] = RewardTermCfg(func=mdp.face_ball, weight=1.0)
  # Our addition, not in JOGA; see the term's docstring. Runs 1-3 all ended
  # standing beside a motionless ball.
  cfg.rewards["drive_ball_along_command"] = RewardTermCfg(
    func=mdp.drive_ball_along_command,
    weight=1.5,
    params={"command_name": COMMAND_NAME},
  )

  cfg.rewards["upright"].weight = 1.0
  pose = cfg.rewards["pose"]
  pose.weight = 1.0
  pose.params["asset_cfg"].joint_names = (".*",)
  # The head is left free to look around at every speed.
  head = r".*[Hh]ead.*"
  body = r".*(Hip|Knee|Ankle|Shoulder|Elbow).*"
  pose.params["std_standing"] = {body: 0.05, head: 0.6}
  pose.params["std_walking"] = {
    r".*Hip_Pitch.*": 0.3,
    r".*Hip_Roll.*": 0.15,
    r".*Hip_Yaw.*": 0.2,
    r".*Knee.*": 0.35,
    r".*Ankle_Pitch.*": 0.25,
    r".*Ankle_Roll.*": 0.1,
    r".*Shoulder_Pitch.*": 0.15,
    r".*Shoulder_Roll.*": 0.15,
    r".*Elbow.*": 0.15,
    head: 0.6,
  }
  pose.params["std_running"] = {
    r".*Hip_Pitch.*": 0.5,
    r".*Hip_Roll.*": 0.2,
    r".*Hip_Yaw.*": 0.25,
    r".*Knee.*": 0.6,
    r".*Ankle_Pitch.*": 0.35,
    r".*Ankle_Roll.*": 0.15,
    r".*Shoulder_Pitch.*": 0.5,
    r".*Shoulder_Roll.*": 0.2,
    r".*Elbow.*": 0.35,
    head: 0.6,
  }
  pose.params["walking_threshold"] = 0.05
  pose.params["running_threshold"] = 0.8

  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot")
  clearance = cfg.rewards["foot_clearance"]
  clearance.weight = -2.0
  clearance.params["power"] = 1
  clearance.params["only_below"] = False
  clearance.params["target_height"] = 0.08
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["soft_landing"].weight = -1e-5
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  cfg.rewards["feet_too_close"] = RewardTermCfg(
    func=mdp.feet_too_close,
    weight=-1.0,
    params={
      "min_distance": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )

  ##
  # Terminations, curriculum, metrics.
  ##

  cfg.terminations["ball_out_of_reach"] = TerminationTermCfg(
    func=mdp.ball_out_of_reach, params={"max_distance": 3.0}, time_out=True
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum, params={"minimum_height": 0.3}
  )

  cfg.curriculum.pop("command_vel", None)
  if not play:
    cfg.curriculum["ball_command"] = CurriculumTermCfg(
      func=mdp.ball_command_stages,
      params={"command_name": COMMAND_NAME, "stages": BALL_COMMAND_STAGES},
    )

  cfg.metrics["ball_visible"] = MetricsTermCfg(
    func=mdp.ball_visible_fraction,
    params={
      "camera_cfg": _camera_cfg(),
      "camera_offset": CAMERA_OFFSET,
      "hfov_deg": CAMERA_HFOV_DEG,
      "vfov_deg": CAMERA_VFOV_DEG,
    },
  )

  if play:
    actor.terms["ball"] = _ball_camera_term(corrupt=False)
    command = cfg.commands[COMMAND_NAME]
    assert isinstance(command, BallVelocityCommandCfg)
    command.speed_range = (0.5, MAX_BALL_SPEED)
    command.resampling_time_range = (4.0, 7.0)
    command.rel_standing_envs = 0.0
    cfg.terminations.pop("ball_out_of_reach", None)

  return cfg
