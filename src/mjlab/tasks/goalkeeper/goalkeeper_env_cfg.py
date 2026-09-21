"""Goalkeeper task configuration.

A shot is rolled at the goalie every few seconds and the goalie has to get its body in
the way without leaving its feet. The policy is commanded exactly as the robot's
behaviour system will command it: ``[active, dy, t, v]``, built from a simulated ball
estimate (see ``mdp/shot_command.py``).

Scope of v0: stance and side-steps. There is no reward for reaching with an arm or a
leg, and the posture terms hold the arms near the ready stance, so the policy has to
solve the problem by stepping. Limb extension comes once the measured envelope shows
where stepping alone runs out.

This module is robot-independent; ``config/<robot>/env_cfgs.py`` fills in the robot,
its joints and its tuning.
"""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.goalkeeper import mdp
from mjlab.tasks.goalkeeper.mdp import ShotCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

BALL_RADIUS = 0.095
"""FIFA size 3, as used by the Humanoid League Middle division."""
BALL_MASS = 0.33

ROLLING_DECELERATION = 0.5
"""How fast a rolling ball slows (m/s^2). NUSim's ball and a carpet pitch both show
about this, and the shot command predicts the crossing time with it."""

ROLLING_FRICTION = 0.007
"""MuJoCo rolling friction, which is a length rather than a ratio: the resisting
torque is this times the normal force, so the deceleration it produces depends on the
ball's radius and mass. Measured at 0.52 m/s^2 for this ball; 0.05 gives 3.3."""


def get_ball_spec(
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
  rgba: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
) -> mujoco.MjSpec:
  """A rolling ball: a sphere on a free joint, with rolling resistance."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, radius))
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    rgba=rgba,
    # condim 6 so the torsional and rolling friction below actually apply.
    condim=6,
    friction=(0.7, 0.005, ROLLING_FRICTION),
  )
  return spec


def make_goalkeeper_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base goalkeeper task configuration."""

  ##
  # Sensors
  ##

  ball_contact = ContactSensorCfg(
    name="ball_robot_contact",
    primary=ContactMatch(mode="body", pattern="ball", entity="ball"),
    secondary=ContactMatch(mode="subtree", pattern="", entity="robot"),  # Per-robot.
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "shot": ShotCommandCfg(
      # One shot per cycle: place, pause, kick, and watch it through.
      resampling_time_range=(4.0, 6.0),
      ball_radius=BALL_RADIUS,
      rolling_deceleration=ROLLING_DECELERATION,
      debug_vis=True,
    )
  }

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "shot"},
    ),
  }

  # The critic sees the truth the actor is denied: unbiased joints, and the ball as it
  # really is rather than as the simulated estimate reports it.
  critic_terms = {
    **actor_terms,
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "ball_state": ObservationTermCfg(
      func=mdp.true_ball_state,
      params={"command_name": "shot"},
    ),
    "contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    # The goalie starts on its line, facing the shooter. Depth and heading are the
    # positioning layer's job, so only small errors are trained against.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.2, 0.2),
          "z": (0.01, 0.03),
          "yaw": (-0.15, 0.15),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 10.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (0.0, 0.0),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (0.0, 0.0),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.7, 1.3),
        "shared_random": False,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # The task.
    "line_up": RewardTermCfg(
      func=mdp.approach_crossing,
      # The only term that pays for moving sideways, so it has to outweigh the comfort
      # of standing still. At weight 3 against the stance rewards the first policy
      # simply stood there and blocked whatever arrived at its body.
      weight=8.0,
      params={
        "command_name": "shot",
        "reach": 0.25,
        "sharpness": 3.0,
        "horizon": 1.0,
        "switch_time": 0.35,
      },
    ),
    "close_on_crossing": RewardTermCfg(
      # The gradient that gets it moving at all. line_up is flat past about 0.6 m, so
      # without this a goalie a metre off the ball's line is paid the same for
      # stepping towards it as for standing still.
      func=mdp.close_on_crossing,
      weight=2.0,
      params={"command_name": "shot", "reference_speed": 1.0, "deadband": 0.05},
    ),
    "touched": RewardTermCfg(
      # Getting a touch is progress and keeps the gradient that taught it to step,
      # but it is not the job.
      func=mdp.touched,
      weight=20.0,
      params={"command_name": "shot"},
    ),
    "defused": RewardTermCfg(
      # Pays for the outcome of a touch, from the moment of contact, rather than
      # leaving the policy to infer it from the sparse save at the end.
      func=mdp.defused,
      weight=3.0,
      params={"command_name": "shot"},
    ),
    "saved": RewardTermCfg(
      # The job: a shot that was going in, kept out.
      func=mdp.saved,
      weight=100.0,
      params={"command_name": "shot"},
    ),
    "cleared": RewardTermCfg(
      # And once it is kept out, send it back up the field rather than leave it for
      # the shooter. Paid only on a save, and worth less than one.
      func=mdp.cleared,
      weight=60.0,
      params={"command_name": "shot"},
    ),
    "conceded": RewardTermCfg(
      func=mdp.conceded,
      weight=-100.0,
      params={"command_name": "shot"},
    ),
    # Stay on the line, facing the field. Both are bounded rewards rather than
    # penalties: unbounded ones grow as the robot topples, which pays it to fall.
    "hold_line": RewardTermCfg(
      func=mdp.hold_line,
      weight=1.0,
      params={"std": 0.3},
    ),
    "face_shooter": RewardTermCfg(
      func=mdp.face_shooter,
      weight=1.0,
      params={"std": 0.4},
    ),
    # Ready stance, and stillness when there is nothing to do.
    "posture": RewardTermCfg(
      func=mdp.posture,
      # Held low on purpose: a goalie that is paid well for standing in its default
      # pose will do exactly that. It only has to be enough to keep the stance tidy.
      weight=0.5,
      params={
        "std": 0.35,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Per-robot.
      },
    ),
    "arm_posture": RewardTermCfg(
      # v0 blocks with the body and the feet. Holding the arms in the ready stance
      # keeps the policy from discovering an arm block it is not yet trained or
      # measured for.
      func=mdp.posture,
      weight=0.5,
      params={
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "ready_when_idle": RewardTermCfg(
      func=mdp.ready_when_idle,
      weight=0.25,
      params={"command_name": "shot", "std": 0.3},
    ),
    # Keep it on its feet and the motion clean.
    "upright": RewardTermCfg(
      func=mdp.upright_with_dead_zone,
      # Staying on its feet has to be worth more than the smoothness penalties cost,
      # or the quickest way to stop paying them is to fall over. Leaning is free up to
      # the dead zone, which is where a keeper reaching a wide ball lives; the cost
      # only starts beyond that, well before the angle that ends the episode.
      weight=2.0,
      params={
        "std": math.sqrt(0.2),
        "dead_zone_deg": 25.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-30.0),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.2),
    "foot_slip": RewardTermCfg(
      func=mdp.foot_slip,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"ball": EntityCfg(spec_fn=get_ball_spec)},  # Robot added per-robot.
      sensors=(ball_contact,),
      num_envs=1,
      # Shots start up to 5 m out, so environments need room not to overlap.
      env_spacing=8.0,
      extent=8.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-10.0,
      azimuth=180.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
