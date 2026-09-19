"""Booster K1 goalkeeper environment configuration.

The K1 goalie shares the velocity task's robot, sensor-noise and delay model, action
scale and 25-step observation history, so a policy trained here deploys through the
same path as the walk. The actor observation is
3 (ang vel) + 3 (gravity) + 20 (joint pos) + 20 (joint vel) + 20 (actions)
+ 4 (block command) = 70 dims, which is the contract ``skill::K1BlockPolicy``
implements.

v0 blocks with the body and the feet only: see ``goalkeeper_env_cfg.py``.
"""

import copy

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.goalkeeper.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from mjlab.tasks.goalkeeper.mdp import ShotCommandCfg
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import (
  HISTORY_WINDOW,
  K1_POLICY_JOINT_REGEX,
)
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

_HEAD_ACTION_SCALE_KEYS = ("AAHead_yaw", "Head_pitch")

K1_ARM_JOINT_REGEX = r"^A?(Left|Right)_(Shoulder_(Pitch|Roll)|Elbow_(Pitch|Yaw))$"
"""Arms only. v0 holds these in the ready stance rather than blocking with them."""


def _policy_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(K1_POLICY_JOINT_REGEX,))


def booster_k1_block_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster K1 goalkeeper (block policy) configuration."""
  cfg = make_goalkeeper_env_cfg()

  cfg.scene.entities = {"robot": get_k1_robot_cfg(), **(cfg.scene.entities or {})}

  # Sensor noise and delays, shared with the velocity task so the two policies see the
  # same robot.
  cfg.observations["actor"].terms["base_ang_vel"].noise = Gnoise(
    mean=0.0, std=(0.02, 0.03, 0.03)
  )
  cfg.observations["actor"].terms["projected_gravity"].noise = Gnoise(
    mean=0.0, std=(3.9e-03, 4.3e-03, 5.9e-04)
  )
  cfg.observations["actor"].terms["joint_pos"].noise = Gnoise(mean=0.0, std=0.01)
  cfg.observations["actor"].terms["joint_vel"].noise = Gnoise(mean=0.0, std=0.05)
  cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 2  # 0-40 ms
  cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 2
  cfg.observations["actor"].terms["joint_pos"].delay_max_lag = 3  # 0-60 ms
  cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 3

  # Head excluded everywhere: on hardware it belongs to the vision system.
  for group in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      term = cfg.observations[group].terms.get(term_name)
      if term is not None:
        term.params["asset_cfg"] = _policy_cfg()
  cfg.events["reset_robot_joints"].params["asset_cfg"] = _policy_cfg()

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (K1_POLICY_JOINT_REGEX,)
  joint_pos_action.scale = {
    name: scale
    for name, scale in K1_ACTION_SCALE.items()
    if name not in _HEAD_ACTION_SCALE_KEYS
  }

  # Feet on the ground, and the ball against any part of the robot.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(left_foot_link|right_foot_link)$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)
  for sensor in cfg.scene.sensors:
    if sensor.name == "ball_robot_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.secondary = ContactMatch(mode="subtree", pattern="Trunk", entity="robot")

  site_names = ("left_foot", "right_foot")
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = tuple(
    f"{side}_foot_collision" for side in ("left", "right")
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["posture"].params["asset_cfg"].joint_names = (K1_POLICY_JOINT_REGEX,)
  cfg.rewards["arm_posture"].params["asset_cfg"].joint_names = (K1_ARM_JOINT_REGEX,)
  cfg.rewards["dof_pos_limits"].params = {"asset_cfg": _policy_cfg()}
  cfg.viewer.body_name = "Trunk"

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.njmax = 300

  # Shot envelope. The reach numbers in PLAN.md say a K1 that may not leave its feet
  # covers roughly +-0.5 m, so v0 trains shots that a step or two can reach, plus some
  # it cannot, which it should at least not fall over chasing.
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  shot.crossing = (-0.8, 0.8)
  shot.speed = (1.5, 4.0)
  shot.distance = (2.0, 4.5)

  # Actor observation history, as in the velocity task: a 25-step window of the exact
  # actor observation vector, encoded by a TCN inside the model. This block must stay
  # after every actor-term change above, because the window clones the final layout;
  # that equality is the deployment contract.
  cfg.observations["history"] = ObservationGroupCfg(
    terms={
      name: copy.deepcopy(term)
      for name, term in cfg.observations["actor"].terms.items()
    },
    concatenate_terms=True,
    enable_corruption=True,
    history_length=HISTORY_WINDOW,
    flatten_history_dim=False,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["history"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
