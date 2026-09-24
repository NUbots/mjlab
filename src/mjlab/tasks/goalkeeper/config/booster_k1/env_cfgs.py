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
from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.asset_zoo.robots.booster_k1.k1_constants import FULL_COLLISION_GND_ONLY
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.goalkeeper import mdp
from mjlab.tasks.goalkeeper.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from mjlab.tasks.goalkeeper.mdp import ShotCommandCfg, ShotLevel
from mjlab.tasks.goalkeeper.mdp.handoff import WalkHandoffActionCfg
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import (
  HISTORY_WINDOW,
  K1_POLICY_JOINT_REGEX,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

_HEAD_ACTION_SCALE_KEYS = ("AAHead_yaw", "Head_pitch")

K1_ARM_JOINT_REGEX = r"^A?(Left|Right)_(Shoulder_(Pitch|Roll)|Elbow_(Pitch|Yaw))$"
"""Arms only. v0 holds these in the ready stance rather than blocking with them."""


SHOT_LEVELS: tuple[ShotLevel, ...] = (
  # Straight at the keeper, slow: the drill is stopping the ball, not moving to it.
  # Run 6 leaves this as the weak spot — 55% of its misses had a foot within 20 cm of
  # the ball and it went in anyway — so this is where the curriculum starts.
  {
    "name": "stop it",
    "crossing": (-0.05, 0.05),
    "speed": (1.5, 2.5),
    "distance": (2.0, 3.0),
  },
  # Then the ball starts arriving to one side, still slowly.
  {
    "name": "one step",
    "crossing": (-0.3, 0.3),
    "speed": (1.5, 3.0),
    "distance": (2.0, 3.5),
  },
  # Then faster and wider.
  {
    "name": "wider",
    "crossing": (-0.55, 0.55),
    "speed": (1.5, 3.5),
    "distance": (2.0, 4.0),
  },
  # And finally everything, including shots it cannot reach on its feet and should not
  # fall over chasing.
  {
    "name": "full",
    "crossing": (-0.8, 0.8),
    "speed": (1.5, 4.0),
    "distance": (2.0, 4.5),
  },
)
"""The drills, easiest first. A level is left behind on save rate, not on a step
count, and later levels keep serving a quarter of their shots from earlier ones."""


def _policy_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(K1_POLICY_JOINT_REGEX,))


def booster_k1_block_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster K1 goalkeeper (block policy) configuration."""
  cfg = make_goalkeeper_env_cfg()

  robot = get_k1_robot_cfg()
  # The K1 ships with only its feet collidable, which is fine for walking and wrong
  # for goalkeeping: the ball passes straight through the shins, knees and body, so
  # the only thing that can ever touch it is a foot. That alone would cap what any
  # reward can teach. Every collision geom is enabled here, against the ground and the
  # ball but not against itself, which keeps the contact count down.
  robot.collisions = (FULL_COLLISION_GND_ONLY,)
  cfg.scene.entities = {"robot": robot, **(cfg.scene.entities or {})}

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

  # A whole collidable robot makes far more contacts than a pair of feet.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.njmax = 600

  # Shot envelope. The reach numbers in PLAN.md say a K1 that may not leave its feet
  # covers roughly +-0.5 m, so v0 trains shots that a step or two can reach, plus some
  # it cannot, which it should at least not fall over chasing.
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  shot.levels = SHOT_LEVELS

  cfg.curriculum["shot_levels"] = CurriculumTermCfg(
    func=mdp.shot_levels,
    params={"command_name": "shot", "advance_at": 0.6, "min_shots": 400},
  )

  # The upright bonus is scaffolding for the early training collapse, not part of the
  # job. It is eased off as the keeper clears drills, so a keeper that can stand is
  # free to lean, dip and reach rather than being held rigidly vertical.
  cfg.curriculum["upright_relaxation"] = CurriculumTermCfg(
    func=mdp.relax_upright,
    params={
      "command_name": "shot",
      "reward_name": "upright",
      "weights": (2.0, 1.0, 0.5, 0.25),
    },
  )

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
    # Play and envelope measurement face the full envelope, not the drill the keeper
    # happens to be on, and without earlier levels mixed in.
    cfg.curriculum.pop("shot_levels", None)
    cfg.curriculum.pop("upright_relaxation", None)
    shot.start_level = len(SHOT_LEVELS) - 1
    shot.mix_fraction = 0.0
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["history"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


WALK_CHECKPOINT = Path(__file__).parent / "assets" / "k1_walk_t98yksya.pt"
"""The walk the goalie is handed over from: velocity-task run t98yksya
(k1-noclock-linvel, model_14999), trained at mjlab 55b975c96. Its actor weights only;
the file records the checkpoint it came from."""

WALK_COMMAND_RANGES = UniformVelocityCommandCfg.Ranges(
  lin_vel_x=(-0.3, 0.8),
  lin_vel_y=(-0.5, 0.5),
  ang_vel_z=(-1.0, 1.0),
)
"""What the goalie's walks ask for: NUbots_K1's PlanWalkPath caps the walk at 0.8 m/s
forward, 0.5 m/s sideways and 1 rad/s, well inside what t98yksya was trained on."""


def _walk_observation_terms() -> dict[str, ObservationTermCfg]:
  """One frame of the walk's actor observation, exactly as t98yksya was trained: the
  velocity task's K1 terms, noise and delays, 72 dims."""
  policy = _policy_cfg()
  return {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Gnoise(mean=0.0, std=(0.05, 0.05, 0.08)),
      delay_max_lag=3,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Gnoise(mean=0.0, std=(0.02, 0.03, 0.03)),
      delay_max_lag=2,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Gnoise(mean=0.0, std=(3.9e-03, 4.3e-03, 5.9e-04)),
      delay_max_lag=2,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": policy},
      noise=Gnoise(mean=0.0, std=0.01),
      delay_max_lag=3,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": copy.deepcopy(policy)},
      noise=Gnoise(mean=0.0, std=0.05),
      delay_max_lag=3,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
  }


def booster_k1_block_moving_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The goalkeeper task with the goalie walking before shots, as on the robot.

  Run 12 was only ever handed the goalie standing in its stance, and on the robot it
  fell taking it over mid-stride. Here a frozen walk drives the goalie through half the
  shot cycles and hands it over as planning::PlanSave does: once the kick has been seen,
  or before it for a ball placed near, and mostly once both feet are down. The block
  policy's observation contract is unchanged, so it fine-tunes from run 12 and deploys
  through skill::K1BlockPolicy as before. See ``mdp/handoff.py``.
  """
  cfg = booster_k1_block_env_cfg(play=play)

  # The walk's velocity command, and its own observation window.
  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 3.0),
    rel_standing_envs=0.1,
    heading_command=False,
    ranges=copy.deepcopy(WALK_COMMAND_RANGES),
    debug_vis=False,
  )
  cfg.observations["walk_history"] = ObservationGroupCfg(
    terms=_walk_observation_terms(),
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=HISTORY_WINDOW,
    flatten_history_dim=False,
  )

  # The walk drives the goalie until the hand-off, through the same joint targets.
  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  cfg.actions["joint_pos"] = WalkHandoffActionCfg(
    entity_name=joint_pos.entity_name,
    actuator_names=joint_pos.actuator_names,
    scale=joint_pos.scale,
    use_default_offset=joint_pos.use_default_offset,
    walk_checkpoint=str(WALK_CHECKPOINT),
  )

  # The block policy sees an inactive command while the walk has the goalie, as the
  # frames K1BlockPolicy records then are built. The critic keeps the real one.
  for group in ("actor", "history"):
    terms = cfg.observations[group].terms
    terms["command"] = replace(terms["command"], func=mdp.block_command)

  # The learner is only paid, or charged, for what it does itself.
  for term in cfg.rewards.values():
    term.func = mdp.while_blocking(term.func)

  # Fine-tuning run 12, which cleared every drill: stay on the full envelope.
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  shot.start_level = len(SHOT_LEVELS) - 1

  return cfg
