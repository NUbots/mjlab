"""Booster K1 velocity environment configurations.

A port of the NUgus velocity recipe (same base reward set and weights, same
sensor noise/delay model, same 25-step observation-history actor) to the K1.
No competence tracking or gait clock: the policy steps from the contact-based
swing terms alone.

The head (AAHead_yaw, Head_pitch) is not policy-controlled; on hardware it
belongs to the vision system. Its actuators hold the default pose and the
policy neither observes nor commands it, so the actor observation is
3 (ang vel) + 3 (gravity) + 20 (joint pos) + 20 (joint vel) + 20 (actions)
+ 3 (command) = 69 dims.
"""

import copy

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, VelocityStage
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

HISTORY_WINDOW = 25
"""Length (in control steps) of the actor observation window fed to the
policy's history encoder (``mjlab.rl.obs_history``). At the 50 Hz policy
rate this is 0.5 s, roughly one gait cycle."""

K1_POLICY_JOINT_REGEX = (
  r"^A?(Left|Right)_(Hip_(Pitch|Roll|Yaw)|Knee_Pitch|Ankle_(Pitch|Roll)|"
  r"Shoulder_(Pitch|Roll)|Elbow_(Pitch|Yaw))$"
)
"""Policy-controlled joints: legs + arms, head excluded. Booster's shoulder
pitch joints carry an "A" ordering prefix (ALeft_Shoulder_Pitch), hence the
optional ``A?``."""

_HEAD_ACTION_SCALE_KEYS = ("AAHead_yaw", "Head_pitch")

_STEPS_PER_ITER = 24
"""Env steps per PPO iteration; must match ``num_steps_per_env`` in rl_cfg.
The command curriculum is keyed on env steps, so stages are written in
iterations and converted with this."""

VELOCITY_STAGES: list[VelocityStage] = [
  # Learn to step at walking pace before anything else.
  {
    "step": 0,
    "lin_vel_x": (-0.3, 0.5),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.5, 0.5),
  },
  {
    "step": 1_500 * _STEPS_PER_ITER,
    "lin_vel_x": (-0.5, 0.8),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.8, 0.8),
  },
  {
    "step": 4_000 * _STEPS_PER_ITER,
    "lin_vel_x": (-0.6, 1.0),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-1.0, 1.0),
  },
  # Extend only the forward ceiling; the rest of training refines this
  # final envelope.
  {
    "step": 8_000 * _STEPS_PER_ITER,
    "lin_vel_x": (-0.6, 1.2),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-1.0, 1.0),
  },
]
"""Time-staged command envelope, widened in steps from a gentle walk to the
final range the K1 is expected to track."""


def _policy_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(K1_POLICY_JOINT_REGEX,))


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # The deployed policy has no odometry, so it must not observe base linear
  # velocity. Terrain height is privileged: critic-only.
  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].terms.pop("height_scan", None)

  # Sensor noise, as used for the NUgus (IMU measured, encoders from the
  # position/velocity resolution, both with a safety factor).
  cfg.observations["actor"].terms["base_ang_vel"].noise = Gnoise(
    mean=0.0, std=(0.02, 0.03, 0.03)
  )
  cfg.observations["actor"].terms["projected_gravity"].noise = Gnoise(
    mean=0.0, std=(3.9e-03, 4.3e-03, 5.9e-04)
  )
  cfg.observations["actor"].terms["joint_pos"].noise = Gnoise(mean=0.0, std=0.01)
  cfg.observations["actor"].terms["joint_vel"].noise = Gnoise(mean=0.0, std=0.05)

  # Sensor delays.
  cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 2  # 0-40ms
  cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
  cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 2
  cfg.observations["actor"].terms["joint_pos"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_pos"].delay_max_lag = 3  # 0-60ms
  cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 3

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  # Scope joint observations, the joint reset and joint-space rewards to the
  # policy joints (head excluded).
  for group in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      term = cfg.observations[group].terms.get(term_name)
      if term is not None:
        term.params["asset_cfg"] = _policy_cfg()
  cfg.events["reset_robot_joints"].params["asset_cfg"] = _policy_cfg()
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (K1_POLICY_JOINT_REGEX,)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _policy_cfg()

  # Terms the NUgus recipe runs at weight 0. Zero-weight terms are still
  # constructed, so drop them rather than wire K1 joint names into them.
  for reward_name in ("actuation_power", "cot_proxy", "limb_symmetry"):
    del cfg.rewards[reward_name]

  # Set raycast sensor frame to the K1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "Trunk"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(f"{side}_foot_collision" for side in ("left", "right"))

  # Wire foot height scan to per-foot-corner sites.
  # 4 corners per foot ordered left-foot-first, right-foot-second.
  # group_size=4 reduces each group of 4 corners to one clearance value (the
  # minimum, i.e. the lowest corner), so heights retains shape [B, 2].
  corner_site_names = tuple(
    f"{side}_foot_c{i}" for side in ("left", "right") for i in range(4)
  )
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in corner_site_names
      )
      # Single downward ray per corner — the corners are already spread across
      # the foot geometry, so no ring needed.
      sensor.pattern = RingPatternCfg(rings=(), include_center=True)
      sensor.group_size = 4

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Policy drives legs + arms only; the head actuators hold the default pose.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (K1_POLICY_JOINT_REGEX,)
  joint_pos_action.scale = {
    name: scale
    for name, scale in K1_ACTION_SCALE.items()
    if name not in _HEAD_ACTION_SCALE_KEYS
  }

  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.2

  cfg.curriculum["command_vel"].params["velocity_stages"] = VELOCITY_STAGES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway.
  # - Ankle roll is very tight for balance; ankle pitch looser for clearance.
  # - Shoulders/elbows get moderate freedom for natural arm swing.
  # Running values are ~1.5-2x walking values.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*Hip_Pitch.*": 0.3,
    r".*Hip_Roll.*": 0.10,
    r".*Hip_Yaw.*": 0.10,
    r".*Knee.*": 0.35,
    r".*Ankle_Pitch.*": 0.25,
    r".*Ankle_Roll.*": 0.1,
    # Arms.
    r".*Shoulder_Pitch.*": 0.15,
    r".*Shoulder_Roll.*": 0.15,
    r".*Elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*Hip_Pitch.*": 0.5,
    r".*Hip_Roll.*": 0.2,
    r".*Hip_Yaw.*": 0.2,
    r".*Knee.*": 0.6,
    r".*Ankle_Pitch.*": 0.35,
    r".*Ankle_Roll.*": 0.15,
    # Arms.
    r".*Shoulder_Pitch.*": 0.5,
    r".*Shoulder_Roll.*": 0.2,
    r".*Elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.rewards["gait_phase_regularity"].params["command_threshold"] = 0.02

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Squared, one-sided (below-target only) clearance error so the gradient
  # grows as the foot sits below target and a high apex is never penalized.
  # Velocity weighting is kept as the swing/stance gate.
  cfg.rewards["foot_clearance"].params["power"] = 2
  cfg.rewards["foot_clearance"].params["only_below"] = True
  cfg.rewards["foot_clearance"].weight = -15.0

  # Flat-foot shaping: the K1 sole is the bottom face of the foot box, so the
  # sole normal is the foot body's local Z axis.
  cfg.rewards["foot_flat"].params["asset_cfg"].body_names = (
    "left_foot_link",
    "right_foot_link",
  )
  cfg.rewards["foot_flat"].params["sole_normal_axis"] = 2
  cfg.rewards["foot_flat"].params["command_threshold"] = 0.02

  cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_distance"].params["nominal_distance"] = (
    0.192  # Keyframe lateral separation (hip spacing).
  )
  cfg.rewards["feet_distance"].params["sharpness"] = 8.0

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.01
  cfg.rewards["air_time"].weight = 0.08
  cfg.rewards["gait_phase_regularity"].weight = -0.1
  cfg.rewards["feet_distance"].weight = -0.1
  cfg.rewards["foot_flat"].weight = -0.5

  # Actor observation history (see mjlab.rl.obs_history): a HISTORY_WINDOW-step
  # window of the actor observation stream, shaped [B, T, D], encoded by a TCN
  # inside the actor model and concatenated onto the current observation.
  #
  # This block MUST stay after every actor-term mutation above: the window is
  # built from deep copies of the actor terms, so it only clones the final
  # layout. That equality is the deployment contract — one history frame is
  # byte-for-byte the actor observation vector, so the robot keeps a single
  # ring buffer of the vector it already builds.
  #
  # The copies carry their own noise/delay state, so the window draws an
  # independent corruption realization rather than replaying the exact frames
  # the policy saw. That is intended: the encoder should read the signal, not
  # memorize one noise draw.
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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.observations["history"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # The command curriculum runs on every reset and would overwrite the
    # play command ranges.
    cfg.curriculum.pop("command_vel", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def booster_k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain velocity configuration."""
  cfg = booster_k1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan: drop the raycast sensor and the critic height scan.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Disable terrain curriculum.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.1, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)

  return cfg
