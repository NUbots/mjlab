"""Booster K1 velocity environment configurations.

Mirrors the tuned Nugus velocity configuration, adapted to the K1's
official Booster joint/body naming, actuator set, and foot geometry.
"""

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # The deployed policy has no odometry, so it must not observe base linear
  # velocity.
  cfg.observations["actor"].terms.pop("base_lin_vel", None)

  # Remove height_scan observation since terrain_scan sensor isn't configured (TODO)
  if "height_scan" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("height_scan")
  if "height_scan" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms.pop("height_scan")

  # Realistic IMU/encoder noise (reusing the values measured on the Nugus's
  # IMU as a sensible default for a similar hobby-grade sensor stack).
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
  cfg.observations["actor"].terms["joint_pos"].delay_max_lag = 3  # 20-60ms

  cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 3

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

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
  corner_site_names = (
    "left_foot_c0",
    "left_foot_c1",
    "left_foot_c2",
    "left_foot_c3",
    "right_foot_c0",
    "right_foot_c1",
    "right_foot_c2",
    "right_foot_c3",
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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = K1_ACTION_SCALE
  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.2

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  # Rationale for std values (carried over from the Nugus tuning):
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
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
    # Head.
    r".*Head_yaw.*": 0.1,
    r".*Head_pitch.*": 0.1,
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
    # Head.
    r".*Head_yaw.*": 0.15,
    r".*Head_pitch.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
  # Cover the full leg (roll/yaw included, not just sagittal pitch) so the term
  # also penalizes the side-leaning, uneven weight-shift asymmetry that produces
  # a lop-sided walk.
  cfg.rewards["limb_symmetry"].params["asset_cfg"].joint_names = (
    r"^(Left|Right)_(Hip_Yaw|Hip_Roll|Hip_Pitch|Knee_Pitch|Ankle_Pitch|Ankle_Roll)$",
  )
  # The default substitutions are lowercase; the K1 uses Left_/Right_.
  cfg.rewards["limb_symmetry"].params["name_substitutions"] = (
    ("Left_", "Right_"),
    ("Right_", "Left_"),
  )
  cfg.rewards["limb_symmetry"].params["velocity_weight"] = 0.2
  cfg.rewards["limb_symmetry"].params["position_weight"] = 1.0

  cfg.rewards["cot_proxy"].params["speed_floor"] = 0.12
  cfg.rewards["cot_proxy"].params["command_threshold"] = 0.02

  cfg.rewards["gait_phase_regularity"].params["command_threshold"] = 0.02

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # De-game foot_clearance (E5): squared, one-sided (below-target only) error so
  # the gradient grows as the foot sits below target and a high apex is never
  # penalized. Velocity weighting is kept as the swing/stance gate. The squared
  # one-sided form is ~10-15x smaller than the original linear |Δ|, so the
  # weight is bumped to keep the term's magnitude meaningful -- retune against
  # the logged reward value.
  cfg.rewards["foot_clearance"].params["power"] = 2
  cfg.rewards["foot_clearance"].params["only_below"] = True
  cfg.rewards["foot_clearance"].weight = -15.0  # Starting point; tune.

  # Independent gait-clock swing-height tracking (improved E2). A fixed-frequency
  # clock the policy does not control drives a desired per-foot swing arc, so the
  # foot is genuinely penalized for not lifting on schedule (unlike the previous
  # air-time-driven phase, whose target adapted to whatever the foot did). The
  # clock is also fed to the policy as an observation so it can step
  # periodically. ``GAIT_PERIOD`` is the full gait-cycle duration -- a larger
  # value commands a slower cadence, which is the main knob for "larger, slower
  # steps"; ``swing_ratio`` is the swing fraction of each cycle. The obs and
  # reward MUST share ``GAIT_PERIOD``.
  # Booster's K1 walking configs command 1.5-2.4 Hz gaits; 0.6 s sits in that
  # range.
  GAIT_PERIOD = 0.6  # seconds per full gait cycle; raise for a slower gait.
  clock_obs = ObservationTermCfg(
    func=mdp.gait_clock,
    params={
      "period": GAIT_PERIOD,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.observations["actor"].terms["gait_clock"] = clock_obs
  cfg.observations["critic"].terms["gait_clock"] = clock_obs
  swing_height = cfg.rewards["foot_swing_height"]
  swing_height.func = mdp.feet_swing_height_clock
  swing_height.weight = 0.75
  swing_height.params = {
    "height_sensor_name": "foot_height_scan",
    "target_height": 0.08,
    "period": GAIT_PERIOD,
    "swing_ratio": 0.45,
    "std": 0.05,
    "profile": "sin",
    "command_name": "twist",
    "command_threshold": 0.05,
  }

  # Flat-foot shaping: the K1 foot sole is the bottom face of the foot box, so
  # the sole normal is the foot body's local Z axis. Penalizing in-swing tilt
  # keeps the foot level and stops the toe from pitching down and digging into
  # the ground on touchdown.
  cfg.rewards["foot_flat"].params["asset_cfg"].body_names = (
    "left_foot_link",
    "right_foot_link",
  )
  cfg.rewards["foot_flat"].params["sole_normal_axis"] = 2
  cfg.rewards["foot_flat"].params["command_threshold"] = 0.02

  cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_distance"].params["nominal_distance"] = (
    0.192  # keyframe lateral separation (hip spacing)
  )
  cfg.rewards["feet_distance"].params["sharpness"] = 8.0

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.01
  cfg.rewards["air_time"].weight = 0.08
  cfg.rewards["actuation_power"].weight = 0.0  # Disable (debugging)
  cfg.rewards["cot_proxy"].weight = -0.00  # Disable (debugging)
  cfg.rewards["gait_phase_regularity"].weight = -0.1
  cfg.rewards["limb_symmetry"].weight = -0.0  # Disable (debugging)
  cfg.rewards["feet_distance"].weight = -0.1
  cfg.rewards["foot_flat"].weight = -0.5  # Encourage flat-footed, level swing.

  # The K1 walks faster than the base curriculum's ±0.5 m/s cap; widen the
  # staged ranges to its envelope while keeping the easy-to-hard staging.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.1, 0.1),
    },
    {
      "step": 9000 * 24,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 12000 * 24,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.4, 0.4),
      "ang_vel_z": (-1.0, 1.0),
    },
  ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
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

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  # Disable terrain curriculum.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.1, 0.8)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)

  return cfg
