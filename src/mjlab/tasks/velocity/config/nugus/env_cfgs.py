"""NUbots Nugus velocity environment confiurations."""

import copy

from mjlab.asset_zoo.robots import (
  NUGUS_ACTION_SCALE,
  NUGUS_MOTOR_JOINT_REGEX,
  get_nugus_robot_cfg,
)
from mjlab.asset_zoo.robots.nugus.nugus_dcmotor import (
  NUGUS_DCMOTOR_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  PhaseDeltaActionCfg,
  ScriptedHeadActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
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

HISTORY_WINDOW = 25
"""Length (in control steps) of the actor observation window fed to the
policy's history encoder (``mjlab.rl.obs_history``). At the 50 Hz policy
rate this is 0.5 s, roughly one gait cycle."""


def nubots_nugus_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # Nugus policy should not observe base linear velocity.
  cfg.observations["actor"].terms.pop("base_lin_vel", None)

  # Remove height_scan observation since terrain_scan sensor isn't configured (TODO)
  if "height_scan" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms.pop("height_scan")
  if "height_scan" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms.pop("height_scan")

  # Override observation sensor noise parameters with more realistic values based on real sensor measurements.
  cfg.observations["actor"].terms["base_ang_vel"].noise = Gnoise(
    mean=0.0, std=(0.02, 0.03, 0.03)
  )  # rads/s stdev for gyroscope noise (measured from real IMU) * 10 for factor of safety.
  cfg.observations["actor"].terms["projected_gravity"].noise = Gnoise(
    mean=0.0, std=(3.9e-03, 4.3e-03, 5.9e-04)
  )  # From measurements of Z component of Htw Rotation matrix (rounded) then * 10 for factor of safety.
  cfg.observations["actor"].terms["joint_pos"].noise = Gnoise(
    mean=0.0, std=0.01
  )  # Came from the motor position units (0.088 deg for the MX series) * factor of safety.
  cfg.observations["actor"].terms["joint_vel"].noise = Gnoise(
    mean=0.0, std=0.05
  )  # Came from the motor velocity units (0.229 rpm for the X series) * factor of safety.

  # Sensor delays
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

  cfg.scene.entities = {"robot": get_nugus_robot_cfg()}

  # Scope joint observations / rewards / events to motor joints only so the
  # passive *_backlash sibling joints (servo gear play) aren't included.
  def motor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))

  for group in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      term = cfg.observations[group].terms.get(term_name)
      if term is not None:
        term.params["asset_cfg"] = motor_cfg()
  cfg.events["reset_robot_joints"].params["asset_cfg"] = motor_cfg()
  for reward_name in ("pose", "actuation_power"):
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (
      NUGUS_MOTOR_JOINT_REGEX,
    )
  # joint_pos_limits has no asset_cfg param by default; add one scoped to motors.
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = motor_cfg()

  # Set raycast sensor frame to Nugus torso.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "torso"

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
      pattern=r"^(left_foot|right_foot)$",
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
  joint_pos_action.scale = NUGUS_ACTION_SCALE  # Note: This is really small (0.05)-> seems to correspond to a less falling over early on in training.
  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.10,
    r".*hip_yaw.*": 0.10,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*elbow.*": 0.15,
    # Head
    r".*neck_yaw.*": 0.1,
    r".*head_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*elbow.*": 0.35,
    # Head
    r".*neck_yaw.*": 0.15,
    r".*head_pitch.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)
  # Cover the full leg (roll/yaw included, not just sagittal pitch) so the term
  # also penalizes the side-leaning, uneven weight-shift asymmetry that produces
  # a lop-sided walk.
  cfg.rewards["limb_symmetry"].params["asset_cfg"].joint_names = (
    r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee_pitch|ankle_pitch|ankle_roll)$",
  )
  cfg.rewards["limb_symmetry"].params["velocity_weight"] = 0.2
  cfg.rewards["limb_symmetry"].params["position_weight"] = 1.0

  cfg.rewards["cot_proxy"].params["asset_cfg"].joint_names = (NUGUS_MOTOR_JOINT_REGEX,)
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
  GAIT_PERIOD = 0.7  # seconds per full gait cycle; raise for a slower gait.
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

  # Flat-foot shaping: the Nugus foot sole is perpendicular to the foot body's
  # local X axis (all four corner sites share the same local-X coord), so the
  # sole normal is axis 0. Penalizing in-swing tilt keeps the foot level and
  # stops the toe from pitching down and digging into the turf on touchdown.
  cfg.rewards["foot_flat"].params["asset_cfg"].body_names = ("left_foot", "right_foot")
  cfg.rewards["foot_flat"].params["sole_normal_axis"] = 0
  cfg.rewards["foot_flat"].params["command_threshold"] = 0.02

  cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_distance"].params["nominal_distance"] = (
    0.2536  # keyframe lateral separation
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


def nubots_nugus_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus flat terrain velocity configuration."""
  cfg = nubots_nugus_rough_env_cfg(play=play)

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
    twist_cmd.ranges.lin_vel_x = (0.1, 0.5)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)

  return cfg


def add_actor_history(
  cfg: ManagerBasedRlEnvCfg, window: int = HISTORY_WINDOW
) -> ManagerBasedRlEnvCfg:
  """Publish a ``"history"`` group: a window of the actor observation stream.

  Shaped ``[num_envs, window, actor_dim]``, encoded by a TCN inside the actor
  model and concatenated onto the current observation. See
  :mod:`mjlab.rl.obs_history`.

  Applied to a *finished* config, never part way through building one: the
  window is built from deep copies of the actor terms, so it only clones the
  final layout. That equality is the deployment contract -- one history frame
  is byte-for-byte the actor observation vector, so the robot keeps a single
  ring buffer of the vector it already builds.

  Corruption follows the actor group, which is what puts noise on the window
  during training and takes it off in play mode. The copies carry their own
  noise and delay state, so in training the window draws an independent
  corruption realization rather than replaying the exact frames the policy saw.
  That is intended: the encoder should read the signal, not memorize one noise
  draw.
  """
  cfg.observations["history"] = ObservationGroupCfg(
    terms={
      name: copy.deepcopy(term)
      for name, term in cfg.observations["actor"].terms.items()
    },
    concatenate_terms=True,
    enable_corruption=cfg.observations["actor"].enable_corruption,
    history_length=window,
    flatten_history_dim=False,
  )
  return cfg


def nubots_nugus_rough_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough terrain, with the actor observation window published."""
  return add_actor_history(nubots_nugus_rough_env_cfg(play=play))


def nubots_nugus_flat_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain, with the actor observation window published."""
  return add_actor_history(nubots_nugus_flat_env_cfg(play=play))


##
# The add-phase-clock "clock_owned" generations (v44, v57).
#
# A checkpoint only loads against the task that builds its observation
# vector, and these runs were trained on a branch with things this task
# does not otherwise have. Both share a policy-owned gait clock: a
# ``phase_delta`` action advances the phase and the observation reports
# where the policy put it. v57 adds a per-actuator current estimate, a
# shared-bus voltage estimate whose sag scales each servo's torque
# authority, a scripted head, and a 25-frame observation window read by a
# TCN inside the actor.
#
# The resulting layouts, both verified against the trained checkpoints:
#
#   v44  actor 72   action 21   plain MLP actor
#   v57  actor 112  action 21   history 25x112, TCN encoder
#
# Deliberately NOT reproduced: the critic group. The evaluation harness
# loads the actor only (``load_cfg={"actor": True}``), so the critic's
# privileged terms -- a height scan and the two domain-randomization
# vectors -- would be ported for nothing. The critic here is whatever the
# base task builds and is never read.
##

_NUGUS_CURRENT_KT: dict[str, float] = {
  r"(shoulder|elbow|neck|head)": 1.5,
  # Back-EMF constant measured by sysid on hardware walking data: pooled
  # 2.68 Nm/A over the XH540 legs. The spec-sheet ~2.0 is the stall-derived
  # effective constant, which bakes in gear losses; hardware present-current
  # reads electrical current, so the observation model uses the electrical
  # constant. Copied from the training branch, where it also sets the
  # torque authority the bus-voltage model scales.
  "default": 2.68,
}
_CURRENT_QUANTIZE_A = 0.00269
"""Dynamixel XH540-W270 "present current" unit: 2.69 mA per LSB."""

_CLOCK_OWNED_HEAD_JOINTS = ("neck_yaw", "head_pitch")
_CLOCK_OWNED_GAIT_PERIOD = 0.7
_V57_PHASE_RAW_MIN = 0.35
_V57_PHASE_RAW_MAX = 2.5
"""v57's phase-delta clamp (PHASE_RAW_MIN/MAX in its manifest).

v44 sets neither, and the action term treats that as unbounded. The clamp
is the only thing that changed in the action between the two commits, so
leaving it off is exactly what v44 trained against -- applying v57's bounds
to v44 would drive its clock through a limiter it never saw.
"""


def add_clock_owned_layout(
  cfg: ManagerBasedRlEnvCfg,
  *,
  head_scripted: bool = False,
  raw_min: float | None = None,
  raw_max: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Add the ``clock_owned`` gait clock and phase-delta action.

  What every generation in that family shares: the policy advances its own
  gait phase through a ``phase_delta`` action and the clock observation
  reports where it put the phase, instead of reading episode time.

  ``head_scripted`` adds the scripted head, which consumes no policy output
  (zero action dims) but drives the head joints on a script. It is inserted
  before ``phase_delta`` so the action dict matches the order the trained
  policy was built with. ``raw_min``/``raw_max`` clamp the phase-delta
  action; leave both ``None`` for a generation that trained unbounded.
  """
  clock = ObservationTermCfg(
    func=mdp.gait_clock,
    params={
      "period": _CLOCK_OWNED_GAIT_PERIOD,
      "command_name": "twist",
      "command_threshold": 0.05,
      "phase_source": "policy",
    },
  )
  # Assigning over the existing key keeps the clock in the position the base
  # task gave it -- last in the actor group -- so anything appended after
  # this lands where the trained vector expects it.
  cfg.observations["actor"].terms["gait_clock"] = clock
  cfg.observations["critic"].terms["gait_clock"] = clock

  if head_scripted:
    cfg.actions["scripted_head"] = ScriptedHeadActionCfg(
      entity_name="robot",
      joint_names=_CLOCK_OWNED_HEAD_JOINTS,
    )
  cfg.actions["phase_delta"] = PhaseDeltaActionCfg(
    entity_name="robot",
    period=_CLOCK_OWNED_GAIT_PERIOD,
    command_name="twist",
    command_threshold=0.05,
    raw_min=raw_min,
    raw_max=raw_max,
  )

  # The action scale follows the plant. These policies trained against the
  # DC-motor actuator, whose effort limits imply a scale about 18% larger
  # than the builtin actuator's; leaving the builtin scale in place would
  # quietly rescale every joint target the policy emits. Pair this with
  # ``--plant dcmotor``, which is the actuator these numbers come from.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = NUGUS_DCMOTOR_ACTION_SCALE
  return cfg


def add_servo_telemetry(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add the per-actuator current and shared-bus voltage observations.

  Appended after the gait clock, which is where the trained vector has
  them. v57 has both; v44 has neither.
  """
  # Estimated per-actuator electrical current (tau / Kt), quantized to the
  # Dynamixel present-current resolution.
  current = ObservationTermCfg(
    func=mdp.actuator_current,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kt": _NUGUS_CURRENT_KT,
      "quantize": _CURRENT_QUANTIZE_A,
    },
    noise=Gnoise(mean=0.02, std=0.25),
  )
  cfg.observations["actor"].terms["actuator_current"] = current
  cfg.observations["critic"].terms["actuator_current"] = current

  # Shared-bus voltage: per-servo supply sags with fleet current and chain
  # position, scaling each servo's torque authority. The step event is the
  # plant model, not a randomization -- the policy was trained against a
  # robot whose torque authority moves with load, so an evaluation that
  # dropped it would measure a different machine.
  cfg.events["bus_voltage"] = EventTermCfg(
    mode="step",
    func=mdp.bus_voltage_step,
    params={"kt": _NUGUS_CURRENT_KT, "asset_cfg": SceneEntityCfg("robot")},
  )
  voltage = ObservationTermCfg(
    func=mdp.servo_voltage,
    params={"kt": _NUGUS_CURRENT_KT},
    noise=Gnoise(mean=0.0, std=0.1),
  )
  cfg.observations["actor"].terms["servo_voltage"] = voltage
  cfg.observations["critic"].terms["servo_voltage"] = voltage
  return cfg


def add_v44_layout(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Rebuild the v44 observation and action layout on a finished config.

  v44 is the ``clock_owned`` family without any of what v57 later added:
  no servo telemetry, no scripted head, no phase-delta clamp, and no
  observation window -- its actor is a plain MLP over 72 dims, and its
  action vector is ``joint_pos(20) + phase_delta(1) = 21``.
  """
  return add_clock_owned_layout(cfg)


def add_v57_layout(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Rebuild the v57 observation and action layout on a finished config.

  Applied before :func:`add_actor_history`, which clones whatever the actor
  group ends up holding -- the history window has to be the same 112 dims
  the policy's TCN was trained on. The action vector is
  ``joint_pos(20) + scripted_head(0) + phase_delta(1) = 21``.
  """
  add_clock_owned_layout(
    cfg,
    head_scripted=True,
    raw_min=_V57_PHASE_RAW_MIN,
    raw_max=_V57_PHASE_RAW_MAX,
  )
  return add_servo_telemetry(cfg)


def nubots_nugus_flat_v44_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain in the v44 layout. No history window: v44 has no encoder."""
  return add_v44_layout(nubots_nugus_flat_env_cfg(play=play))


def nubots_nugus_flat_v57_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain in the v57 layout, with the observation window published."""
  return add_actor_history(add_v57_layout(nubots_nugus_flat_env_cfg(play=play)))
