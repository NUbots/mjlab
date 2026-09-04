"""NUbots Nugus velocity environment confiurations."""

import os

from mjlab.asset_zoo.robots import (
  NUGUS_ACTION_SCALE,
  NUGUS_MOTOR_JOINT_REGEX,
  get_nugus_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.curriculums import RewardCurriculumStage
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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

NUGUS_FOCUS = mdp.balanced()
"""What this run is trying to be good at.

Edit here, or override per-run without touching the file:

* ``MJLAB_VELOCITY_FOCUS=<preset>`` picks a different preset entirely --
  see ``mdp.FOCUS_PRESETS`` for the list.
* ``MJLAB_VELOCITY_FOCUS_BALANCE=<0..1>`` slides the chosen preset between
  all-stability (``0``) and all-speed (``1``) without changing its shape,
  which is the knob to sweep when hunting for the middle ground.

Env vars rather than ``--env.…`` overrides because tyro cannot introspect
into a term's ``params`` dict, which is where the focus lands.
"""

_FOCUS_PRESET_ENV_VAR = "MJLAB_VELOCITY_FOCUS"
_FOCUS_BALANCE_ENV_VAR = "MJLAB_VELOCITY_FOCUS_BALANCE"


def _resolve_focus() -> mdp.TrainingFocusCfg:
  """``NUGUS_FOCUS`` with the environment-variable overrides applied."""
  focus = NUGUS_FOCUS
  preset = os.environ.get(_FOCUS_PRESET_ENV_VAR)
  if preset:
    focus = mdp.get_focus_preset(preset)
  balance = os.environ.get(_FOCUS_BALANCE_ENV_VAR)
  if balance:
    focus = focus.with_balance(float(balance))
  return focus


# Peak weights the competence-gated movement penalties ramp toward. Each
# term starts at 0.0 and only advances a stage while the population is
# demonstrably stable, so these are the pressure at full competence, never
# the pressure a fresh policy sees.
_JOULE_HEATING_PEAK_W = -3e-4
_JOINT_ACC_PEAK_W = -1e-4
_TORQUE_RATE_PEAK_W = -1e-3
_SOFT_LANDING_PEAK_W = -1e-5


def _competence_penalty_stages(
  final_weight: float, *, n_steps: int = 4
) -> list[RewardCurriculumStage]:
  """Build competence-gated penalty ramp stages (0 -> full in ``n_steps``)."""
  return [{"step": i, "weight": final_weight * i / n_steps} for i in range(n_steps + 1)]


def _add_competence_tracker_event(cfg: ManagerBasedRlEnvCfg) -> None:
  """Accumulate the per-step competence statistics on every env step."""
  cfg.events["competence_tracker"] = EventTermCfg(
    mode="step",
    func=mdp.competence_tracker_step,
    params={},
  )


def _add_competence_penalty_gating(cfg: ManagerBasedRlEnvCfg) -> None:
  """Ramp movement penalties in only while stability competence holds.

  Applying full penalty pressure to a policy that has not yet learned to
  walk suppresses the motion it needs to learn from; applying it and never
  releasing it lets a policy slide down the penalty gradient with no way
  back. Each term walks a five-stage ladder: promote on demonstrated
  stability, demote when it is badly lost, freeze in between.
  """
  for reward_name, peak_weight in (
    ("joule_heating", _JOULE_HEATING_PEAK_W),
    ("joint_acc_l2", _JOINT_ACC_PEAK_W),
    ("torque_rate", _TORQUE_RATE_PEAK_W),
    ("soft_landing", _SOFT_LANDING_PEAK_W),
  ):
    cfg.curriculum[f"{reward_name}_competence"] = CurriculumTermCfg(
      func=mdp.staged_on_competence,
      params={
        "reward_name": reward_name,
        "stages": _competence_penalty_stages(peak_weight),
      },
    )


def _add_competence_diagnostics(cfg: ManagerBasedRlEnvCfg) -> None:
  """Stratify pushes by cohort and publish the frontier diagnostics.

  Only the low ``PUSH_COHORT_FRAC`` share of env indices is pushed; the
  clean cohort is the uncontaminated tracking-competence baseline (and the
  only source of frontier exposure), and it matches the mostly push-free
  deployment distribution.
  """
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = mdp.push_cohort_by_setting_velocity
  cfg.curriculum["competence_diagnostics"] = CurriculumTermCfg(
    func=mdp.competence_diagnostics,
    params={},
  )


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
  for reward_name in (
    "pose",
    "actuation_power",
    "joint_acc_l2",
    "joule_heating",
    "torque_rate",
  ):
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
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.15}
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

  # Training focus. Must come after every per-robot reward edit above: it
  # wraps the stability terms, and a wrapped term's params are no longer
  # reachable as ``cfg.rewards[name].params``.
  mdp.apply_training_focus(cfg, _resolve_focus(), add_diagnostics=not play)

  # Competence-gated movement penalties plus the frontier diagnostics that
  # explain them. Always on.
  _add_competence_tracker_event(cfg)
  _add_competence_penalty_gating(cfg)
  if not play:
    _add_competence_diagnostics(cfg)

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
