"""NUbots Nugus velocity environment confiurations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.asset_zoo.robots import (
  NUGUS_ACTION_SCALE,
  NUGUS_MOTOR_JOINT_REGEX,
  get_nugus_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
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
from mjlab.tasks.velocity.config.nugus.rl_cfg import nubots_nugus_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_NUM_STEPS_PER_ENV = 24
_DEFAULT_MAX_ITERATIONS = nubots_nugus_ppo_runner_cfg().max_iterations
_DEFAULT_GAIT_PERIOD = 0.7
_DEFAULT_JOULE_W = -3e-4
_DEFAULT_PHASE_C_FRAC = 0.6
_DEFAULT_STAND_W = -0.15
_DEFAULT_EFFORT_LO = 0.7
_DEFAULT_EFFORT_HI = 1.2
# Per-servo torque constants (Nm/A) for the actuator-current observation.
# XH540-W270 (legs) and MX106 (hip yaw) are ~2.0 Nm/A; the smaller MX64 arm/head
# servos are approximated lower. Matched by regex against actuator names.
_NUGUS_CURRENT_KT: dict[str, float] = {
  r"(shoulder|elbow|neck|head)": 1.5,
  "default": 2.0,
}
# Dynamixel XH540-W270 "present current" unit: 2.69 mA per LSB.
_CURRENT_QUANTIZE_A = 0.00269
_PHASE_C_JOINT_ACC_W = -1e-4
_PHASE_C_TORQUE_RATE_W = -1e-3
_PHASE_C_SOFT_LANDING_W = -0.01
_PHASE_C_BASE_HEIGHT_W = 0.3


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _phase_steps(max_iterations: int, phase_c_frac: float) -> tuple[int, int, int]:
  """Return P1, P2, P3 curriculum thresholds in common_step_counter units."""
  total = max_iterations * _NUM_STEPS_PER_ENV
  p1 = int(0.25 * total)
  p2 = int(phase_c_frac * total)
  p3 = int(0.85 * total)
  return p1, p2, p3


@requires_model_fields("actuator_forcerange")
def effort_limit_drift(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  drift_factor: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Gradually scale actuator effort limits down over an episode."""
  from mjlab.entity import Entity

  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_names = asset_cfg.joint_names
  if joint_names is None:
    raise ValueError("effort_limit_drift requires joint_names on asset_cfg")

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]
  if not isinstance(actuators, list):
    actuators = [actuators]

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] *= drift_factor
    env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] *= drift_factor


def _phase_c_ramp_stages(
  p2: int, p3: int, peak: float, n_steps: int = 4
) -> list[dict[str, float]]:
  """Build a staged step-function ramp for one Phase-C reward term.

  The reward curriculum (``mdp.reward_curriculum``) is a step function -- it
  applies the last stage whose ``step`` has been reached, with no interpolation.
  To approximate a gradual ramp we emit ``n_steps`` intermediate stages that
  step the weight up from ``peak / n_steps`` at ``p2`` to the full ``peak`` at
  ``p3``, holding ``peak`` thereafter. The weight is zero before ``p2`` so the
  term's onset is tied to ``PHASE_C_FRAC`` (which sets ``p2``).

  The term's sign is preserved automatically: each stage weight is a positive
  fraction of ``peak`` (negative for penalties, positive for ``base_height``).
  Stage steps are integers and clamped to be nondecreasing so they satisfy the
  ``_validate_stages`` ordering check.
  """
  stages: list[dict[str, float]] = [{"step": 0, "weight": 0.0}]
  for i in range(n_steps):
    frac = (i + 1) / n_steps
    step = p2 if n_steps == 1 else int(round(p2 + (p3 - p2) * i / (n_steps - 1)))
    step = max(step, int(stages[-1]["step"]))
    stages.append({"step": step, "weight": peak * frac})
  return stages


def _add_phase_c_curriculum(
  cfg: ManagerBasedRlEnvCfg,
  *,
  p2: int,
  p3: int,
  joule_w: float,
) -> None:
  """Anneal smoothness + energy penalties in during Phase C."""
  for reward_name, peak_weight in (
    ("joule_heating", joule_w),
    ("joint_acc_l2", _PHASE_C_JOINT_ACC_W),
    ("torque_rate", _PHASE_C_TORQUE_RATE_W),
    ("soft_landing", _PHASE_C_SOFT_LANDING_W),
    ("base_height", _PHASE_C_BASE_HEIGHT_W),
  ):
    cfg.curriculum[f"{reward_name}_rampup"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": reward_name,
        "stages": _phase_c_ramp_stages(p2, p3, peak_weight),
      },
    )


def _add_gait_curriculum(
  cfg: ManagerBasedRlEnvCfg,
  *,
  variant: Literal["clock_anneal", "self_paced", "clock_persist"],
  p1: int,
  p2: int,
  p3: int,
) -> None:
  """Configure staged gait handoff curriculums for the selected variant."""
  if variant == "clock_anneal":
    cfg.curriculum["clock_anneal"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "foot_swing_height",
        "stages": [
          {"step": 0, "weight": 0.75},
          {"step": p1, "weight": 0.4},
          {"step": p2, "weight": 0.1},
          {"step": p3, "weight": 0.0},
        ],
      },
    )
    cfg.curriculum["selfpaced_rampup"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "foot_swing_height_landing",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": p1, "weight": -0.5},
          {"step": p2, "weight": -1.0},
        ],
      },
    )
    cfg.curriculum["air_time_rampup"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "air_time",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": p1, "weight": 0.04},
          {"step": p2, "weight": 0.08},
        ],
      },
    )
  elif variant == "self_paced":
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.curriculum["selfpaced_rampup"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "foot_swing_height_landing",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": p1, "weight": -0.5},
          {"step": p2, "weight": -1.0},
        ],
      },
    )
    cfg.curriculum["air_time_rampup"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "air_time",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": p1, "weight": 0.04},
          {"step": p2, "weight": 0.08},
        ],
      },
    )
  elif variant == "clock_persist":
    cfg.rewards["foot_swing_height"].weight = 0.75
    cfg.rewards["foot_swing_height_landing"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.08


def nubots_nugus_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus rough terrain velocity configuration."""
  variant_raw = os.environ.get("MJLAB_VARIANT", "clock_anneal")
  if variant_raw not in ("clock_anneal", "self_paced", "clock_persist"):
    raise ValueError(
      f"MJLAB_VARIANT must be clock_anneal, self_paced, or clock_persist; got {variant_raw!r}"
    )
  variant: Literal["clock_anneal", "self_paced", "clock_persist"] = variant_raw  # type: ignore[assignment]

  gait_period = _env_float("GAIT_PERIOD", _DEFAULT_GAIT_PERIOD)
  joule_w = _env_float("JOULE_W", _DEFAULT_JOULE_W)
  if joule_w > 0:
    joule_w = -joule_w
  phase_c_frac = _env_float("PHASE_C_FRAC", _DEFAULT_PHASE_C_FRAC)
  stand_w = _env_float("STAND_W", _DEFAULT_STAND_W)
  if stand_w > 0:
    stand_w = -stand_w
  effort_lo = _env_float("EFFORT_LO", _DEFAULT_EFFORT_LO)
  effort_hi = _env_float("EFFORT_HI", _DEFAULT_EFFORT_HI)
  silence_clock = _env_bool("SILENCE_CLOCK", False)
  current_obs = _env_bool("CURRENT_OBS", False)
  max_iterations = _env_int("MAX_ITERATIONS", _DEFAULT_MAX_ITERATIONS)
  # Phase curriculum boundaries are derived from PHASE_ITERATIONS, which is
  # decoupled from MAX_ITERATIONS (the latter only drives the
  # --agent.max-iterations training-length flag). Defaulting PHASE_ITERATIONS to
  # MAX_ITERATIONS preserves the previous behaviour, while setting it explicitly
  # lets a run extend (or resume) past its original length with the phase
  # boundaries frozen at fixed absolute step counts.
  phase_iterations = _env_int("PHASE_ITERATIONS", max_iterations)
  p1, p2, p3 = _phase_steps(phase_iterations, phase_c_frac)

  cfg = make_velocity_env_cfg()

  if "SEED" in os.environ:
    cfg.seed = int(os.environ["SEED"])

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
    "joule_heating",
    "joint_acc_l2",
    "torque_rate",
    "stand_still_pose",
    "stand_still_motion",
  ):
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (
      NUGUS_MOTOR_JOINT_REGEX,
    )
  # joint_pos_limits has no asset_cfg param by default; add one scoped to motors.
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = motor_cfg()
  cfg.rewards["base_height"].params["asset_cfg"].body_names = ("torso",)

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
  joint_pos_action.scale = NUGUS_ACTION_SCALE  # ~0.245 rad (0.25 * e/s * 5).
  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  twist_cmd.rel_stop_envs = 0.5
  twist_cmd.stop_ramp_time = 0.75
  twist_cmd.stop_settle_time = 0.75

  cfg.rewards["stand_still_pose"].weight = stand_w
  cfg.rewards["stand_still_motion"].weight = -0.003

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  # Per-servo strength / stiction DR — re-sample each episode.
  cfg.events["pd_gains"].mode = "reset"
  cfg.events["effort_limits"] = EventTermCfg(
    mode="reset",
    func=dr.effort_limits,
    params={
      "asset_cfg": motor_cfg(),
      "operation": "scale",
      "effort_limit_range": (effort_lo, effort_hi),
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": motor_cfg(),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="reset",
    func=dr.joint_armature,
    params={
      "asset_cfg": motor_cfg(),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["effort_drift"] = EventTermCfg(
    mode="interval",
    interval_range_s=(2.0, 4.0),
    func=effort_limit_drift,
    params={
      "asset_cfg": motor_cfg(),
      "drift_factor": 0.995,
    },
  )

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
  clock_params: dict = {
    "period": gait_period,
    "command_name": "twist",
    "command_threshold": 0.05,
  }
  # Clock-silencing variant: fade the gait-clock OBSERVATION out to zero on the
  # same staged schedule as the clock REWARD anneal (foot_swing_height: 0.75 ->
  # 0.4 -> 0.1 -> 0.0 over 0/p1/p2/p3). The scale mirrors that weight normalized
  # to 1.0 at the start, so by p3 the policy receives a zero clock and cannot
  # depend on a phase signal it would lack on hardware. Only meaningful for the
  # clock_anneal variant (the only one that anneals foot_swing_height).
  if silence_clock and variant == "clock_anneal":
    clock_params["silence_stages"] = [
      {"step": 0, "scale": 1.0},
      {"step": p1, "scale": 0.4 / 0.75},
      {"step": p2, "scale": 0.1 / 0.75},
      {"step": p3, "scale": 0.0},
    ]
  clock_obs = ObservationTermCfg(func=mdp.gait_clock, params=clock_params)
  cfg.observations["actor"].terms["gait_clock"] = clock_obs
  cfg.observations["critic"].terms["gait_clock"] = clock_obs

  # Servo current/torque observation. Estimated electrical current per actuator
  # (tau / Kt, quantized to the Dynamixel present-current resolution) given to
  # the policy AND critic. This grows the actor input dimension, so runs with
  # CURRENT_OBS enabled cannot resume an old checkpoint and must train from
  # scratch. A GaussianNoiseCfg (with a small constant mean bias) models
  # measurement noise/bias, while per-servo gain/offset variation is randomized
  # on reset by the current_sensor event below.
  if current_obs:
    current_term = ObservationTermCfg(
      func=mdp.actuator_current,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kt": _NUGUS_CURRENT_KT,
        "quantize": _CURRENT_QUANTIZE_A,
      },
      noise=Gnoise(mean=0.02, std=0.05),  # Amps: constant bias + sensor noise.
    )
    cfg.observations["actor"].terms["actuator_current"] = current_term
    cfg.observations["critic"].terms["actuator_current"] = current_term
    cfg.events["current_sensor"] = EventTermCfg(
      mode="reset",
      func=dr.current_sensor,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "gain_range": (0.9, 1.1),
        "offset_range": (-0.1, 0.1),
      },
    )
  swing_height = cfg.rewards["foot_swing_height"]
  swing_height.func = mdp.feet_swing_height_clock
  swing_height.weight = 0.75
  swing_height.params = {
    "height_sensor_name": "foot_height_scan",
    "target_height": 0.08,
    "period": gait_period,
    "swing_ratio": 0.45,
    "std": 0.05,
    "profile": "sin",
    "command_name": "twist",
    "command_threshold": 0.05,
  }

  # Self-paced sparse swing-height (peak-at-landing) handoff target.
  landing_height = cfg.rewards["foot_swing_height_landing"]
  landing_height.weight = 0.0
  landing_height.params = {
    "sensor_name": "feet_ground_contact",
    "height_sensor_name": "foot_height_scan",
    "target_height": 0.08,
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
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["actuation_power"].weight = 0.0  # Disable (debugging)
  cfg.rewards["cot_proxy"].weight = -0.00  # Disable (debugging)
  cfg.rewards["gait_phase_regularity"].weight = -0.1
  cfg.rewards["limb_symmetry"].weight = -0.0  # Disable (debugging)
  cfg.rewards["feet_distance"].weight = -0.1
  cfg.rewards["foot_flat"].weight = -0.5  # Encourage flat-footed, level swing.
  cfg.rewards["joule_heating"].weight = 0.0
  cfg.rewards["joint_acc_l2"].weight = 0.0
  cfg.rewards["torque_rate"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["base_height"].weight = 0.0

  _add_gait_curriculum(cfg, variant=variant, p1=p1, p2=p2, p3=p3)
  _add_phase_c_curriculum(cfg, p2=p2, p3=p3, joule_w=joule_w)

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
