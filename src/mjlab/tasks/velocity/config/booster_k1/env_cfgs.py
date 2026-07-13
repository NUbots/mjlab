"""Booster K1 velocity environment configurations.

A competence-first port of the Nugus velocity task. The adaptive machinery
(competence tracker, adaptive command/push levels, competence-gated movement
penalties, track-reward watchdog, diagnostics) is ON by default here, unlike
the Nugus config where each piece is opt-in via env vars.

Deliberately omitted relative to the Nugus config: RMA / observation history
groups, actuator-current and bus-voltage observations, backlash joint scoping,
scripted head, gait-clock variants (the clock is fixed), and the mirror
augmentation map. Those exist to close the Nugus's sim-to-real gap; the K1's
industrial actuators and sensors should not need them.

The head (AAHead_yaw, Head_pitch) is NOT policy-controlled: on hardware it
belongs to the vision system. Its actuators hold the default pose and the
policy neither observes nor commands it, so the actor observation is
3 (ang vel) + 3 (gravity) + 3 (command) + 20 (joint pos) + 20 (joint vel)
+ 20 (actions) + 2 (gait clock) = 71 dims.

The gait clock follows Booster's own K1/T1 recipe (booster_gym feeds
cos/sin(2*pi*f*t) to the policy, zeroed at standstill, with a phase-windowed
feet_swing reward), so it is kept as a permanent observation rather than
annealed away.
"""

from __future__ import annotations

import os

from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
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
from mjlab.tasks.velocity.config.nugus.dr_observations import dr_ratios
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

# Policy-controlled joints: legs + arms, head excluded (vision owns it on
# hardware). Booster's shoulder-pitch joints carry an "A" ordering prefix
# (ALeft_Shoulder_Pitch / ARight_Shoulder_Pitch), hence the optional "A?".
K1_POLICY_JOINT_REGEX = (
  r"^A?(Left|Right)_(Hip_(Pitch|Roll|Yaw)|Knee_Pitch|Ankle_(Pitch|Roll)|"
  r"Shoulder_(Pitch|Roll)|Elbow_(Pitch|Yaw))$"
)
_HEAD_ACTION_SCALE_KEYS = ("AAHead_yaw", "Head_pitch")
_K1_LINK_MASS_BODY_REGEX = (
  r"^(Trunk|(Left|Right)_(Arm_[1-4]|Hip_(Pitch|Roll|Yaw)|Shank)|"
  r"(left|right)_foot_link)$"
)

_DEFAULT_GAIT_PERIOD = 0.6  # Booster's K1 configs command 1.5-2.4 Hz gaits.
_DEFAULT_SWING_TARGET_HEIGHT = 0.08
_TRUNK_STAND_HEIGHT = 0.5435  # Bent-knee keyframe trunk height.
# Movement-penalty peaks reached via the competence-gated ramp. The K1's leg
# torques are ~10x the Nugus's, and joule/torque_rate are quadratic in torque,
# so the Nugus peaks would crush the gait here; these start ~1-2 orders softer
# and are env-tunable.
_DEFAULT_JOULE_W = -1e-5
_DEFAULT_JOINT_ACC_W = -1e-4
_DEFAULT_TORQUE_RATE_W = -1e-4
_DEFAULT_SOFT_LANDING_W = -0.01
_BASE_HEIGHT_W = 0.3
_DEFAULT_STAND_W = -0.15
_DEFAULT_UPRIGHT_W = 1.0
_DEFAULT_RESAMPLE_MIN = 3.0
_DEFAULT_FOOT_FLAT_W = -0.5


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw in (None, "") else float(raw)


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  return default if raw in (None, "") else int(raw)


def _env_bool(name: str, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw in (None, ""):
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str) -> str:
  raw = os.environ.get(name)
  return default if raw in (None, "") else raw.strip()


def _policy_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(K1_POLICY_JOINT_REGEX,))


def _competence_threshold_params() -> dict[str, float | int]:
  return {
    "promote_track_err": _env_float("COMPETENCE_PROMOTE_TRACK_ERR", 0.25),
    "demote_track_err": _env_float("COMPETENCE_DEMOTE_TRACK_ERR", 0.45),
    "promote_attain": _env_float("COMPETENCE_PROMOTE_ATTAIN", 0.75),
    "demote_attain": _env_float("COMPETENCE_DEMOTE_ATTAIN", 0.5),
    "promote_wobble": _env_float("COMPETENCE_PROMOTE_WOBBLE", 0.10),
    "demote_wobble": _env_float("COMPETENCE_DEMOTE_WOBBLE", 0.25),
    "promote_fell": _env_float("COMPETENCE_PROMOTE_FELL", 0.3),
    "demote_fell": _env_float("COMPETENCE_DEMOTE_FELL", 0.35),
    "cooldown_iters": _env_int("COMPETENCE_COOLDOWN_ITERS", 150),
    "demote_fast_fell": _env_float("COMPETENCE_DEMOTE_FAST_FELL", 0.5),
    "top_streak_required": _env_int("COMPETENCE_TOP_STREAK", 5),
  }


def _competence_penalty_stages(
  final_weight: float, *, n_steps: int = 4
) -> list[dict[str, float]]:
  """Build competence-gated penalty ramp stages (0 -> full in ``n_steps``)."""
  stages: list[dict[str, float]] = [{"step": 0, "weight": 0.0}]
  for i in range(1, n_steps + 1):
    stages.append({"step": i, "weight": final_weight * i / n_steps})
  return stages


def _add_competence_tracker_event(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events["competence_tracker"] = EventTermCfg(
    mode="step",
    func=mdp.competence_tracker_step,
    params={},
  )


def _add_adaptive_command_curriculum(cfg: ManagerBasedRlEnvCfg, *, l_max: int) -> None:
  cfg.curriculum["adaptive_command_level"] = CurriculumTermCfg(
    func=mdp.adaptive_command_level,
    params={
      "command_name": "twist",
      "l_max": l_max,
      **_competence_threshold_params(),
    },
  )
  if "command_vel" in cfg.curriculum:
    del cfg.curriculum["command_vel"]


def _add_adaptive_push_curriculum(cfg: ManagerBasedRlEnvCfg, *, l_max: int) -> None:
  cfg.curriculum["adaptive_push_level"] = CurriculumTermCfg(
    func=mdp.adaptive_push_level,
    params={
      "event_name": "push_robot",
      "l_max": l_max,
      "start_level": 1,
      **_competence_threshold_params(),
    },
  )


def _add_aimd_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Continuous TCP-style difficulty: one scalar, no levels."""
  cfg.curriculum["aimd_difficulty"] = CurriculumTermCfg(
    func=mdp.aimd_difficulty,
    params={
      "command_name": "twist",
      "event_name": "push_robot",
      "alpha": _env_float("AIMD_ALPHA", 0.002),
      "beta": _env_float("AIMD_BETA", 0.7),
      "congest_bar": _env_float("AIMD_CONGEST_BAR", 0.35),
      "emergency_bar": _env_float("AIMD_EMERGENCY_BAR", 0.55),
      "gate_attain": _env_float("AIMD_GATE_ATTAIN", 0.40),
      "beta_arrest": _env_float("AIMD_BETA_ARREST", 0.93),
      "envelope_scale": _env_float("AIMD_ENVELOPE_SCALE", 1.0),
      "push_congest_bar": _env_float("AIMD_PUSH_CONGEST_BAR", 0.30),
      "attain_slide_frac": _env_float("AIMD_ATTAIN_SLIDE_FRAC", 0.95),
      "landing_anneal": _env_bool("LANDING_ANNEAL", default=False),
      "attain_band_hi": _env_float("AIMD_ATTAIN_BAND_HI", 0.66),
      "attain_band_lo": _env_float("AIMD_ATTAIN_BAND_LO", 0.60),
      "extend_attain_bar": _env_float("AIMD_EXTEND_BAR", 0.80),
      "floor_frac": _env_float("AIMD_FLOOR_FRAC", 0.95),
      "frontier_headroom": _env_float("AIMD_FRONTIER_HEADROOM", 1.15),
      "push_survival_bar": _env_float("PUSH_SURVIVAL_BAR", 0.85),
      "push_gate_excess": _env_float("AIMD_PUSH_GATE_EXCESS", 0.15),
    },
  )
  if "command_vel" in cfg.curriculum:
    del cfg.curriculum["command_vel"]


def _add_competence_penalty_gating(
  cfg: ManagerBasedRlEnvCfg,
  *,
  joule_w: float,
  joint_acc_w: float,
  torque_rate_w: float,
  soft_landing_w: float,
) -> None:
  """Ramp movement penalties in only while stability competence holds."""
  cfg.rewards["base_height"].weight = _BASE_HEIGHT_W
  for reward_name, peak_weight in (
    ("joule_heating", joule_w),
    ("joint_acc_l2", joint_acc_w),
    ("torque_rate", torque_rate_w),
    ("soft_landing", soft_landing_w),
  ):
    cfg.curriculum[f"{reward_name}_competence"] = CurriculumTermCfg(
      func=mdp.staged_on_competence,
      params={
        "reward_name": reward_name,
        "stages": _competence_penalty_stages(peak_weight),
        **_competence_threshold_params(),
      },
    )


def _add_track_reward_watchdog(cfg: ManagerBasedRlEnvCfg) -> None:
  """Fail fast on rotted policies: armed once tracking reward passes 2.0,
  fires if it sustains below 1.0."""
  cfg.curriculum["track_reward_watchdog"] = CurriculumTermCfg(
    func=mdp.track_reward_watchdog,
    params={
      "reward_name": "track_linear_velocity",
      "arm_above": _env_float("WATCHDOG_ARM_ABOVE", 2.0),
      "fail_below": _env_float("WATCHDOG_FAIL_BELOW", 1.0),
      "fail_persist_iters": _env_int("WATCHDOG_PERSIST_ITERS", 60),
    },
  )


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity configuration."""
  gait_period = _env_float("GAIT_PERIOD", _DEFAULT_GAIT_PERIOD)
  swing_target_height = _env_float("SWING_TARGET_HEIGHT", _DEFAULT_SWING_TARGET_HEIGHT)
  track_lin_w = _env_float("TRACK_LIN_W", 2.0)
  track_ang_w = _env_float("TRACK_ANG_W", 2.0)
  alive_w = _env_float("ALIVE_W", 0.0)
  joule_w = -abs(_env_float("JOULE_W", _DEFAULT_JOULE_W))
  joint_acc_w = -abs(_env_float("JOINT_ACC_W", _DEFAULT_JOINT_ACC_W))
  torque_rate_w = -abs(_env_float("TORQUE_RATE_W", _DEFAULT_TORQUE_RATE_W))
  soft_landing_w = -abs(_env_float("SOFT_LANDING_W", _DEFAULT_SOFT_LANDING_W))
  stand_w = -abs(_env_float("STAND_W", _DEFAULT_STAND_W))
  upright_w = _env_float("UPRIGHT_W", _DEFAULT_UPRIGHT_W)
  foot_flat_w = -abs(_env_float("FOOT_FLAT_W", _DEFAULT_FOOT_FLAT_W))
  resample_min = _env_float("RESAMPLE_MIN", _DEFAULT_RESAMPLE_MIN)
  effort_lo = _env_float("EFFORT_LO", 0.8)
  effort_hi = _env_float("EFFORT_HI", 1.2)
  link_mass_scale_min = _env_float("LINK_MASS_SCALE_MIN", 0.85)
  link_mass_scale_max = _env_float("LINK_MASS_SCALE_MAX", 1.15)
  payload_kg_min = _env_float("PAYLOAD_KG_MIN", -0.3)
  payload_kg_max = _env_float("PAYLOAD_KG_MAX", 0.5)
  push_interval_scale = _env_float("PUSH_INTERVAL_SCALE", 1.0)
  if push_interval_scale <= 0:
    raise ValueError(
      f"PUSH_INTERVAL_SCALE must be positive; got {push_interval_scale!r}"
    )
  # Competence machinery: all ON by default for the K1.
  adaptive_commands = _env_bool("ADAPTIVE_COMMANDS", default=True)
  adaptive_pushes = _env_bool("ADAPTIVE_PUSHES", default=True)
  penalty_gate = _env_str("PENALTY_GATE", "competence")
  if penalty_gate not in ("time", "competence"):
    raise ValueError(
      f"PENALTY_GATE must be 'time' or 'competence'; got {penalty_gate!r}"
    )
  curriculum_style = _env_str("CURRICULUM_STYLE", "levels")
  if curriculum_style not in ("levels", "aimd"):
    raise ValueError(
      f"CURRICULUM_STYLE must be 'levels' or 'aimd'; got {curriculum_style!r}"
    )
  # Level 5 tops out at lin_vel_x +/-0.75 m/s, inside the K1's ~1 m/s
  # walking envelope (the Nugus stops at l_max=3 = +/-0.5).
  adaptive_cmd_lmax = _env_int("ADAPTIVE_CMD_LMAX", 5)
  adaptive_push_lmax = _env_int("ADAPTIVE_PUSH_LMAX", 3)
  push_cohort_frac = _env_float("PUSH_COHORT_FRAC", 1.0)
  if not 0.0 < push_cohort_frac <= 1.0:
    raise ValueError(f"PUSH_COHORT_FRAC must be in (0, 1]; got {push_cohort_frac!r}")

  cfg = make_velocity_env_cfg()

  if "SEED" in os.environ:
    cfg.seed = int(os.environ["SEED"])

  # The deployed policy has no odometry, so it must not observe base linear
  # velocity. Terrain height is privileged: critic-only.
  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].terms.pop("height_scan", None)

  # IMU/encoder noise: the K1's sensor stack is industrial-grade, so this
  # keeps moderate bench-style values rather than the Nugus's walking-
  # vibration-calibrated ones.
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

  # Scope joint observations to the policy joints (head excluded).
  for group in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      term = cfg.observations[group].terms.get(term_name)
      if term is not None:
        term.params["asset_cfg"] = _policy_cfg()

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
  twist_cmd.rel_stop_envs = 0.5
  twist_cmd.stop_ramp_time = 0.75
  twist_cmd.stop_settle_time = 0.75
  twist_cmd.resampling_time_range = (
    resample_min,
    twist_cmd.resampling_time_range[1],
  )

  cfg.rewards["stand_still_pose"].weight = stand_w
  cfg.rewards["stand_still_motion"].weight = -0.003
  cfg.rewards["is_alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=alive_w)
  cfg.rewards["track_linear_velocity"].weight = track_lin_w
  cfg.rewards["track_angular_velocity"].weight = track_ang_w

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.events["link_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_K1_LINK_MASS_BODY_REGEX,)),
      "operation": "scale",
      "ranges": (link_mass_scale_min, link_mass_scale_max),
    },
  )
  cfg.events["payload"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      "operation": "add",
      "ranges": (payload_kg_min, payload_kg_max),
    },
  )

  # Per-actuator strength / joint friction DR — re-sample each episode.
  cfg.events["pd_gains"].mode = "reset"
  cfg.events["effort_limits"] = EventTermCfg(
    mode="reset",
    func=dr.effort_limits,
    params={
      "asset_cfg": _policy_cfg(),
      "operation": "scale",
      "effort_limit_range": (effort_lo, effort_hi),
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": _policy_cfg(),
      "operation": "scale",
      "ranges": (0.5, 1.5),
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="reset",
    func=dr.joint_damping,
    params={
      "asset_cfg": _policy_cfg(),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="reset",
    func=dr.joint_armature,
    params={
      "asset_cfg": _policy_cfg(),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )

  if push_interval_scale != 1.0 and "push_robot" in cfg.events:
    push_event = cfg.events["push_robot"]
    interval = push_event.interval_range_s
    if interval is not None:
      push_lo, push_hi = interval
      push_event.interval_range_s = (
        push_lo * push_interval_scale,
        push_hi * push_interval_scale,
      )

  # Privileged DR realizations for the critic (robot-agnostic term shared
  # with the Nugus config).
  cfg.observations["critic"].terms["dr_ratios"] = ObservationTermCfg(
    func=dr_ratios,
    params={
      "motor_asset_cfg": _policy_cfg(),
      "torso_body_name": "Trunk",
      "foot_geom_names": geom_names,
    },
  )

  # Scope joint-space rewards to the policy joints.
  for reward_name in (
    "pose",
    "actuation_power",
    "joule_heating",
    "joint_acc_l2",
    "torque_rate",
    "stand_still_pose",
    "stand_still_motion",
    "cot_proxy",
  ):
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (K1_POLICY_JOINT_REGEX,)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _policy_cfg()

  # Rationale for std values (carried over from the Nugus tuning):
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

  cfg.rewards["upright"].weight = upright_w
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["base_height"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["base_height"].params["target_height"] = _TRUNK_STAND_HEIGHT

  # Cover the full leg (roll/yaw included) so the term also penalizes
  # side-leaning, uneven weight-shift asymmetry.
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

  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names

  # De-game foot_clearance: squared, one-sided (below-target only) error so
  # the gradient grows as the foot sits below target and a high apex is never
  # penalized.
  cfg.rewards["foot_clearance"].params["power"] = 2
  cfg.rewards["foot_clearance"].params["only_below"] = True
  cfg.rewards["foot_clearance"].params["target_height"] = swing_target_height
  cfg.rewards["foot_clearance"].weight = -15.0

  # Independent gait-clock swing-height tracking. A fixed-frequency clock the
  # policy does not control drives a desired per-foot swing arc, and the same
  # clock is fed to the policy as an observation (Booster's own K1/T1 recipe).
  # The obs and reward MUST share ``gait_period``.
  clock_obs = ObservationTermCfg(
    func=mdp.gait_clock,
    params={
      "period": gait_period,
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
    "sensor_name": "feet_ground_contact",
    "target_height": swing_target_height,
    "period": gait_period,
    "swing_ratio": 0.45,
    "std": 0.05,
    "profile": "sin",
    "command_name": "twist",
    "command_threshold": 0.05,
  }
  cfg.rewards["foot_swing_height_landing"].weight = 0.0
  cfg.rewards["foot_swing_height_landing"].params = {
    "sensor_name": "feet_ground_contact",
    "height_sensor_name": "foot_height_scan",
    "target_height": swing_target_height,
    "command_name": "twist",
    "command_threshold": 0.05,
  }

  # Flat-foot shaping: the K1 foot sole is the bottom face of the foot box,
  # so the sole normal is the foot body's local Z axis.
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

  # One-sided minimum-separation cost: the symmetric feet_distance pull is
  # far too gentle near zero gap. 0.13 m center-to-center leaves ~0.06 m edge
  # gap for the 0.07 m-wide feet.
  cfg.rewards["feet_min_sep"] = RewardTermCfg(
    func=mdp.feet_min_separation_cost,
    weight=-abs(_env_float("FEET_MIN_SEP_W", 1.0)),
    params={
      "min_distance": _env_float("FEET_MIN_SEP", 0.13),
      "sharpness": _env_float("FEET_MIN_SEP_SHARPNESS", 12.0),
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.01
  cfg.rewards["air_time"].weight = _env_float("AIR_TIME_W", 0.08)
  cfg.rewards["actuation_power"].weight = 0.0
  cfg.rewards["cot_proxy"].weight = 0.0
  cfg.rewards["gait_phase_regularity"].weight = -0.1
  cfg.rewards["limb_symmetry"].weight = 0.0
  cfg.rewards["feet_distance"].weight = -0.1
  cfg.rewards["foot_flat"].weight = foot_flat_w
  # Movement penalties start at zero; the competence gate (or the flat
  # fallback below) brings them in.
  cfg.rewards["joule_heating"].weight = 0.0
  cfg.rewards["joint_acc_l2"].weight = 0.0
  cfg.rewards["torque_rate"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["base_height"].weight = 0.0

  ##
  # Competence machinery.
  ##

  if penalty_gate == "competence":
    _add_competence_penalty_gating(
      cfg,
      joule_w=joule_w,
      joint_acc_w=joint_acc_w,
      torque_rate_w=torque_rate_w,
      soft_landing_w=soft_landing_w,
    )
  else:
    # Time gate: apply the peak weights from iteration 0.
    cfg.rewards["joule_heating"].weight = joule_w
    cfg.rewards["joint_acc_l2"].weight = joint_acc_w
    cfg.rewards["torque_rate"].weight = torque_rate_w
    cfg.rewards["soft_landing"].weight = soft_landing_w
    cfg.rewards["base_height"].weight = _BASE_HEIGHT_W

  has_tracker = adaptive_commands or adaptive_pushes or penalty_gate == "competence"
  if has_tracker:
    _add_competence_tracker_event(cfg)
  if curriculum_style == "aimd" and (adaptive_commands or adaptive_pushes):
    _add_aimd_curriculum(cfg)
  else:
    if adaptive_commands:
      _add_adaptive_command_curriculum(cfg, l_max=adaptive_cmd_lmax)
    if adaptive_pushes:
      _add_adaptive_push_curriculum(cfg, l_max=adaptive_push_lmax)
  if not play and has_tracker and _env_bool("TRACK_WATCHDOG", default=True):
    _add_track_reward_watchdog(cfg)

  # Push-cohort stratification + log-only diagnostics. At frac < 1.0 only the
  # low env indices get pushed; the clean cohort is the uncontaminated
  # tracking-competence baseline.
  if not play and has_tracker:
    if push_cohort_frac < 1.0 and "push_robot" in cfg.events:
      push_params = dict(cfg.events["push_robot"].params)
      push_params["cohort_frac"] = push_cohort_frac
      cfg.events["push_robot"].func = mdp.push_cohort_by_setting_velocity
      cfg.events["push_robot"].params = push_params
    cfg.curriculum["competence_diagnostics"] = CurriculumTermCfg(
      func=mdp.competence_diagnostics,
      params={
        "cohort_frac": push_cohort_frac,
        "frontier_hazard_bar": _env_float("FRONTIER_HAZARD_BAR", 5e-4),
        "push_obs_window_s": _env_float("PUSH_OBS_WINDOW_S", 6.0),
      },
    )
    # Shadow Lagrangian energy multiplier: log-only pilot of the constrained
    # penalty controller.
    if _env_bool("JOULE_LAMBDA_SHADOW", default=True):
      cfg.curriculum["joule_lambda_shadow"] = CurriculumTermCfg(
        func=mdp.joule_lambda_shadow,
        params={
          "reward_name": "joule_heating",
          "lambda_cap": _env_float("LAMBDA_CAP", 2e-5),
          "ramp_iters": _env_int("LAMBDA_RAMP_ITERS", 1000),
          "apply_live": _env_bool("JOULE_LAMBDA_LIVE", default=False),
        },
      )

  ##
  # Observability metrics (pure logging, no gradient).
  ##

  cfg.metrics["foot_heel_toe_pitch_deg"] = MetricsTermCfg(
    func=mdp.foot_heel_toe_pitch_deg,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_foot_link", "right_foot_link")
      ),
      "sole_normal_axis": 2,
    },
  )
  cfg.metrics["foot_lateral_roll_deg"] = MetricsTermCfg(
    func=mdp.foot_lateral_roll_deg,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_foot_link", "right_foot_link")
      ),
      "sole_normal_axis": 2,
    },
  )
  cfg.metrics["foot_toeout_deg"] = MetricsTermCfg(
    func=mdp.foot_toeout_deg,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_foot_link", "right_foot_link")
      ),
      "torso_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      "foot_signs": (1.0, -1.0),
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.metrics["flight_frac"] = MetricsTermCfg(
    func=mdp.flight_fraction,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.metrics["arm_joint_speed"] = MetricsTermCfg(
    func=mdp.joint_speed_abs,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r"^A?(Left|Right)_(Shoulder_(Pitch|Roll)|Elbow_(Pitch|Yaw))$",),
      ),
    },
  )

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
    twist_cmd.ranges.lin_vel_x = (0.1, 0.75)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.0, 0.0)

  return cfg
