import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

NUGUS_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nugus" / "xmls" / "nugus.xml"
)
assert NUGUS_XML.exists(), f"XML not found: {NUGUS_XML}"

##
# Servo backlash model.
##

# Regex matching only motor joints (i.e. excludes the passive *_backlash siblings).
# Use this anywhere a config wants "all actuated joints" so the passive backlash
# joints aren't accidentally included in observations/rewards/events.
NUGUS_MOTOR_JOINT_REGEX: str = r"^(?!.*_backlash$).*"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(NUGUS_XML))


##
# Actuator config.
##

# Actuators: MX106, MX64, XH540-W270
# Motor specs (from Nugus XML defaults).
# Using armature values directly from the XML

# BAM m1 Coulomb-viscous baselines (Rhoban/bam params/mx64/m1.json).
FRICTIONLOSS_MX64 = 0.09038677246291783
VISCOUS_DAMPING_MX64 = 0.011691602145974832

# BAM m1 Coulomb-viscous baselines (Rhoban/bam params/mx106/m1.json).
FRICTIONLOSS_MX106 = 0.10352026623606064
VISCOUS_DAMPING_MX106 = 0.03520238029013507

# PROVISIONAL: XH540-W270 from MX-106 scaled by gear ratio (270.4:1 vs 225:1).
_XH540_GEAR_SCALE = 270.4 / 225.0
FRICTIONLOSS_XH540 = FRICTIONLOSS_MX106 * _XH540_GEAR_SCALE
VISCOUS_DAMPING_XH540 = VISCOUS_DAMPING_MX106 * _XH540_GEAR_SCALE

# Bus voltage for no-load speed / torque scaling (4S LiPo nominal).
# VERIFY: confirm NUgus supply matches before hardware transfer.
_BUS_VOLTAGE = 14.8
_REFERENCE_VOLTAGE = 12.0

# No-load speed @ 12 V (rpm) from ROBOTIS e-Manual: MX-106 / XH540 / MX-64.
_RPM_NO_LOAD_12V_MX106 = 45.0
_RPM_NO_LOAD_12V_XH540 = 39.0
_RPM_NO_LOAD_12V_MX64 = 63.0
# XH540 @ 14.8 V is listed directly in e-Manual (46 rpm); MX series scaled.
_RPM_NO_LOAD_14V8_XH540 = 46.0


def _rpm_to_rad_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


def _rpm_at_bus(rpm_12v: float, v_bus: float = _BUS_VOLTAGE) -> float:
  return rpm_12v * (v_bus / _REFERENCE_VOLTAGE)


VELOCITY_LIMIT_MX106 = _rpm_to_rad_s(_rpm_at_bus(_RPM_NO_LOAD_12V_MX106))
VELOCITY_LIMIT_XH540 = _rpm_to_rad_s(_RPM_NO_LOAD_14V8_XH540)
VELOCITY_LIMIT_MX64 = _rpm_to_rad_s(_rpm_at_bus(_RPM_NO_LOAD_12V_MX64))

# Stall / rated torque @ 14.8 V from ROBOTIS e-Manual (MX/XH540 spec tables).
# VERIFY: bench replay on NUgus servos may refine these.
_STALL_TORQUE_MX106 = 10.0
_STALL_TORQUE_MX64 = 7.3
_STALL_TORQUE_XH540 = 11.7
# Rated continuous torque @ 12 V (performance-graph mid-range); scaled to bus.
_RATED_TORQUE_12V_MX106 = 4.4
_RATED_TORQUE_12V_MX64 = 3.0
_RATED_TORQUE_12V_XH540 = 5.4
_VOLTAGE_SCALE = _BUS_VOLTAGE / _REFERENCE_VOLTAGE
RATED_TORQUE_MX106 = _RATED_TORQUE_12V_MX106 * _VOLTAGE_SCALE
RATED_TORQUE_MX64 = _RATED_TORQUE_12V_MX64 * _VOLTAGE_SCALE
RATED_TORQUE_XH540 = _RATED_TORQUE_12V_XH540 * _VOLTAGE_SCALE

# MX106 motor (hip yaw joints)
ARMATURE_MX106 = 0.0266
ACTUATOR_MX106 = ElectricActuator(
  reflected_inertia=ARMATURE_MX106,
  velocity_limit=VELOCITY_LIMIT_MX106,
  effort_limit=_STALL_TORQUE_MX106,
)

# MX64 motor (most other joints)
ARMATURE_MX64 = 0.01195
ACTUATOR_MX64 = ElectricActuator(
  reflected_inertia=ARMATURE_MX64,
  velocity_limit=VELOCITY_LIMIT_MX64,
  effort_limit=_STALL_TORQUE_MX64,
)

# XH540-W270 motor (knee joints)
# Measured by servo sysid on hardware walking data (2026-07-08, doc 17):
# reflected inertia J = 0.0496 +/- 0.0108 kg.m^2 pooled over 10/12 leg
# joints (electrical fit R^2 0.85-0.99). The previous 0.0266 was the
# MX-106 value copied over and sat ~1.9x low - outside even the +/-20%
# DR band, so no training env ever saw realistic leg inertia.
ARMATURE_XH540 = 0.0496
ACTUATOR_XH540 = ElectricActuator(
  reflected_inertia=ARMATURE_XH540,
  velocity_limit=VELOCITY_LIMIT_XH540,
  effort_limit=_STALL_TORQUE_XH540,
)

# Natural frequency and damping ratio for PD control
# Use kp values from xml directly as stiffness
STIFFNESS_MX106 = 56.052
DAMPING_MX106 = 1.6548

STIFFNESS_MX64 = 31.1558
DAMPING_MX64 = 0.6782

STIFFNESS_XH540 = 56.052
DAMPING_XH540 = 1.6548

# Actuator configs for different joint groups
NUGUS_ACTUATOR_ARMS = DcMotorActuatorCfg(
  target_names_expr=(
    "right_shoulder_pitch",
    "left_shoulder_pitch",
    "right_shoulder_roll",
    "left_shoulder_roll",
    "right_elbow_pitch",
    "left_elbow_pitch",
  ),
  stiffness=STIFFNESS_MX64,
  damping=DAMPING_MX64,
  effort_limit=_STALL_TORQUE_MX64,
  saturation_effort=_STALL_TORQUE_MX64,
  velocity_limit=VELOCITY_LIMIT_MX64,
  armature=ACTUATOR_MX64.reflected_inertia,
  frictionloss=FRICTIONLOSS_MX64,
  viscous_damping=VISCOUS_DAMPING_MX64,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_ACTUATOR_HIPS = DcMotorActuatorCfg(
  target_names_expr=(
    "right_hip_yaw",
    "left_hip_yaw",
  ),
  stiffness=STIFFNESS_MX106,
  damping=DAMPING_MX106,
  effort_limit=_STALL_TORQUE_MX106,
  saturation_effort=_STALL_TORQUE_MX106,
  velocity_limit=VELOCITY_LIMIT_MX106,
  armature=ACTUATOR_MX106.reflected_inertia,
  frictionloss=FRICTIONLOSS_MX106,
  viscous_damping=VISCOUS_DAMPING_MX106,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_ACTUATOR_LEGS = DcMotorActuatorCfg(
  target_names_expr=(
    "right_hip_roll",
    "left_hip_roll",
    "right_hip_pitch",
    "left_hip_pitch",
    "right_knee_pitch",
    "left_knee_pitch",
    "right_ankle_pitch",
    "left_ankle_pitch",
    "right_ankle_roll",
    "left_ankle_roll",
  ),
  stiffness=STIFFNESS_XH540,
  damping=DAMPING_XH540,
  effort_limit=_STALL_TORQUE_XH540,
  saturation_effort=_STALL_TORQUE_XH540,
  velocity_limit=VELOCITY_LIMIT_XH540,
  armature=ACTUATOR_XH540.reflected_inertia,
  frictionloss=FRICTIONLOSS_XH540,
  viscous_damping=VISCOUS_DAMPING_XH540,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_ACTUATOR_HEAD = DcMotorActuatorCfg(
  target_names_expr=("neck_yaw", "head_pitch"),
  stiffness=STIFFNESS_MX64,
  damping=DAMPING_MX64,
  effort_limit=_STALL_TORQUE_MX64,
  saturation_effort=_STALL_TORQUE_MX64,
  velocity_limit=VELOCITY_LIMIT_MX64,
  armature=ACTUATOR_MX64.reflected_inertia,
  frictionloss=FRICTIONLOSS_MX64,
  viscous_damping=VISCOUS_DAMPING_MX64,
  delay_min_lag=1,
  delay_max_lag=3,
)

##
# Keyframe config.
##

STAND_STRAIGHTER_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.49500),
  joint_pos={
    "right_shoulder_pitch": 1.714,
    "left_shoulder_pitch": 1.716,
    "right_shoulder_roll": -0.198,
    "left_shoulder_roll": 0.198,
    "right_elbow_pitch": -0.720,
    "left_elbow_pitch": -0.715,
    "right_hip_yaw": -0.033,
    "left_hip_yaw": 0.034,
    "right_hip_roll": -0.162,
    "left_hip_roll": 0.163,
    "right_hip_pitch": -0.604,
    "left_hip_pitch": -0.604,
    "right_knee_pitch": 0.800,
    "left_knee_pitch": 0.800,
    "right_ankle_pitch": -0.310,
    "left_ankle_pitch": -0.310,
    "right_ankle_roll": 0.168,
    "left_ankle_roll": -0.167,
    "neck_yaw": 0.0,
    "head_pitch": 0.0,
  },
  joint_vel={".*": 0.0},
)

STAND_BENT_KNEES_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.473774),
  joint_pos={
    "right_shoulder_pitch": 1.71,
    "left_shoulder_pitch": 1.71,
    "right_shoulder_roll": -0.197,
    "left_shoulder_roll": 0.197,
    "right_elbow_pitch": -0.718,
    "left_elbow_pitch": -0.713,
    "right_hip_yaw": -0.0329,
    "left_hip_yaw": 0.0339,
    "right_hip_roll": -0.162,
    "left_hip_roll": 0.163,
    "right_hip_pitch": -0.904,
    "left_hip_pitch": -0.904,
    "right_knee_pitch": 1.20,
    "left_knee_pitch": 1.20,
    "right_ankle_pitch": -0.508,
    "left_ankle_pitch": -0.510,
    "right_ankle_roll": 0.167,
    "left_ankle_roll": -0.166,
    "neck_yaw": 0.0,
    "head_pitch": 0.0000645,
  },
  joint_vel={".*": 0.0},
)

##
#  Collision Config.
##

FOOT_COLLISION_REGEX = r".*foot_collision$"

# Increase this to raise foot-ground traction.
FOOT_GROUND_FRICTION = 1.0

# Keep non-foot contact friction lower to reduce limb sticking.
NON_FOOT_COLLISION_FRICTION = 1.0

# Basic collision
FEET_COLLISION = CollisionCfg(
  geom_names_expr=(".*foot_collision",),
  contype=1,
  # Feet collide with the ground and with each other.
  conaffinity=1,
  condim=3,
  friction=(FOOT_GROUND_FRICTION,),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  # Enable robot-robot and robot-ground contacts for collision geoms.
  conaffinity=1,
  condim=3,
  friction={
    FOOT_COLLISION_REGEX: (FOOT_GROUND_FRICTION,),
    ".*_collision": (NON_FOOT_COLLISION_FRICTION,),
  },
)

# Full body collision with ground only
FULL_COLLISION_GND_ONLY = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  # Ground/environment contacts only (no self-collisions).
  conaffinity=0,
  condim=3,
  friction={
    FOOT_COLLISION_REGEX: (FOOT_GROUND_FRICTION,),
    ".*_collision": (NON_FOOT_COLLISION_FRICTION,),
  },
)


##
# Final config.
##

# Note: The order of the actuators in the inference will be as they appear in the XML (inside the <worldbody> section)
NUGUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    NUGUS_ACTUATOR_ARMS,
    NUGUS_ACTUATOR_HIPS,
    NUGUS_ACTUATOR_LEGS,
    NUGUS_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,  # TODO
)


def get_nugus_robot_cfg() -> EntityCfg:
  """Get a fresh Nugus robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,  # Change the keyframe here
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=NUGUS_ARTICULATION,
  )


NUGUS_ACTION_SCALE: dict[str, float] = {}
for a in NUGUS_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    NUGUS_ACTION_SCALE[n] = (0.25 * e / s) * 5  # Approx 0.049 * 5 = 0.245

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_nugus_robot_cfg())

  viewer.launch(robot.spec.compile())
