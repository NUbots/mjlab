"""Booster K1 robot constants.

Model and actuator data come from the official Booster Robotics assets and
Isaac Lab training configs (https://github.com/BoosterRobotics/booster_assets
and https://github.com/BoosterRobotics/booster_train, BSD-3-Clause). Motor
specs (effort limits, velocity limits, rotor armature) are per-joint motor
models from Booster's actuator library; PD gains follow Booster's formula
``kp = armature * (2*pi*f)**2``, ``kd = 2 * zeta * armature * (2*pi*f)`` with
their published natural frequency ``f`` and damping ratio ``zeta`` per joint
group.
"""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "k1.xml"
)
assert K1_XML.exists(), f"XML not found: {K1_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(K1_XML))


##
# Actuator config.
##


def _stiffness(armature: float, natural_freq_hz: float) -> float:
  return armature * (2 * math.pi * natural_freq_hz) ** 2


def _damping(armature: float, natural_freq_hz: float, damping_ratio: float) -> float:
  return 2 * damping_ratio * armature * (2 * math.pi * natural_freq_hz)


# Booster motor models used in the K1 (from booster_train's actuator library).
ACTUATOR_E6408 = ElectricActuator(  # Hip pitch.
  reflected_inertia=0.0478125,
  velocity_limit=14.66,
  effort_limit=68.0,
)
ACTUATOR_E4315 = ElectricActuator(  # Hip roll.
  reflected_inertia=0.0339552,
  velocity_limit=12.57,
  effort_limit=76.0,
)
ACTUATOR_E4310 = ElectricActuator(  # Hip yaw; also the ankle base motor.
  reflected_inertia=0.0282528,
  velocity_limit=17.59,
  effort_limit=38.3,
)
ACTUATOR_E6416 = ElectricActuator(  # Knee.
  reflected_inertia=0.095625,
  velocity_limit=12.57,
  effort_limit=112.0,
)
# The ankle is a parallel mechanism driven by two E4310s; Booster's serial
# approximation doubles the reflected inertia per ankle joint.
ACTUATOR_ANKLE = ElectricActuator(
  reflected_inertia=2 * ACTUATOR_E4310.reflected_inertia,
  velocity_limit=ACTUATOR_E4310.velocity_limit,
  effort_limit=ACTUATOR_E4310.effort_limit,
)
ACTUATOR_R14 = ElectricActuator(  # Arms.
  reflected_inertia=0.001,
  velocity_limit=33.51,
  effort_limit=14.0,
)
ACTUATOR_HT4438 = ElectricActuator(  # Head.
  reflected_inertia=0.001,
  velocity_limit=7.85,
  effort_limit=6.0,
)

# Natural frequency (Hz) / damping ratio per joint group, from Booster's K1
# training config. Arms and head use the actuator library defaults (10 Hz, 2.0).
_LEG_FREQ = 4.0
_ARM_FREQ = 10.0
_ARM_DAMPING_RATIO = 2.0

# Command delay in physics timesteps, matching Booster's actuator delay
# randomization (min_delay=2, max_delay=8 at the same 5 ms physics step).
_DELAY_MIN_LAG = 2
_DELAY_MAX_LAG = 8

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=_stiffness(ACTUATOR_E6408.reflected_inertia, _LEG_FREQ),
  damping=_damping(ACTUATOR_E6408.reflected_inertia, _LEG_FREQ, 1.5),
  effort_limit=ACTUATOR_E6408.effort_limit,
  armature=ACTUATOR_E6408.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Roll",),
  stiffness=_stiffness(ACTUATOR_E4315.reflected_inertia, _LEG_FREQ),
  damping=_damping(ACTUATOR_E4315.reflected_inertia, _LEG_FREQ, 1.5),
  effort_limit=ACTUATOR_E4315.effort_limit,
  armature=ACTUATOR_E4315.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Yaw",),
  stiffness=_stiffness(ACTUATOR_E4310.reflected_inertia, _LEG_FREQ),
  damping=_damping(ACTUATOR_E4310.reflected_inertia, _LEG_FREQ, 1.5),
  effort_limit=ACTUATOR_E4310.effort_limit,
  armature=ACTUATOR_E4310.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=_stiffness(ACTUATOR_E6416.reflected_inertia, _LEG_FREQ),
  damping=_damping(ACTUATOR_E6416.reflected_inertia, _LEG_FREQ, 1.0),
  effort_limit=ACTUATOR_E6416.effort_limit,
  armature=ACTUATOR_E6416.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
  stiffness=_stiffness(ACTUATOR_ANKLE.reflected_inertia, _LEG_FREQ),
  damping=_damping(ACTUATOR_ANKLE.reflected_inertia, _LEG_FREQ, 1.5),
  effort_limit=ACTUATOR_ANKLE.effort_limit,
  armature=ACTUATOR_ANKLE.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  stiffness=_stiffness(ACTUATOR_R14.reflected_inertia, _ARM_FREQ),
  damping=_damping(ACTUATOR_R14.reflected_inertia, _ARM_FREQ, _ARM_DAMPING_RATIO),
  effort_limit=ACTUATOR_R14.effort_limit,
  armature=ACTUATOR_R14.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

K1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=_stiffness(ACTUATOR_HT4438.reflected_inertia, _ARM_FREQ),
  damping=_damping(ACTUATOR_HT4438.reflected_inertia, _ARM_FREQ, _ARM_DAMPING_RATIO),
  effort_limit=ACTUATOR_HT4438.effort_limit,
  armature=ACTUATOR_HT4438.reflected_inertia,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
)

##
# Keyframe config.
##

# Bent-knee stand: Booster's arms-down init pose plus the hip/knee/ankle bend
# used by their K1 walking configs. Base height puts the foot soles on the
# ground (computed via forward kinematics on the compiled model).
STAND_BENT_KNEES_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.5435),
  joint_pos={
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_REGEX = r".*foot_collision$"

FOOT_GROUND_FRICTION = 1.0
NON_FOOT_COLLISION_FRICTION = 1.0

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

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE,
    K1_ACTUATOR_ANKLE,
    K1_ACTUATOR_ARMS,
    K1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh Booster K1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


# Booster's action scale convention: 0.25 * effort_limit / stiffness.
K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  assert isinstance(s, float)
  for n in a.target_names_expr:
    K1_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_k1_robot_cfg())

  viewer.launch(robot.spec.compile())
