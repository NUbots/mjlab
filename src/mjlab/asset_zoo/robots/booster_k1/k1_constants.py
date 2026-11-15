"""Booster K1 robot constants.

The model comes from the official Booster Robotics assets
(https://github.com/BoosterRobotics/booster_assets, BSD-3-Clause), and the motor
specs (effort limits, velocity limits, rotor armature) from Booster's actuator
library (https://github.com/BoosterRobotics/booster_train).

PD gains, the default pose, joint ranges and collision geometry follow
booster_mjlab (https://github.com/IntelligentRoboticsLab/booster_mjlab), whose
velocity policies transfer to the real K1. Policies trained on the previous
gains here (``armature * (2*pi*f)**2`` at f = 4 Hz, about 30 N*m/rad at the
hip) did not.
"""

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

# Command delay in physics timesteps (2-8 at the 5 ms physics step), with a 30%
# chance per step of holding the previous delay, as in booster_mjlab.
_DELAY_MIN_LAG = 2
_DELAY_MAX_LAG = 8
_DELAY_HOLD_PROB = 0.3

# PD gains (N*m/rad, N*m*s/rad) per joint group, from booster_mjlab.
_LEG_KP, _LEG_KD = 80.0, 4.0
_ANKLE_KP, _ANKLE_KD = 50.0, 2.0
_ARM_KP, _ARM_KD = 10.0, 1.0
_HEAD_KP, _HEAD_KD = 4.0, 0.25


def _actuator(
  target_names_expr: tuple[str, ...],
  motor: ElectricActuator,
  stiffness: float,
  damping: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=motor.effort_limit,
    armature=motor.reflected_inertia,
    delay_min_lag=_DELAY_MIN_LAG,
    delay_max_lag=_DELAY_MAX_LAG,
    delay_hold_prob=_DELAY_HOLD_PROB,
  )


K1_ACTUATOR_HIP_PITCH = _actuator((".*_Hip_Pitch",), ACTUATOR_E6408, _LEG_KP, _LEG_KD)
K1_ACTUATOR_HIP_ROLL = _actuator((".*_Hip_Roll",), ACTUATOR_E4315, _LEG_KP, _LEG_KD)
K1_ACTUATOR_HIP_YAW = _actuator((".*_Hip_Yaw",), ACTUATOR_E4310, _LEG_KP, _LEG_KD)
K1_ACTUATOR_KNEE = _actuator((".*_Knee_Pitch",), ACTUATOR_E6416, _LEG_KP, _LEG_KD)
K1_ACTUATOR_ANKLE = _actuator(
  (".*_Ankle_Pitch", ".*_Ankle_Roll"), ACTUATOR_ANKLE, _ANKLE_KP, _ANKLE_KD
)
K1_ACTUATOR_ARMS = _actuator(
  (
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  ACTUATOR_R14,
  _ARM_KP,
  _ARM_KD,
)
K1_ACTUATOR_HEAD = _actuator(
  ("AAHead_yaw", "Head_pitch"), ACTUATOR_HT4438, _HEAD_KP, _HEAD_KD
)

##
# Keyframe config.
##

# booster_mjlab's home pose: a deep bent-knee crouch with the arms down. The
# base height puts the foot soles on the ground (booster_mjlab's 0.5125 sinks
# them 5.6 mm; 0.518 was its earlier value).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.518),
  joint_pos={
    "Left_Shoulder_Roll": -1.4,
    "Right_Shoulder_Roll": 1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.4,
    ".*_Knee_Pitch": 0.8,
    ".*_Ankle_Pitch": -0.4,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_REGEX = r".*foot_collision$"

FOOT_GROUND_FRICTION = 1.0
NON_FOOT_COLLISION_FRICTION = 0.45

# Feet get priority 1 so their (randomized) friction decides foot-ground
# contact instead of MuJoCo taking the max of the foot and terrain values.
# booster_mjlab gets the same effect by giving its terrain near-zero friction.
_FOOT_PRIORITY = {FOOT_COLLISION_REGEX: 1, ".*": 0}
_FRICTION: dict[str, tuple[float, ...]] = {
  FOOT_COLLISION_REGEX: (FOOT_GROUND_FRICTION,),
  ".*_collision": (NON_FOOT_COLLISION_FRICTION,),
}
# Non-foot geoms, as in booster_mjlab: stiff robot-robot contacts, and a small
# solmix so the terrain's parameters decide contacts with the ground. The feet
# keep the XML's solref, since their priority makes it decide foot-ground
# contact.
_NON_FOOT_REGEX = r"^(?!.*foot_collision$).*_collision$"
_NON_FOOT_SOLREF: dict[str, tuple[float, ...]] = {_NON_FOOT_REGEX: (0.01, 1.0)}
_NON_FOOT_SOLMIX: dict[str, float] = {_NON_FOOT_REGEX: 0.001}

FEET_COLLISION = CollisionCfg(
  geom_names_expr=(".*foot_collision",),
  contype=1,
  # Feet collide with the ground and with each other.
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(FOOT_GROUND_FRICTION,),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  # Enable robot-robot and robot-ground contacts for collision geoms.
  conaffinity=1,
  condim=3,
  priority=_FOOT_PRIORITY,
  friction=_FRICTION,
  solref=_NON_FOOT_SOLREF,
  solmix=_NON_FOOT_SOLMIX,
)

FULL_COLLISION_GND_ONLY = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  # Ground/environment contacts only (no self-collisions).
  conaffinity=0,
  condim=3,
  priority=_FOOT_PRIORITY,
  friction=_FRICTION,
  solref=_NON_FOOT_SOLREF,
  solmix=_NON_FOOT_SOLMIX,
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
    init_state=HOME_KEYFRAME,
    # Every collision geom, self-collisions included, as in booster_mjlab.
    collisions=(FULL_COLLISION,),
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
