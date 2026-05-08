from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

NUGUS_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nugus" / "xmls" / "nugus.xml"
)
assert NUGUS_XML.exists(), f"XML not found: {NUGUS_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, NUGUS_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(NUGUS_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# Actuators: MX106, MX64, XH540-W270
# Motor specs (from Nugus XML defaults).
# Using armature values directly from the XML

# MX106 motor (hip yaw joints)
ARMATURE_MX106 = 0.0266
ACTUATOR_MX106 = ElectricActuator(
    reflected_inertia=ARMATURE_MX106,
    velocity_limit=30,      # TODO, not specified in XML
    effort_limit=11.086,    # From forcerange in xml
)

# MX64 motor (most other joints)
ARMATURE_MX64 = 0.01195
ACTUATOR_MX64 = ElectricActuator(
    reflected_inertia=ARMATURE_MX64,
    velocity_limit=30,      # TODO, not specified in xml
    effort_limit=6.1621,    # From forcerange in xml
)

# XH540-W270 motor (knee joints)
ARMATURE_XH540=0.0266
ACTUATOR_XH540 = ElectricActuator(
    reflected_inertia=ARMATURE_XH540,
    velocity_limit=30,      # TODO, not specified in xml
    effort_limit=11.086,    # From forcerange in xml
)

# Natural frequency and damping ratio for PD control
# Use kp values from xml directly as stiffness
STIFFNESS_MX106 = 56.052
DAMPING_MX106   = 1.6548

STIFFNESS_MX64 = 31.1558
DAMPING_MX64   = 0.6782

STIFFNESS_XH540 = 56.052
DAMPING_XH540   = 1.6548

# Actuator configs for different joint groups
# Using order from poster in NUbots lab (right then left)
NUGUS_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
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
    effort_limit=ACTUATOR_MX64.effort_limit,
    armature=ACTUATOR_MX64.reflected_inertia,
)

NUGUS_ACTUATOR_HIPS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "right_hip_yaw",
    "left_hip_yaw",
  ),
  stiffness=STIFFNESS_MX106,
  damping=DAMPING_MX106,
  effort_limit=ACTUATOR_MX106.effort_limit,
  armature=ACTUATOR_MX106.reflected_inertia,
)

NUGUS_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
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
  effort_limit=ACTUATOR_XH540.effort_limit,
  armature=ACTUATOR_XH540.reflected_inertia,
)

NUGUS_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=("neck_yaw", "head_pitch"),
    stiffness=STIFFNESS_MX64,
    damping=DAMPING_MX64,
    effort_limit=ACTUATOR_MX64.effort_limit,
    armature=ACTUATOR_MX64.reflected_inertia,
)

##
# Keyframe config.
##

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
FOOT_GROUND_FRICTION = 1.1

# Keep non-foot contact friction lower to reduce limb sticking.
NON_FOOT_COLLISION_FRICTION = 0.5

# Basic collision
FEET_COLLISION = CollisionCfg(
    geom_names_expr=(".*foot_collision",),
    contype=1,
    conaffinity=2,
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
    soft_joint_pos_limit_factor=0.9, # TODO
)

def get_nugus_robot_cfg() -> EntityCfg:
    """ Get a fresh Nugus robot configuration instance.
    
    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=STAND_BENT_KNEES_KEYFRAME,
        collisions=(FEET_COLLISION,),
        spec_fn=get_spec, 
        articulation=NUGUS_ARTICULATION,
    )

NUGUS_ACTION_SCALE: dict[str, float] = {}
for a in NUGUS_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    NUGUS_ACTION_SCALE[n] = 0.25 * e / s # Approx 0.049

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_nugus_robot_cfg())

  viewer.launch(robot.spec.compile())
