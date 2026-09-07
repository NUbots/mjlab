"""The NUgus with the DC-motor actuator model the v-generations trained on.

The ``add-phase-clock`` branch drives the NUgus through
:class:`~mjlab.actuator.DcMotorActuatorCfg` -- a torque-speed model whose
limits come from the ROBOTIS e-Manual at the robot's 14.8 V bus -- while
this branch drives it through
:class:`~mjlab.actuator.BuiltinPositionActuatorCfg` with effort limits read
off the XML ``forcerange``. Everything else about the robot agrees:
kinematics, masses, geometry, collisions and the keyframe are identical, and
so are the PD stiffness and damping.

Two consequences, and both matter for evaluating a policy from that branch:

* **Torque authority differs.** The DC model saturates against a
  speed-dependent envelope rather than a constant force range, so the same
  joint target produces a different torque at speed.
* **The action scale differs.** ``NUGUS_ACTION_SCALE`` is derived from each
  actuator's ``effort_limit / stiffness``, so a plant swap silently rescales
  every action the policy emits -- about 18% on the MX64 groups. A policy
  measured under the wrong scale is being asked for joint targets it never
  learned.

A robust policy absorbs this; v57, with domain randomization and an
observation window, walks either way. v44 does not: on the builtin-actuator
plant it is within a few degrees of the fall bound while merely standing.
That is the transplant being measured, not the controller, which is exactly
what the harness's plant selection exists to avoid.
"""

from __future__ import annotations

import math

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  ARMATURE_MX64,
  ARMATURE_MX106,
  ARMATURE_XH540,
  DAMPING_MX64,
  DAMPING_MX106,
  DAMPING_XH540,
  FEET_COLLISION,
  STAND_BENT_KNEES_KEYFRAME,
  STIFFNESS_MX64,
  STIFFNESS_MX106,
  STIFFNESS_XH540,
  get_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Torque-speed envelope, from the ROBOTIS e-Manual at the NUgus bus voltage.
##

_BUS_VOLTAGE = 14.8
_REFERENCE_VOLTAGE = 12.0
_VOLTAGE_SCALE = _BUS_VOLTAGE / _REFERENCE_VOLTAGE

# No-load speed @ 12 V (rpm): MX-106 / XH540 / MX-64. The XH540 figure at
# 14.8 V is listed directly; the MX series is scaled by the bus ratio.
_RPM_NO_LOAD_12V_MX106 = 45.0
_RPM_NO_LOAD_12V_XH540 = 39.0
_RPM_NO_LOAD_12V_MX64 = 63.0
_RPM_NO_LOAD_14V8_XH540 = 46.0


def _rpm_to_rad_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


def _rpm_at_bus(rpm_12v: float, v_bus: float = _BUS_VOLTAGE) -> float:
  return rpm_12v * (v_bus / _REFERENCE_VOLTAGE)


VELOCITY_LIMIT_MX106 = _rpm_to_rad_s(_rpm_at_bus(_RPM_NO_LOAD_12V_MX106))
VELOCITY_LIMIT_XH540 = _rpm_to_rad_s(_RPM_NO_LOAD_14V8_XH540)
VELOCITY_LIMIT_MX64 = _rpm_to_rad_s(_rpm_at_bus(_RPM_NO_LOAD_12V_MX64))

# Stall torque @ 14.8 V, and rated continuous torque @ 12 V scaled to the bus.
_STALL_TORQUE_MX106 = 10.0
_STALL_TORQUE_MX64 = 7.3
_STALL_TORQUE_XH540 = 11.7
_RATED_TORQUE_12V_MX106 = 4.4
_RATED_TORQUE_12V_MX64 = 3.0
_RATED_TORQUE_12V_XH540 = 5.4
RATED_TORQUE_MX106 = _RATED_TORQUE_12V_MX106 * _VOLTAGE_SCALE
RATED_TORQUE_MX64 = _RATED_TORQUE_12V_MX64 * _VOLTAGE_SCALE
RATED_TORQUE_XH540 = _RATED_TORQUE_12V_XH540 * _VOLTAGE_SCALE

# Coulomb and viscous friction, from bench sysid on the servos. The XH540
# shares the MX106's measurement scaled by the gear ratio.
_XH540_GEAR_SCALE = 270.4 / 225.0
FRICTIONLOSS_MX64 = 0.09038677246291783
VISCOUS_DAMPING_MX64 = 0.011691602145974832
FRICTIONLOSS_MX106 = 0.10352026623606064
VISCOUS_DAMPING_MX106 = 0.03520238029013507
FRICTIONLOSS_XH540 = FRICTIONLOSS_MX106 * _XH540_GEAR_SCALE
VISCOUS_DAMPING_XH540 = VISCOUS_DAMPING_MX106 * _XH540_GEAR_SCALE

##
# Actuator groups. Stiffness, damping and armature match the builtin-actuator
# robot exactly; only the torque model and its limits differ.
##

NUGUS_DC_ACTUATOR_ARMS = DcMotorActuatorCfg(
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
  armature=ARMATURE_MX64,
  frictionloss=FRICTIONLOSS_MX64,
  viscous_damping=VISCOUS_DAMPING_MX64,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_DC_ACTUATOR_HIPS = DcMotorActuatorCfg(
  target_names_expr=(
    "right_hip_yaw",
    "left_hip_yaw",
  ),
  stiffness=STIFFNESS_MX106,
  damping=DAMPING_MX106,
  effort_limit=_STALL_TORQUE_MX106,
  saturation_effort=_STALL_TORQUE_MX106,
  velocity_limit=VELOCITY_LIMIT_MX106,
  armature=ARMATURE_MX106,
  frictionloss=FRICTIONLOSS_MX106,
  viscous_damping=VISCOUS_DAMPING_MX106,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_DC_ACTUATOR_LEGS = DcMotorActuatorCfg(
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
  armature=ARMATURE_XH540,
  frictionloss=FRICTIONLOSS_XH540,
  viscous_damping=VISCOUS_DAMPING_XH540,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_DC_ACTUATOR_HEAD = DcMotorActuatorCfg(
  target_names_expr=("neck_yaw", "head_pitch"),
  stiffness=STIFFNESS_MX64,
  damping=DAMPING_MX64,
  effort_limit=_STALL_TORQUE_MX64,
  saturation_effort=_STALL_TORQUE_MX64,
  velocity_limit=VELOCITY_LIMIT_MX64,
  armature=ARMATURE_MX64,
  frictionloss=FRICTIONLOSS_MX64,
  viscous_damping=VISCOUS_DAMPING_MX64,
  delay_min_lag=1,
  delay_max_lag=3,
)

NUGUS_DC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    NUGUS_DC_ACTUATOR_ARMS,
    NUGUS_DC_ACTUATOR_HIPS,
    NUGUS_DC_ACTUATOR_LEGS,
    NUGUS_DC_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_nugus_dcmotor_robot_cfg() -> EntityCfg:
  """A fresh NUgus on the DC-motor actuator model."""
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=NUGUS_DC_ARTICULATION,
  )


NUGUS_DCMOTOR_ACTION_SCALE: dict[str, float] = {}
"""Action scale implied by this plant, derived exactly as the builtin one is.

The training branch computes the scale from the actuator it configures, so a
policy trained there emits joint targets against *these* numbers. Evaluating
it under the builtin plant's scale rescales every action it produces.
"""
for _actuator in NUGUS_DC_ARTICULATION.actuators:
  assert isinstance(_actuator, DcMotorActuatorCfg)
  _effort = _actuator.effort_limit
  assert _effort is not None
  for _name in _actuator.target_names_expr:
    NUGUS_DCMOTOR_ACTION_SCALE[_name] = (0.25 * _effort / _actuator.stiffness) * 5
