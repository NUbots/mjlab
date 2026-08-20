"""NUgus configured to match the dynamics of NUbots' own MuJoCo simulator.

This config reproduces NUbots' simulation dynamics
(``shared/utility/platform/models/nugus/nugus.xml``) on the *same* kinematic
tree as the default mjlab NUgus, so the two can be run against each other and
the difference attributed rather than guessed at. It is a **parity** model, not
a higher-fidelity one -- note the 100 N.m force range below, which the real
11.09 N.m servos plainly do not have. For comparing controllers use
:mod:`~mjlab.asset_zoo.robots.nugus.nugus_eval_constants` instead.

Its other use is as the reference for the question this whole model raises: the
quintic walk engine walks on the real robot and on NUbots' simulator, but falls
after 1.72 s on the model mjlab trains policies against. Which difference is
responsible? Each one applied alone to the training model, driving the deployed
walk at 0.3 m/s for 20 s
(``scripts/tools/play_quintic_walk.py --plant training``):

==================================  ==============================
change                              outcome
==================================  ==============================
training model, unmodified          falls after 1.72 s
force range 11.09 -> 100 N.m        falls after 1.72 s
frictionloss 0 -> 0.03              falls after 1.72 s
``solref`` 0.02 -> 0.005            falls after 1.70 s
leg joint ranges -> hardware        falls after 1.80 s
backlash joints removed             walks, no fall
==================================  ==============================

The backlash joints are the whole of it. Contact stiffness, servo torque limit,
joint friction and the RL joint clamps each change the fall time by less than a
tenth of a second; removing the passive backlash siblings -- and nothing else --
walks. Which is what you would expect of an engine with no feedback on joint
position: 5 mrad of unmodelled play per servo, six of them in series down each
leg, is a foot placement error the engine never learns about.

The default NUgus is left untouched, so policies trained against it keep working.
"""

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  FEET_COLLISION,
  NUGUS_XML,
  STAND_BENT_KNEES_KEYFRAME,
)
from mjlab.asset_zoo.robots.nugus.nugus_nubots_constants import MOTOR_JOINT_NAMES
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Actuators.
##

# Every <position> element in NUbots' model is declared without a class, so they
# all inherit the global default. The MX106/MX64/XH540 classes are attached to
# the *joints*, and therefore only affect joint dynamics -- not the control
# gains, and not the force range. Reproduced faithfully, including the 100 N.m
# range that the real 11.09 N.m servos plainly do not have.
NUBOTS_SIM_STIFFNESS = 40.1
"""``<position kp="40.1">``."""

NUBOTS_SIM_FORCE_RANGE = 100.0
"""``<position forcerange="-100 100">``."""

##
# Joint dynamics, per servo class.
##

# NUbots' model classes only hip_yaw (MX106) and knee_pitch (XH540-W270) on the
# legs, plus the neck, head and arms (MX64). The remaining leg joints fall
# through to the global <joint> default.
JOINT_DYNAMICS: dict[str, tuple[float, float, float]] = {
  # joint name -> (damping, armature, frictionloss)
  "left_hip_yaw": (1.6548, 0.0266, 0.10352),
  "right_hip_yaw": (1.6548, 0.0266, 0.10352),
  "left_knee_pitch": (1.6548, 0.0266, 0.10352),
  "right_knee_pitch": (1.6548, 0.0266, 0.10352),
  "neck_yaw": (0.6782, 0.01195, 0.09039),
  "head_pitch": (0.6782, 0.01195, 0.09039),
  "left_shoulder_pitch": (0.6782, 0.01195, 0.09039),
  "left_shoulder_roll": (0.6782, 0.01195, 0.09039),
  "left_elbow_pitch": (0.6782, 0.01195, 0.09039),
  "right_shoulder_pitch": (0.6782, 0.01195, 0.09039),
  "right_shoulder_roll": (0.6782, 0.01195, 0.09039),
  "right_elbow_pitch": (0.6782, 0.01195, 0.09039),
}

NUBOTS_SIM_DEFAULT_DYNAMICS = (1.084, 0.045, 0.03)
"""``<joint damping="1.084" armature="0.045" frictionloss="0.03"/>``.

Applies to every joint not listed in :data:`JOINT_DYNAMICS`, which on the legs
means hip roll, hip pitch, ankle pitch and ankle roll.
"""

NUBOTS_SIM_SOLREF = (0.005, 1.0)
"""``<geom solref="0.005 1">``: roughly four times stiffer contact than mjlab's."""


def get_spec() -> mujoco.MjSpec:
  """Build the NUgus spec with NUbots' simulation dynamics.

  Shares the XML with the default NUgus, so the kinematic tree stays identical
  by construction -- only the passive backlash joints, joint dynamics and
  contact parameters differ.

  Damping is applied here as *passive joint* damping rather than through the
  actuator config, because mjlab's ``BuiltinPositionActuatorCfg.damping`` is the
  controller's derivative gain. The two enter the equations of motion
  identically, but an actuator term is subject to the force range while passive
  damping is not, and MuJoCo integrates passive damping implicitly.
  """
  spec = mujoco.MjSpec.from_file(str(NUGUS_XML))

  # Drop the passive backlash siblings entirely, rather than locking them, so
  # the model has the same degrees of freedom as NUbots' (nq 27, not 47).
  for joint in list(spec.joints):
    if joint.name.endswith("_backlash"):
      spec.delete(joint)

  for joint in spec.joints:
    if joint.name not in MOTOR_JOINT_NAMES:
      continue
    damping, armature, frictionloss = JOINT_DYNAMICS.get(
      joint.name, NUBOTS_SIM_DEFAULT_DYNAMICS
    )
    # damping is a 3-vector on MjsJoint (ball and free joints use all three);
    # for a hinge only the first element applies. armature and frictionloss are
    # scalars.
    joint.damping[0] = damping
    joint.armature = armature
    joint.frictionloss = frictionloss

  for geom in spec.geoms:
    geom.solref[0] = NUBOTS_SIM_SOLREF[0]
    geom.solref[1] = NUBOTS_SIM_SOLREF[1]

  return spec


NUGUS_NUBOTS_SIM_ACTUATORS = BuiltinPositionActuatorCfg(
  target_names_expr=MOTOR_JOINT_NAMES,
  stiffness=NUBOTS_SIM_STIFFNESS,
  # Zero derivative gain: NUbots' <position> elements specify only kp, and the
  # velocity term comes from passive joint damping set in get_spec().
  damping=0.0,
  effort_limit=NUBOTS_SIM_FORCE_RANGE,
)
"""One uniform actuator group, matching NUbots' unclassed ``<position>`` elements.

Note the absence of ``delay_min_lag``/``delay_max_lag``: NUbots' model has no
actuator latency, unlike the default mjlab NUgus.
"""

NUGUS_NUBOTS_SIM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(NUGUS_NUBOTS_SIM_ACTUATORS,),
  soft_joint_pos_limit_factor=0.9,
)


def get_nugus_nubots_sim_robot_cfg() -> EntityCfg:
  """Fresh NUgus config matching NUbots' simulator dynamics."""
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=NUGUS_NUBOTS_SIM_ARTICULATION,
  )
