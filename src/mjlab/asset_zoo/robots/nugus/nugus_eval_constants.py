"""NUgus configured for evaluating controllers rather than training them.

The default mjlab NUgus (:mod:`~mjlab.asset_zoo.robots.nugus.nugus_constants`)
is a *training* model. It deliberately hardens the robot for sim-to-real
transfer -- passive backlash joints on every servo, actuator latency -- and it
clamps the leg joints to the range the policy is allowed to explore. A learned
policy is trained against those perturbations and inside those clamps. A
hand-tuned controller such as the ported NUbots quintic walk engine has seen
none of them and has no mechanism to adapt.

This config is the same robot with the sim-to-real randomisation set to its
nominal value and the leg joint limits restored to the hardware's. Training with
domain randomisation and evaluating at nominal is standard practice, and it is
the comparison both controllers can be held to: one of them was tuned on
hardware, the other trained across a range that contains this model.

Two changes, and nothing else:

Randomisation at nominal
  The backlash joints are removed and the actuator delay is switched off. Gear
  play amplitude and bus lag are randomisation ranges whose nominal value is
  zero, not physical constants. Dropping the passive joints rather than locking
  them also puts the degree-of-freedom count back to the robot's own 27.

Hardware joint limits
  The twelve leg joints go back to +/-pi; see :data:`HARDWARE_JOINT_RANGE`.

Everything else is deliberately left alone -- mjlab's XML, so every sensor, site
and name a task pipeline reads stays intact; mjlab's per-servo actuator
characterisation, including the real 11.086 / 6.162 N.m effort limits that
NUbots' own model does not have; and mjlab's contact parameters, so a policy
meets the contacts it trained on.

That last one is worth spelling out, because NUbots' MJCF uses a four times
stiffer contact (``solref`` 0.005 against mjlab's 0.02) and it would have been
easy to import. It is not a randomisation range and not a fidelity bug, so
changing it would just be a second plant. Measured, it does not matter to the
walk engine either: driven at 0.2 and 0.3 m/s for 20 s, this model walks with
either value (0.155 / 0.193 m/s at 0.005, 0.166 / 0.210 m/s at 0.02, upright
throughout in all four). Use
:mod:`~mjlab.asset_zoo.robots.nugus.nugus_nubots_sim_constants` if you want
NUbots' contact model.
"""

from dataclasses import replace

import mujoco

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  FEET_COLLISION,
  NUGUS_ACTUATOR_ARMS,
  NUGUS_ACTUATOR_HEAD,
  NUGUS_ACTUATOR_HIPS,
  NUGUS_ACTUATOR_LEGS,
  NUGUS_XML,
  STAND_BENT_KNEES_KEYFRAME,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

LEG_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_yaw",
  "left_hip_roll",
  "left_hip_pitch",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_yaw",
  "right_hip_roll",
  "right_hip_pitch",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
)

HARDWARE_JOINT_RANGE = (-3.14159, 3.14159)
"""Leg joint travel of the real robot.

Both ``robot.urdf`` and ``nugus.xml`` in the NUbots repository give every leg
joint this range. mjlab's XML instead clamps them to a much narrower window --
``ankle_pitch`` to [-0.6, 1.0], ``left_hip_roll`` to [0.0, 0.6] -- which is a
training artifact: a narrow range keeps a policy out of self-collision and
shrinks what it has to explore.

Those clamps are not neutral for a controller that was tuned on hardware.
Walking at 0.3 m/s the engine commands ``ankle_pitch`` to -0.74 rad on every
step, 0.14 rad past the clamp, and it uses the whole of the hip roll sway. Since
MuJoCo enforces the range in the physics rather than on the command -- mjlab
leaves position actuators ``ctrllimited=False`` on purpose, see
:func:`~mjlab.utils.spec.create_position_actuator` -- the ankle simply stops
where the XML says, several centimetres from where the engine asked. Restoring
the hardware range is a fidelity fix, not a retuning.

It is not, on its own, what keeps the robot upright: on the training model the
widened range delays the fall by 0.08 s and no more. The backlash joints are what
decide that.
"""

EVAL_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
"""Kept at the training model's value, but note that it means less here.

``soft_joint_pos_limit_factor`` scales the joint range down to the band that
``mdp.joint_pos_limits`` penalises and that reset randomisation samples in. It
clamps nothing in the physics, so it has no effect on open-loop playback. It
does mean the soft band on this model is 0.9 * +/-pi rather than 0.9 of a narrow
window, which leaves the joint-limit penalty effectively inert -- harmless for
an evaluation model, and a reason not to train against this one.
"""


def get_eval_spec() -> mujoco.MjSpec:
  """Build the NUgus spec with randomisation at nominal and hardware limits."""
  spec = mujoco.MjSpec.from_file(str(NUGUS_XML))

  for joint in list(spec.joints):
    if joint.name.endswith("_backlash"):
      spec.delete(joint)

  for name in LEG_JOINT_NAMES:
    joint = spec.joint(name)
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.range[0], joint.range[1] = HARDWARE_JOINT_RANGE

  return spec


def get_nugus_eval_robot_cfg() -> EntityCfg:
  """Fresh NUgus config for comparing controllers on equal terms.

  The joint ranges are widened in the spec *before* the actuators are added, so
  each ``<position>`` actuator's informational ``ctrlrange`` -- derived from the
  joint range in :func:`~mjlab.utils.spec.create_position_actuator` -- widens
  with it.
  """
  actuators = tuple(
    replace(actuator, delay_min_lag=0, delay_max_lag=0)
    for actuator in (
      NUGUS_ACTUATOR_ARMS,
      NUGUS_ACTUATOR_HIPS,
      NUGUS_ACTUATOR_LEGS,
      NUGUS_ACTUATOR_HEAD,
    )
  )
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_eval_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=EVAL_SOFT_JOINT_POS_LIMIT_FACTOR,
    ),
  )
