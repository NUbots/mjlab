"""NUgus exactly as NUbots' own MuJoCo simulator defines it.

A verbatim copy of ``shared/utility/platform/models/nugus/nugus.xml`` from the
NUbots repository, with only ``meshdir`` retargeted so it reuses the meshes
already vendored alongside the default NUgus. Actuators, joint dynamics and
contact parameters are the XML's own -- nothing is injected by mjlab.

**What this is for.** Validating the ported quintic walk engine against the
simulator NUbots actually runs. Driving this model with the port and comparing
against a NUbots run isolates any remaining discrepancy to the controller,
because the model is byte-identical to theirs.

**What this is not for.** Comparing controllers against each other. This model
differs from the default NUgus in ways that materially change stability -- no
backlash joints, which is on its own the difference between the quintic engine
walking and falling; four times stiffer contacts; and a 100 N.m force range
against the real servos' 11.09 -- and those differences flatter an open-loop
controller. Running the quintic engine here and a policy on the default NUgus
would confound the controller with the model. Use
:func:`~mjlab.asset_zoo.robots.nugus.nugus_eval_constants.get_nugus_eval_robot_cfg`
for that, which puts both controllers on one model.

**Caveat for task pipelines.** This XML carries NUbots' sensor names
(``accelerometer``, ``gyro``) and lacks the eight ``*_foot_c[0-3]`` corner sites
that mjlab's NUgus velocity task reads for its foot rewards. It is therefore
usable for standalone playback but not as a drop-in for that task without
adding those.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.asset_zoo.robots.nugus.nugus_constants import STAND_BENT_KNEES_KEYFRAME
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

NUGUS_NUBOTS_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nugus" / "xmls" / "nugus_nubots.xml"
)
assert NUGUS_NUBOTS_XML.exists(), f"XML not found: {NUGUS_NUBOTS_XML}"


MOTOR_JOINT_NAMES: tuple[str, ...] = (
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
  "neck_yaw",
  "head_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow_pitch",
)
"""The twenty joints NUbots' model actuates, one ``<position>`` element each."""


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(NUGUS_NUBOTS_XML))


NUGUS_NUBOTS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=MOTOR_JOINT_NAMES),),
)
"""Adopts the XML's own actuators rather than adding any.

:class:`~mjlab.actuator.XmlActuatorCfg` wraps the ``<position>`` elements that
are already there, leaving their ``kp``, ``forcerange`` and ``ctrlrange`` exactly
as NUbots wrote them, so the compiled model is unchanged. Without this the model
still simulates, but nothing in mjlab knows how to command it -- joint position
targets have nowhere to go -- so batched evaluation cannot drive it.
"""


def get_nugus_nubots_robot_cfg() -> EntityCfg:
  """NUgus with NUbots' own actuators and dynamics, untouched.

  Collisions are left as the XML defines them, since NUbots pairs the robot's
  ``contype``/``conaffinity`` with their scene's floor.
  """
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    spec_fn=get_spec,
    articulation=NUGUS_NUBOTS_ARTICULATION,
  )
