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
backlash joints, four times stiffer contacts, and a 100 N.m force range against
the real servos' 11.09 -- and those differences flatter an open-loop controller.
Running the quintic engine here and a policy on the default NUgus would confound
the controller with the model. Use
:func:`~mjlab.asset_zoo.robots.nugus.nugus_nubots_sim_constants.get_nugus_eval_robot_cfg`
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
from mjlab.asset_zoo.robots.nugus.nugus_constants import STAND_BENT_KNEES_KEYFRAME
from mjlab.entity import EntityCfg

NUGUS_NUBOTS_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nugus" / "xmls" / "nugus_nubots.xml"
)
assert NUGUS_NUBOTS_XML.exists(), f"XML not found: {NUGUS_NUBOTS_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(NUGUS_NUBOTS_XML))


def get_nugus_nubots_robot_cfg() -> EntityCfg:
  """NUgus with NUbots' own actuators and dynamics, untouched.

  ``articulation`` is left unset: the XML already declares its twenty
  ``<position>`` actuators, so mjlab must not inject its own. Collisions are
  likewise left as the XML defines them, since NUbots pairs the robot's
  ``contype``/``conaffinity`` with their scene's floor.
  """
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    spec_fn=get_spec,
  )
