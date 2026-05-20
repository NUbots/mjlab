"""NUbots Nugus arm tracking environment configuration."""

import mujoco

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  FEET_COLLISION,
  NUGUS_ACTION_SCALE,
  NUGUS_ARTICULATION,
  NUGUS_XML,
  STAND_BENT_KNEES_KEYFRAME,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.arm_tracking import mdp
from mjlab.tasks.arm_tracking.arm_tracking_env_cfg import (
  make_arm_tracking_env_cfg,
)
from mjlab.tasks.arm_tracking.mdp.commands import SinusoidalJointCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def _get_nugus_fixed_spec() -> mujoco.MjSpec:
  """Load nugus spec with the freejoint removed (fixed base)."""
  spec = mujoco.MjSpec.from_file(str(NUGUS_XML))
  for joint in spec.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      spec.delete(joint)
      break
  return spec


def _get_nugus_fixed_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=STAND_BENT_KNEES_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=_get_nugus_fixed_spec,
    articulation=NUGUS_ARTICULATION,
  )


NUGUS_ARM_JOINT_PAIRS = {
  "shoulder_pitch": ("left_shoulder_pitch", "right_shoulder_pitch"),
  "elbow_pitch": ("left_elbow_pitch", "right_elbow_pitch"),
}


def nubots_nugus_arm_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create NUbots Nugus arm tracking configuration."""
  cfg = make_arm_tracking_env_cfg()

  cfg.scene.entities = {"robot": _get_nugus_fixed_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    r".*shoulder_pitch",
    r".*elbow_pitch",
  )
  joint_pos_action.scale = {
    k: v
    for k, v in NUGUS_ACTION_SCALE.items()
    if "shoulder_pitch" in k or "elbow_pitch" in k
  }

  arm_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(r".*shoulder_pitch", r".*elbow_pitch"),
  )
  for group in cfg.observations.values():
    group.terms["joint_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"asset_cfg": arm_joint_cfg},
    )
    group.terms["joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      params={"asset_cfg": arm_joint_cfg},
    )

  sinusoid_cmd = cfg.commands["sinusoid"]
  assert isinstance(sinusoid_cmd, SinusoidalJointCommandCfg)
  sinusoid_cmd.joint_pairs = NUGUS_ARM_JOINT_PAIRS

  cfg.rewards = {
    "joint_pos_tracking": RewardTermCfg(
      func=mdp.joint_tracking_exp,
      weight=2.0,
      params={"command_name": "sinusoid", "std": 0.25},
    ),
    "joint_vel_tracking": RewardTermCfg(
      func=mdp.joint_velocity_tracking_exp,
      weight=1.0,
      params={"command_name": "sinusoid", "std": 1.0},
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.05,
    ),
  }

  cfg.viewer.body_name = "torso"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    sinusoid_cmd.fixed_frequency = 1.0
    sinusoid_cmd.fixed_amplitude = 1.0
    sinusoid_cmd.fixed_pair_index = 0  # 0 = shoulder_pitch

  return cfg
