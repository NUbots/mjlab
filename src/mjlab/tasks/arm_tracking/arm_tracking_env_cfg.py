"""Arm tracking task configuration.

Sinusoidal joint tracking for sim-to-real motor validation. The robot is
fixed in place (no freejoint) so balance is not a factor — the task
isolates motor response accuracy.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.arm_tracking import mdp
from mjlab.tasks.arm_tracking.mdp.commands import SinusoidalJointCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_arm_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base arm tracking task configuration."""

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel,
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel,
          noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "command": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "sinusoid"},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "command": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "sinusoid"},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "sinusoid": SinusoidalJointCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 6.0),
      joint_pairs={},  # Set per-robot.
      frequency_range=(0.5, 2.0),
      amplitude_range=(0.1, 0.4),
    )
  }

  rewards: dict[str, RewardTermCfg] = {
    "joint_pos_tracking": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=0.0,  # Placeholder, replaced by tracking reward.
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      fovy=55.0,
      elevation=10.0,
      azimuth=150.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
