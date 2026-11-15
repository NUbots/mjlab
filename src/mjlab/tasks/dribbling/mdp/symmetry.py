"""Left-right mirror symmetry for the K1 ball tasks.

Used by rsl_rl's symmetry extension: every transition is also trained on as
its mirror image across the robot's sagittal (x-z) plane, so whatever the one
policy learns for one foot it learns for the other. Nothing rewards
two-footedness otherwise, and without this the dribbling policy settled on
its right foot.

Under the reflection y -> -y:

- polar vectors (velocity, gravity, positions): ``(x, -y, z)``;
- angular velocity (an axial vector): ``(-x, y, -z)``;
- joints: left and right swap, and a joint about the x or z axis (roll, yaw)
  changes sign while one about y (pitch) does not. Joint offsets from the
  default pose and joint-position actions transform the same way because the
  default pose and the action scales are themselves mirror symmetric;
- per-foot quantities swap feet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.envs.mdp.actions.actions import BaseAction

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

_VECTOR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)


def _mirror_name(name: str) -> str:
  if "Left" in name:
    return name.replace("Left", "Right")
  if "Right" in name:
    return name.replace("Right", "Left")
  return name


def _joint_mirror(names: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Index permutation and signs such that ``mirrored = x[..., perm] * sign``."""
  perm = [names.index(_mirror_name(n)) for n in names]
  sign = [1.0 if n.lower().endswith("pitch") else -1.0 for n in names]
  for n in names:
    assert n.lower().endswith(("pitch", "roll", "yaw")), f"Unknown joint axis: {n}"
  return (
    torch.tensor(perm, device=device),
    torch.tensor(sign, device=device),
  )


class _Mirror:
  """Per-env cache of the mirror maps for every observation term and the action."""

  def __init__(self, env: ManagerBasedRlEnv):
    device = env.device
    robot = env.scene["robot"]
    self.joint_perm, self.joint_sign = _joint_mirror(list(robot.joint_names), device)
    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, BaseAction)
    action_names = list(action_term.target_names)
    self.action_perm, self.action_sign = _joint_mirror(action_names, device)
    assert len(action_names) == env.action_manager.total_action_dim

    vector = torch.tensor(_VECTOR, device=device)
    axial = torch.tensor(_AXIAL, device=device)
    swap_feet = torch.tensor([1, 0], device=device)

    def scale(signs: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
      return lambda x: x * signs

    def joints(x: torch.Tensor) -> torch.Tensor:
      return x[..., self.joint_perm] * self.joint_sign

    def feet_scalar(x: torch.Tensor) -> torch.Tensor:
      return x[..., swap_feet]

    def feet_force(x: torch.Tensor) -> torch.Tensor:
      per_foot = x.reshape(*x.shape[:-1], 2, 3)[..., swap_feet, :] * vector
      return per_foot.reshape(x.shape)

    # Frame size and mirror map of each observation term, by name.
    n = len(robot.joint_names)
    self.terms: dict[str, tuple[int, Callable[[torch.Tensor], torch.Tensor]]] = {
      "base_lin_vel": (3, scale(vector)),
      "base_ang_vel": (3, scale(axial)),
      "projected_gravity": (3, scale(vector)),
      "joint_pos": (n, joints),
      "joint_vel": (n, joints),
      "actions": (len(action_names), self.mirror_actions),
      "command": (2, scale(torch.tensor([1.0, -1.0], device=device))),
      "ball": (4, scale(torch.tensor([1.0, -1.0, 1.0, 1.0], device=device))),
      "ball_pos": (3, scale(vector)),
      "ball_vel": (3, scale(vector)),
      "foot_height": (2, feet_scalar),
      "foot_air_time": (2, feet_scalar),
      "foot_contact": (2, feet_scalar),
      "foot_contact_forces": (6, feet_force),
    }
    manager = env.observation_manager
    self.layout = {
      group: list(zip(manager.active_terms[group], dims, strict=True))
      for group, dims in manager.group_obs_term_dim.items()
    }

  def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
    return actions[..., self.action_perm] * self.action_sign

  def mirror_group(self, group: str, obs: torch.Tensor) -> torch.Tensor:
    out = []
    start = 0
    for name, dims in self.layout[group]:
      width = dims[0]
      frame, fn = self.terms[name]
      block = obs[:, start : start + width]
      # Each term is its history flattened: [frames, frame_size].
      frames = block.reshape(block.shape[0], width // frame, frame)
      out.append(fn(frames).reshape(block.shape))
      start += width
    assert start == obs.shape[1], f"{group}: layout {start} != obs {obs.shape[1]}"
    return torch.cat(out, dim=-1)


def _get_mirror(env) -> _Mirror:
  unwrapped = env.unwrapped
  mirror = getattr(unwrapped, "_dribble_mirror", None)
  if mirror is None:
    mirror = _Mirror(unwrapped)
    unwrapped._dribble_mirror = mirror
  return mirror


def mirror_augmentation(
  env,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl_rl symmetry function: return the batch followed by its mirror image."""
  mirror = _get_mirror(env)
  obs_out = None
  if obs is not None:
    batch = obs.batch_size[0]
    obs_out = TensorDict(
      {
        group: torch.cat([value, mirror.mirror_group(str(group), value)], dim=0)
        for group, value in obs.items()
        if isinstance(value, torch.Tensor)
      },
      batch_size=[2 * batch],
      device=obs.device,
    )
  actions_out = None
  if actions is not None:
    actions_out = torch.cat([actions, mirror.mirror_actions(actions)], dim=0)
  return obs_out, actions_out
