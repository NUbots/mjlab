"""RL runner for path tracking tasks.

Extends the velocity runner with the ability to warm-start from a velocity
task checkpoint: the two tasks share their network trunks and every
observation term except the command view, so a velocity checkpoint can be
spliced onto the path-tracking observation layout instead of failing the
resume on mismatched first-layer shapes.
"""

from __future__ import annotations

import math

import torch

from mjlab.managers.observation_manager import ObservationManager
from mjlab.rl.warmstart import ObsLayout, splice_model_state_dict
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def velocity_obs_layout(path_layout: ObsLayout) -> list[tuple[str, int]]:
  """Derive the velocity task's observation layout from a path task's.

  The tasks differ only in the command view: the velocity command is a
  3-dim twist instead of the waypoint window, and there is no critic-only
  ``target_twist`` term.
  """
  layout = []
  for name, dim in path_layout:
    if name == "command":
      layout.append((name, 3))
    elif name != "target_twist":
      layout.append((name, dim))
  return layout


def _group_layout(obs_manager: ObservationManager, group: str) -> list[tuple[str, int]]:
  names = obs_manager.active_terms[group]
  dims = obs_manager.group_obs_term_dim[group]
  return [(n, int(math.prod(d))) for n, d in zip(names, dims, strict=True)]


class PathTrackingOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that can also warm-start from velocity checkpoints.

  ``load()`` inspects the checkpoint's actor input width: a checkpoint that
  already matches the path-tracking layout is loaded normally, while one
  matching the velocity task this configuration was derived from is spliced
  onto the path-tracking layout (shared first-layer columns and
  obs-normalizer statistics copied, new command/``target_twist`` columns
  zero-initialized) with a fresh optimizer state and iteration counter.
  """

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    actor_state = loaded_dict.get("actor_state_dict", {})
    ckpt_width = (
      actor_state["mlp.0.weight"].shape[1] if "mlp.0.weight" in actor_state else None
    )

    obs_manager = self.env.unwrapped.observation_manager
    actor_layout = _group_layout(obs_manager, "actor")
    if ckpt_width is None or ckpt_width == sum(d for _, d in actor_layout):
      return super().load(path, load_cfg, strict, map_location)

    print(
      f"[INFO] Checkpoint at {path} has a {ckpt_width}-wide actor input; "
      "treating it as a velocity-task checkpoint and splicing it onto the "
      "path-tracking observation layout."
    )
    # rsl-rl 4.x key migration, mirroring MjlabOnPolicyRunner.load().
    if "std" in actor_state:
      actor_state["distribution.std_param"] = actor_state.pop("std")
    if "log_std" in actor_state:
      actor_state["distribution.log_std_param"] = actor_state.pop("log_std")

    for group, model in (("actor", self.alg.actor), ("critic", self.alg.critic)):
      target = _group_layout(obs_manager, group)
      source = velocity_obs_layout(target)
      spliced = splice_model_state_dict(
        loaded_dict[f"{group}_state_dict"], target, source
      )
      model.load_state_dict(spliced, strict=True)
      source_dims = dict(source)
      summary = ", ".join(
        f"{name}[{dim}, {'copied' if source_dims.get(name) == dim else 'new'}]"
        for name, dim in target
      )
      print(f"[INFO]   {group}: {summary}")
    # The optimizer state and iteration counter do not transfer across the
    # layout change; training restarts from iteration 0 with fresh Adam
    # state and curricula.
    return {}
