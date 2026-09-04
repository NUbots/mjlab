"""Translate a :class:`~.focus.TrainingFocusCfg` onto a velocity env config.

:func:`apply_training_focus` is the single entry point. It touches three
things and nothing else:

* the two attainment reward weights and their channel weights, which is
  how "I want forward speed more than anything else" reaches the policy;
* the stability reward group, each term wrapped so its value is scaled by
  the live :class:`~.stability_gate.StabilityGate`;
* the ``command_vel`` curriculum's stages, derived from each channel's
  ``target_speed`` instead of hand-written.

Deliberately NOT touched: :class:`~.competence.CompetenceThresholds`. The
competence gate decides when penalty pressure is *safe*, and coupling it
to the focus dial would let a speed-focused run promote penalties onto a
policy that is falling over. Safety and preference stay independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.competence import _NUM_STEPS_PER_ENV
from mjlab.tasks.velocity.mdp.curriculums import VelocityStage
from mjlab.tasks.velocity.mdp.focus import CHANNELS, TrainingFocusCfg
from mjlab.tasks.velocity.mdp.stability_gate import (
  get_stability_gate,
  make_gated_term,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

LINEAR_ATTAINMENT_TERM = "track_linear_velocity_attainment"
ANGULAR_ATTAINMENT_TERM = "track_angular_velocity_attainment"


def command_stages(focus: TrainingFocusCfg) -> list[VelocityStage]:
  """Derive the command curriculum's stages from the channel targets.

  Each stage commands the same *fraction* of every channel's target, so
  the shape of the command box is fixed by the focus and only its size
  ramps. Starting inside the robot's reach gives the attainment gradient
  something to bite on and lets the competence gate promote; the top of
  the ramp is the focus's stated targets.
  """
  n = focus.command_stages
  lin_x, lin_y, ang_z = focus.command_ranges()
  stages: list[VelocityStage] = []
  for i in range(n):
    frac = (
      1.0
      if n == 1
      else focus.command_ramp_start + (1.0 - focus.command_ramp_start) * i / (n - 1)
    )
    iters = 0 if n == 1 else round(focus.command_ramp_iters * i / (n - 1))
    stages.append(
      {
        "step": iters * _NUM_STEPS_PER_ENV,
        "lin_vel_x": (round(lin_x[0] * frac, 4), round(lin_x[1] * frac, 4)),
        "lin_vel_y": (round(lin_y[0] * frac, 4), round(lin_y[1] * frac, 4)),
        "ang_vel_z": (round(ang_z[0] * frac, 4), round(ang_z[1] * frac, 4)),
      }
    )
  return stages


def attainment_weights(focus: TrainingFocusCfg) -> tuple[float, float]:
  """Return the ``(linear, angular)`` attainment reward weights.

  The linear term's channel weights are ratios -- the weighted projection
  is scale-invariant in them -- so they steer *where* the term looks
  without changing how loud it is. Its loudness therefore takes the mean
  of the three linear channels, which keeps every channel's ``speed``
  weight meaning the same thing globally: raise them all and linear
  tracking matters more, raise one and it matters more than its
  neighbours. Yaw is a separate term, so its weight multiplies directly.
  """
  linear_mean = sum(focus.channel(c).speed for c in CHANNELS[:3]) / 3.0
  linear = focus.linear_attainment_weight * focus.speed_scale * linear_mean
  angular = focus.angular_attainment_weight * focus.speed_scale * focus.yaw.speed
  return linear, angular


def apply_training_focus(
  cfg: ManagerBasedRlEnvCfg,
  focus: TrainingFocusCfg,
  *,
  command_name: str = "twist",
  add_diagnostics: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Apply ``focus`` to ``cfg`` in place, returning it for chaining."""
  linear_weight, angular_weight = attainment_weights(focus)

  for name in (LINEAR_ATTAINMENT_TERM, ANGULAR_ATTAINMENT_TERM):
    if name not in cfg.rewards:
      raise KeyError(
        f"Reward term '{name}' is missing, so the focus config has no linear "
        "attainment channel to weight. It is defined in make_velocity_env_cfg."
      )
  cfg.rewards[LINEAR_ATTAINMENT_TERM].weight = linear_weight
  cfg.rewards[LINEAR_ATTAINMENT_TERM].params["channel_weights"] = (
    focus.forward.speed,
    focus.backward.speed,
    focus.strafe.speed,
  )
  cfg.rewards[ANGULAR_ATTAINMENT_TERM].weight = angular_weight

  missing = [name for name in focus.stability_terms if name not in cfg.rewards]
  if missing:
    raise KeyError(
      f"Stability terms {missing} are not reward terms on this config. "
      f"Available: {sorted(cfg.rewards)}."
    )
  for name in focus.stability_terms:
    cfg.rewards[name] = make_gated_term(cfg.rewards[name], focus, command_name)

  if "command_vel" not in cfg.curriculum:
    raise KeyError(
      "Curriculum term 'command_vel' is missing, so the focus config has no "
      "command ramp to drive."
    )
  cfg.curriculum["command_vel"].params["velocity_stages"] = command_stages(focus)

  if add_diagnostics:
    cfg.curriculum["focus_diagnostics"] = CurriculumTermCfg(
      func=focus_diagnostics,
      params={"focus": focus, "command_name": command_name},
    )
  return cfg


class focus_diagnostics:
  """Log-only curriculum term publishing the live gate.

  Without this, two runs at different focus settings are only comparable
  through their outcomes: the gate is a per-step, per-env quantity that
  nothing else in the logs reveals, so a profile that never leaves its
  low-speed knot looks exactly like one that is working.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._gate = get_stability_gate(env, cfg.params["focus"])

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    focus: TrainingFocusCfg,
    command_name: str,
  ) -> dict[str, torch.Tensor]:
    del env_ids, focus, command_name
    gate = self._gate
    # Evaluate rather than read the cached scalars: if every stability
    # term happens to sit at weight zero the reward manager skips them
    # all, and the cached values would silently go stale.
    gate.value(env, self._command_name)
    emphasis = gate.last_emphasis.cpu()
    share = gate.last_share.cpu()
    speed = gate.last_gate_speed.cpu()
    out: dict[str, torch.Tensor] = {"gate": gate.last_gate.cpu()}
    for i, name in enumerate(CHANNELS):
      out[f"emphasis_{name}"] = emphasis[i]
      out[f"share_{name}"] = share[i]
      out[f"gate_speed_{name}"] = speed[i]
    return out
