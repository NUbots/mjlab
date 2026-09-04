"""Per-env, per-step scaling of the stability reward group.

:class:`StabilityGate` turns a :class:`~.focus.TrainingFocusCfg` into one
``[B]`` multiplier per step, and :class:`stability_gated` applies it to an
existing reward term without disturbing that term's own configuration.

The gate speed for a channel is ``min(commanded, achieved)``, and that cap
is load-bearing rather than cosmetic. Stability *penalties* shrink as the
gate falls, so a gate reading achieved speed alone would hand the policy a
lever: sprint, and the penalties recede -- available under any command,
including the slow ones where stability is exactly what was wanted.
Capping at the commanded speed means relief only ever arrives when the
curriculum actually asked for speed AND the policy delivered it, which is
the trade the focus config is meant to express.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.focus import CHANNELS, TrainingFocusCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_EPS = 1e-6


class StabilityGate:
  """Evaluates the focus config's stability profiles against live motion.

  One instance per env, shared by every gated reward term (and by the
  diagnostics), with the gate computed once per environment step and
  cached -- eight wrapped terms must not each rebuild it.
  """

  def __init__(self, env: ManagerBasedRlEnv, focus: TrainingFocusCfg):
    self.focus = focus
    self.device = env.device
    self.num_envs = env.num_envs

    # Pad every channel's knots to a common length by repeating the last
    # one. Flat extrapolation makes the repeat exact, so padding costs
    # nothing in fidelity and buys a single batched interpolation.
    knots_per_channel = focus.stability_knots()
    width = max(2, max(len(k) for k in knots_per_channel))
    xs, ys = [], []
    for knots in knots_per_channel:
      padded = list(knots) + [knots[-1]] * (width - len(knots))
      xs.append([x for x, _ in padded])
      ys.append([y for _, y in padded])
    self._xs = torch.tensor(xs, device=self.device, dtype=torch.float32)
    self._ys = torch.tensor(ys, device=self.device, dtype=torch.float32)

    # Command magnitudes are normalized by each channel's target before
    # being turned into blend shares. Without it the shares would compare
    # m/s against rad/s and yaw, whose numbers are simply bigger, would
    # dominate the blend of every mixed command.
    self._target = torch.tensor(
      [max(focus.channel(name).target_speed, _EPS) for name in CHANNELS],
      device=self.device,
      dtype=torch.float32,
    )

    if focus.standing_stability is not None:
      standing = float(focus.standing_stability) * focus.stability_scale
    else:
      standing = sum(
        focus.channel(name).stability.evaluate(0.0) for name in CHANNELS
      ) / len(CHANNELS)
      standing *= focus.stability_scale
    self._standing = standing

    self._cached_step = -1
    self._cached_command: str | None = None
    self._cached_gate = torch.ones(self.num_envs, device=self.device)
    # Population means kept for the diagnostics term. Deliberately left as
    # device tensors: ``compute`` runs every environment step, and pulling
    # thirteen scalars back to the host there would put that many
    # synchronizations in the hot path. The diagnostics term reads them at
    # reset cadence and pays for the transfer once.
    self.last_gate = torch.ones((), device=self.device)
    self.last_emphasis = torch.ones(len(CHANNELS), device=self.device)
    self.last_share = torch.zeros(len(CHANNELS), device=self.device)
    self.last_gate_speed = torch.zeros(len(CHANNELS), device=self.device)

  def channel_speeds(
    self, env: ManagerBasedRlEnv, command_name: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(commanded, gate)`` speeds, both ``[B, 4]``.

    Both are non-negative magnitudes per channel. ``gate`` is the
    commanded magnitude capped by what was actually delivered along the
    commanded direction, so motion the command did not ask for -- drift,
    a shove, an overshoot -- never unlocks relief.
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    asset = env.scene["robot"]
    lin = asset.data.root_link_lin_vel_b
    ang = asset.data.root_link_ang_vel_b

    zero = torch.zeros((), device=command.device)
    cmd_x, cmd_y, cmd_yaw = command[:, 0], command[:, 1], command[:, 2]
    commanded = torch.stack(
      [
        torch.maximum(cmd_x, zero),
        torch.maximum(-cmd_x, zero),
        cmd_y.abs(),
        cmd_yaw.abs(),
      ],
      dim=1,
    )
    # Achieved is projected onto the commanded direction per channel, so a
    # channel only earns credit for the motion that was asked for.
    achieved = torch.stack(
      [
        torch.maximum(lin[:, 0], zero),
        torch.maximum(-lin[:, 0], zero),
        torch.maximum(lin[:, 1] * torch.sign(cmd_y), zero),
        torch.maximum(ang[:, 2] * torch.sign(cmd_yaw), zero),
      ],
      dim=1,
    )
    return commanded, torch.minimum(commanded, achieved)

  def _emphasis(self, speeds: torch.Tensor) -> torch.Tensor:
    """Interpolate every channel's profile at ``speeds`` ``[B, 4]``."""
    per_channel = speeds.t().contiguous()  # [C, B]
    idx = torch.searchsorted(self._xs, per_channel)
    idx = idx.clamp(1, self._xs.shape[1] - 1)
    x0 = self._xs.gather(1, idx - 1)
    x1 = self._xs.gather(1, idx)
    y0 = self._ys.gather(1, idx - 1)
    y1 = self._ys.gather(1, idx)
    # Clamping t to [0, 1] is what produces the flat extrapolation beyond
    # the outermost knots, and it also neutralizes the padded duplicates
    # (x1 == x0), which would otherwise divide by zero.
    t = ((per_channel - x0) / (x1 - x0).clamp(min=_EPS)).clamp(0.0, 1.0)
    return (y0 + t * (y1 - y0)).t()  # [B, C]

  def compute(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The gate multiplier ``[B]`` for this step."""
    commanded, gate_speed = self.channel_speeds(env, command_name)
    emphasis = self._emphasis(gate_speed)

    # Blend by how much of the (target-normalized) ask sits in each
    # channel: a pure forward command reads the forward profile, a
    # forward-plus-turn command reads a mix.
    norm_cmd = commanded / self._target
    total = norm_cmd.sum(dim=1, keepdim=True)
    share = norm_cmd / total.clamp(min=_EPS)
    # ``stability_knots`` already folded in ``stability_scale``, so the
    # blend is the finished gate -- do not scale it a second time.
    gate = (share * emphasis).sum(dim=1)

    # Standing envs (and anything else with a ~zero command) have no
    # direction to blend, so they fall back to the standing emphasis
    # rather than to whatever the degenerate share vector produced.
    standing = total.squeeze(1) < _EPS
    gate = torch.where(standing, torch.full_like(gate, self._standing), gate)

    self.last_gate = gate.mean()
    self.last_emphasis = emphasis.mean(dim=0)
    self.last_share = share.mean(dim=0)
    self.last_gate_speed = gate_speed.mean(dim=0)
    return gate

  def value(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Cached :meth:`compute` -- one evaluation per environment step."""
    step = env.common_step_counter
    if step != self._cached_step or command_name != self._cached_command:
      self._cached_gate = self.compute(env, command_name)
      self._cached_step = step
      self._cached_command = command_name
    return self._cached_gate


def get_stability_gate(
  env: ManagerBasedRlEnv, focus: TrainingFocusCfg
) -> StabilityGate:
  """Fetch the env's gate, creating it on first use.

  Stashed on the env like the competence tracker so the gated reward
  terms and the diagnostics share one instance regardless of construction
  order.
  """
  gate = getattr(env, "_stability_gate", None)
  if gate is None:
    gate = StabilityGate(env, focus)
    setattr(env, "_stability_gate", gate)  # noqa: B010
  elif gate.focus != focus:
    raise ValueError(
      "Two different TrainingFocusCfg instances reached the stability gate. "
      "Every gated term must be built from the same focus config."
    )
  return gate


class stability_gated:
  """Wrap a reward term so its value is scaled by the stability gate.

  The wrapper owns the inner term rather than the manager: it resolves
  the inner term's ``SceneEntityCfg`` params (the manager only scans
  top-level params, and the inner ones are nested a level down),
  instantiates class-based inner terms the way the manager would, and
  forwards ``reset`` and ``debug_vis`` so wrapping is invisible to
  everything outside this class.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    inner_func: Any = cfg.params["inner_func"]
    inner_params: dict[str, Any] = cfg.params["inner_params"]
    for value in inner_params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    if inspect.isclass(inner_func):
      # Constructed exactly the way ManagerBase._resolve_common_term_cfg
      # would have, had the manager seen the inner term directly. The
      # untyped alias keeps the class-object narrowing from turning this
      # into a call to ``type()``.
      factory: Any = inner_func
      inner_cfg = RewardTermCfg(func=inner_func, weight=cfg.weight, params=inner_params)
      inner_func = factory(cfg=inner_cfg, env=env)
    self._inner = inner_func
    self._inner_params = inner_params
    self._gate = get_stability_gate(env, cfg.params["focus"])
    if hasattr(self._inner, "debug_vis"):
      self.debug_vis = self._inner.debug_vis

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    inner_reset = getattr(self._inner, "reset", None)
    if inner_reset is not None:
      inner_reset(env_ids=env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    inner_func: Any,
    inner_params: dict[str, Any],
    focus: TrainingFocusCfg,
    command_name: str,
  ) -> torch.Tensor:
    # The manager passes every cfg param as a kwarg, so this signature has
    # to accept the ones consumed in __init__.
    del inner_func, inner_params, focus
    return self._inner(env, **self._inner_params) * self._gate.value(env, command_name)


def make_gated_term(
  term_cfg: RewardTermCfg, focus: TrainingFocusCfg, command_name: str
) -> RewardTermCfg:
  """Return ``term_cfg`` wrapped in :class:`stability_gated`.

  The weight is carried across unchanged: the gate scales the term's
  *value*, so the configured weight keeps meaning "the pressure at full
  emphasis", exactly like the competence ladder's peak weights.
  """
  return RewardTermCfg(
    func=stability_gated,
    weight=term_cfg.weight,
    params={
      "inner_func": term_cfg.func,
      "inner_params": term_cfg.params,
      "focus": focus,
      "command_name": command_name,
    },
  )
