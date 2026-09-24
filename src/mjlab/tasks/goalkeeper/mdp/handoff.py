"""Walk the goalie before a shot, and hand it to the block policy as the robot does.

On the robot the goalie is walking to its spot (``skill::K1WalkPolicy``) for much of
the time, and ``planning::PlanSave`` hands it to the block policy when it has to: some
time after a kick is seen (BLOCK), or once the ball is placed near it (GUARD). The
block policy, trained only from a standing start, fell over taking the goalie
mid-stride. Two things on the robot soften that (NUbots_K1, PlanSave and
K1BlockPolicy):

- the hand-off waits for both feet to be down, for at most ``max_wait``;
- the block policy starts from the observation frames it recorded while the walk had
  the robot, with the walk's commands as its previous actions and an inactive command,
  rather than from its first frame repeated as if it had been standing still.

This action term reproduces that here. A frozen walk policy drives the robot through
part of a shot cycle, and the learner takes over at the hand-off. Until then the
learner's actions are ignored, and ``while_blocking`` masks its rewards, since it is
not the one acting. The applied action, the walk's while it walks, is written back as
the step's action, so the block policy's previous-action observation and the
action-rate costs see what the robot was actually commanded, as on the robot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.rl.obs_history import HISTORY_GROUP, HistoryActor
from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def load_frozen_walk(
  path: str | Path,
  window: int,
  obs_dim: int,
  action_dim: int,
  hidden_dims: tuple[int, ...],
  history_cfg: dict[str, Any],
  activation: str,
  device: str,
) -> torch.nn.Module:
  """A walk policy's deterministic actor: a ``[B, window * obs_dim]`` window in,
  actions out, as the robot runs its ONNX. ``path`` holds its ``actor_state_dict``."""
  obs = TensorDict(
    {
      "actor": torch.zeros(1, obs_dim),
      HISTORY_GROUP: torch.zeros(1, window, obs_dim),
    },
    batch_size=[1],
  )
  actor = HistoryActor(
    obs,
    {"actor": ["actor", HISTORY_GROUP]},
    "actor",
    action_dim,
    history_cfg=dict(history_cfg),
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "log",
    },
  )
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  actor.load_state_dict(checkpoint["actor_state_dict"])
  policy = actor.as_onnx(verbose=False).to(device).eval()
  for parameter in policy.parameters():
    parameter.requires_grad_(False)
  return policy


class WalkHandoffAction(JointPositionAction):
  """Joint position targets from a frozen walk until the hand-off, then the learner."""

  cfg: WalkHandoffActionCfg

  def __init__(self, cfg: WalkHandoffActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    n, device = self.num_envs, self.device
    self._walk = load_frozen_walk(
      cfg.walk_checkpoint,
      window=cfg.walk_window,
      obs_dim=cfg.walk_obs_dim,
      action_dim=self.action_dim,
      hidden_dims=cfg.walk_hidden_dims,
      history_cfg=cfg.walk_history_cfg,
      activation=cfg.walk_activation,
      device=device,
    )
    body_ids, _ = self._entity.find_bodies(cfg.foot_body_names, preserve_order=True)
    self._foot_ids = body_ids

    zeros = lambda: torch.zeros(n, device=device)  # noqa: E731
    flags = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
    # Whether the walk has the robot, and the plan for the current shot cycle.
    self.walking = flags()
    self._guard = flags()
    self._gated = flags()
    self._trigger_at = zeros()
    self._waited = zeros()
    self._last_shot = torch.full((n,), -1, dtype=torch.long, device=device)
    # Seconds since the learner last took over, for the falls-after-hand-off metric.
    self._since_handoff = torch.full((n,), 1e9, device=device)

    # Counts for the logs, fading at log_decay per step so the rates follow the policy
    # as it is now rather than averaging over the whole run.
    self._handoffs = 0.0
    self._handoffs_timed_out = 0.0
    self._handoffs_waited_s = 0.0
    self._falls_after_handoff = 0.0

  @property
  def shot(self) -> ShotCommand:
    term = self._env.command_manager.get_term(self.cfg.shot_command)
    assert isinstance(term, ShotCommand)
    return term

  def process_actions(self, actions: torch.Tensor) -> None:
    self._schedule()
    applied = actions
    if bool(self.walking.any()):
      window = self._env.obs_buf[self.cfg.walk_obs_group]
      assert isinstance(window, torch.Tensor)
      with torch.inference_mode():
        walk_actions = self._walk(window.reshape(self.num_envs, -1))
      applied = torch.where(self.walking.unsqueeze(-1), walk_actions.clone(), actions)
    # What the robot was actually commanded: the previous-action observation and the
    # action-rate costs are about that, not about a learner output that was ignored.
    self._env.action_manager.action[:] = applied
    super().process_actions(applied)
    self._log()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    # A fall within a second of the learner taking over is the hand-off's to answer
    # for. Read on the terminal state, before the counters are cleared.
    terminated = self._env.termination_manager.terminated[env_ids]
    soon = self._since_handoff[env_ids] < 1.0
    self._falls_after_handoff += int((terminated & soon).sum())
    self.walking[env_ids] = False
    self._waited[env_ids] = 0.0
    self._since_handoff[env_ids] = 1e9

  ##
  # Shot cycles.
  ##

  def _schedule(self) -> None:
    shot = self.shot
    dt = self._env.step_dt
    self._since_handoff += dt
    keep = 1.0 - self.cfg.log_decay
    self._handoffs *= keep
    self._handoffs_timed_out *= keep
    self._handoffs_waited_s *= keep
    self._falls_after_handoff *= keep

    new_cycle = shot.shot_count != self._last_shot
    if bool(new_cycle.any()):
      ids = new_cycle.nonzero(as_tuple=False).flatten()
      self._last_shot[ids] = shot.shot_count[ids]
      self._start_cycle(ids)

    if not bool(self.walking.any()):
      return
    since_kick = shot.time_since_resample - shot.kick_delay
    triggered = self.walking & torch.where(
      self._guard,
      shot.time_since_resample >= self._trigger_at,
      shot.kicked & (since_kick >= self._trigger_at),
    )
    feet = self._entity.data.body_link_pos_w[:, self._foot_ids, 2]
    feet_down = (feet[:, 0] - feet[:, 1]).abs() < self.cfg.foot_height_tolerance
    timed_out = self._waited >= self.cfg.max_wait
    release = triggered & (~self._gated | feet_down | timed_out)
    self._waited += torch.where(triggered & ~release, dt, 0.0)

    if bool(release.any()):
      self._handoffs += int(release.sum())
      self._handoffs_timed_out += int((release & self._gated & ~feet_down).sum())
      self._handoffs_waited_s += float(self._waited[release].sum())
      self._since_handoff[release] = 0.0
      self.walking &= ~release

  def _start_cycle(self, env_ids: torch.Tensor) -> None:
    """Plan a new shot cycle: walk or not, and when and how to hand over."""
    cfg, shot = self.cfg, self.shot
    n = len(env_ids)

    def uniform(bounds: tuple[float, float]) -> torch.Tensor:
      return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, device=self.device)

    walk = torch.rand(n, device=self.device) < cfg.walk_probability
    guard = torch.rand(n, device=self.device) < cfg.guard_probability
    gated = torch.rand(n, device=self.device) < cfg.gated_probability

    # Give a walking cycle long enough for the gait to settle before the kick.
    shot.kick_delay[env_ids] += torch.where(walk, uniform(cfg.extra_walk_time), 0.0)
    kick_delay = shot.kick_delay[env_ids]
    # GUARD: over to the block policy some time before the kick, once walking.
    guard_at = cfg.min_walk_time + torch.rand(n, device=self.device) * (
      kick_delay - cfg.min_walk_time
    ).clamp(min=0.0)
    # BLOCK: over to it once PlanSave has seen the kick and committed.
    block_at = uniform(cfg.reaction_time)

    self.walking[env_ids] = walk
    self._guard[env_ids] = guard
    self._gated[env_ids] = gated
    self._trigger_at[env_ids] = torch.where(guard, guard_at, block_at)
    self._waited[env_ids] = 0.0

  def _log(self) -> None:
    log = self._env.extras.setdefault("log", {})
    log["Handoff/walking"] = float(self.walking.float().mean())
    handoffs = max(self._handoffs, 1e-6)
    log["Handoff/timed_out_rate"] = self._handoffs_timed_out / handoffs
    log["Handoff/mean_wait_s"] = self._handoffs_waited_s / handoffs
    log["Handoff/fall_rate"] = self._falls_after_handoff / handoffs


@dataclass(kw_only=True)
class WalkHandoffActionCfg(JointPositionActionCfg):
  walk_checkpoint: str
  """A walk policy's actor weights (``actor_state_dict``), trained in the velocity
  task with the same robot, action and observation-history setup."""
  walk_obs_group: str = "walk_history"
  """The observation group holding the walk's own history window."""
  walk_window: int = 25
  walk_obs_dim: int = 72
  walk_hidden_dims: tuple[int, ...] = (512, 256, 128)
  walk_history_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "z_dim": 16,
      "tcn_channels": (32, 32),
      "tcn_kernel": 5,
      "tcn_stride": 2,
    }
  )
  walk_activation: str = "elu"
  shot_command: str = "shot"

  walk_probability: float = 0.5
  """Share of shot cycles the goalie walks through until the hand-off. The rest start
  from the block policy's own stance, as run 12 was trained."""
  guard_probability: float = 0.3
  """Share of walking cycles handed over before the kick, as PlanSave's GUARD does for
  a ball placed near; the rest are handed over after it (BLOCK)."""
  reaction_time: tuple[float, float] = (0.15, 0.35)
  """Kick to PlanSave committing to a block (s). NUSim through the stack: a median
  0.23 s, 0.17-0.31 s for 10-90%."""
  min_walk_time: float = 0.3
  """Earliest a GUARD hand-off comes after the cycle starts (s)."""
  extra_walk_time: tuple[float, float] = (0.0, 1.0)
  """Added to a walking cycle's kick delay (s), so the gait is under way at the kick."""
  gated_probability: float = 0.75
  """Share of hand-offs that wait for both feet down, as PlanSave does. The rest go at
  whatever point of the stride they fall on, as when PlanSave's wait runs out."""
  foot_height_tolerance: float = 0.005
  """Feet level to within this (m) count as both down: PlanSave's test, on the ankles
  (the foot links' origins)."""
  max_wait: float = 0.35
  foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link")
  log_decay: float = 0.002
  """How fast the logged hand-off rates forget, per step: about the last 500 steps, 20
  iterations of 24."""

  def build(self, env: ManagerBasedRlEnv) -> WalkHandoffAction:
    return WalkHandoffAction(self, env)


##
# Observations and rewards around the hand-off.
##


def _handoff(env: ManagerBasedRlEnv, action_name: str) -> WalkHandoffAction:
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, WalkHandoffAction)
  return term


def block_command(
  env: ManagerBasedRlEnv, command_name: str, action_name: str = "joint_pos"
) -> torch.Tensor:
  """The block command, inactive (all zeros) while the walk has the robot: the frames
  K1BlockPolicy records then are built with an inactive command."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  walking = _handoff(env, action_name).walking.unsqueeze(-1)
  return torch.where(walking, torch.zeros_like(command), command)


def while_blocking(
  func: Callable[..., torch.Tensor], action_name: str = "joint_pos"
) -> Callable[..., torch.Tensor]:
  """A reward term paid only while the learner has the robot, not the walk."""

  def masked(env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    value = func(env, **params)
    return torch.where(_handoff(env, action_name).walking, 0.0, value)

  masked.__name__ = f"while_blocking_{getattr(func, '__name__', 'term')}"
  return masked
