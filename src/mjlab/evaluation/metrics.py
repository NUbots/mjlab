"""Walk metrics, computed from raw simulator state.

Every controller under comparison is measured by *this* code and nothing else.
No controller's own idea of what it is doing is used: the input is
:class:`EvalState`, a plain readout of the robot's root link, and
:meth:`EvalState.from_entity` is the only place the harness gets it.

Only the quantities the comparison figures draw are kept: velocity tracking per
axis, and whether and when the robot fell. A metric that only means something
under a disturbance goes in :mod:`mjlab.evaluation.push` instead, which wraps
:class:`WalkMetrics` rather than reimplementing it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

FALL_UPRIGHT_THRESHOLD = 0.5
"""Torso up-axis component below which the robot counts as fallen.

``cos(60 degrees)``. Well past any pitch a walking gait produces, and reached
long before the robot is horizontal, so the fall time is the moment it stopped
walking rather than the moment it landed.
"""


@dataclass(frozen=True)
class EvalState:
  """Raw simulator readout the metrics are computed from.

  Attributes:
    position_w: Shape ``(N, 3)`` root link position in the world frame.
    quaternion_w: Shape ``(N, 4)`` root link orientation, ``(w, x, y, z)``.
    lin_vel_b: Shape ``(N, 3)`` root link linear velocity in the body frame.
    ang_vel_b: Shape ``(N, 3)`` root link angular velocity in the body frame.
  """

  position_w: torch.Tensor
  quaternion_w: torch.Tensor
  lin_vel_b: torch.Tensor
  ang_vel_b: torch.Tensor

  @classmethod
  def from_entity(cls, entity) -> EvalState:
    """Read the state out of an :class:`~mjlab.entity.Entity`.

    Only the root link is read, so this works on any floating-base robot.
    """
    data = entity.data
    return cls(
      position_w=data.root_link_pos_w,
      quaternion_w=data.root_link_quat_w,
      lin_vel_b=data.root_link_lin_vel_b,
      ang_vel_b=data.root_link_ang_vel_b,
    )


def upright_from_quat(quaternion_w: torch.Tensor) -> torch.Tensor:
  """Body up-axis dotted with world up, from a ``(w, x, y, z)`` quaternion.

  This is the ``[2, 2]`` element of the rotation matrix: 1.0 standing, 0.0 on
  its side, -1.0 upside down.
  """
  x = quaternion_w[:, 1]
  y = quaternion_w[:, 2]
  return 1.0 - 2.0 * (x * x + y * y)


def _planar_twist(state: EvalState) -> torch.Tensor:
  """Shape ``(N, 3)`` body-frame ``(vx, vy, wz)``."""
  return torch.cat((state.lin_vel_b[:, :2], state.ang_vel_b[:, 2:3]), dim=-1)


@dataclass(frozen=True)
class PerEnvMetrics:
  """One row per environment. Every field is a shape ``(N,)`` tensor."""

  command_vx: torch.Tensor
  command_vy: torch.Tensor
  command_wz: torch.Tensor
  survived: torch.Tensor
  """1.0 if the robot was still upright at the end of the run."""
  fall_time: torch.Tensor
  """Seconds until the torso tipped past the threshold; NaN if it never did."""
  achieved_vx: torch.Tensor
  """Mean body-frame forward velocity over the measured window, in m/s."""
  achieved_vy: torch.Tensor
  achieved_wz: torch.Tensor
  """Mean body-frame yaw rate over the measured window, in rad/s."""
  error_vx: torch.Tensor
  """Achieved minus commanded, per axis."""
  error_vy: torch.Tensor
  error_wz: torch.Tensor
  tracking_error: torch.Tensor
  """Norm of the planar velocity error, in m/s."""

  def column_names(self) -> list[str]:
    return [f.name for f in fields(self)]

  def rows(self) -> list[list[float]]:
    """Per-environment rows, in :meth:`column_names` order."""
    columns = [getattr(self, name).tolist() for name in self.column_names()]
    return [list(row) for row in zip(*columns, strict=True)]


class WalkMetrics:
  """Accumulates :class:`EvalState` samples into :class:`PerEnvMetrics`.

  Velocity is averaged only while an environment is upright. What a robot does
  after it has fallen is not walking, and averaging it in would make a robot
  that falls early and slides look slow rather than broken -- which is what
  ``fall_time`` is for.
  """

  def __init__(
    self,
    command_b: torch.Tensor,
    dt: float,
    fall_threshold: float = FALL_UPRIGHT_THRESHOLD,
    warmup_s: float = 0.0,
  ) -> None:
    """
    Args:
      command_b: Shape ``(N, 3)`` commanded ``(vx, vy, wz)`` per environment.
      dt: Seconds between :meth:`record` calls.
      fall_threshold: See :data:`FALL_UPRIGHT_THRESHOLD`.
      warmup_s: Seconds discarded from the front of the run before velocity
        starts averaging. A robot starts from standing, so a mean over the whole
        run would report the acceleration as well as the tracking. Survival is
        *not* windowed: a fall during the warm-up is still a fall, dated from
        the first step.
    """
    self.command_b = command_b
    self.dt = dt
    self.fall_threshold = fall_threshold
    self.warmup_steps = int(round(warmup_s / dt))

    num_envs = command_b.shape[0]
    device = command_b.device
    self._steps = 0
    self._alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    self._sample_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self._velocity_sum = torch.zeros(num_envs, 3, device=device)

  def record(self, state: EvalState) -> None:
    """Accumulate one control step."""
    still_up = upright_from_quat(state.quaternion_w) >= self.fall_threshold
    # The sample on which the robot tips is still counted: it is the last one
    # belonging to the walk, and it is what dates the fall.
    counted = self._alive
    just_fell = counted & ~still_up

    self._steps += 1
    self._fall_step = torch.where(
      just_fell, torch.full_like(self._fall_step, self._steps), self._fall_step
    )

    sampled = counted & (self._steps > self.warmup_steps)
    self._sample_steps = self._sample_steps + sampled.long()
    self._velocity_sum = self._velocity_sum + sampled.float().unsqueeze(
      -1
    ) * _planar_twist(state)

    self._alive = counted & still_up

  def result(self) -> PerEnvMetrics:
    """Reduce the accumulated samples. Safe to call more than once."""
    # An environment that fell inside the warm-up contributed no samples, so
    # its averages would be zeros rather than measurements. Say so instead.
    measured = (self._sample_steps > 0).unsqueeze(-1)
    achieved = torch.where(
      measured,
      self._velocity_sum / self._sample_steps.clamp(min=1).unsqueeze(-1).float(),
      torch.full_like(self._velocity_sum, float("nan")),
    )
    error = achieved - self.command_b

    fell = self._fall_step >= 0
    fall_time = torch.where(
      fell,
      self._fall_step.float() * self.dt,
      torch.full_like(self._fall_step, float("nan"), dtype=torch.float32),
    )
    return PerEnvMetrics(
      command_vx=self.command_b[:, 0],
      command_vy=self.command_b[:, 1],
      command_wz=self.command_b[:, 2],
      survived=(~fell).float(),
      fall_time=fall_time,
      achieved_vx=achieved[:, 0],
      achieved_vy=achieved[:, 1],
      achieved_wz=achieved[:, 2],
      error_vx=error[:, 0],
      error_vy=error[:, 1],
      error_wz=error[:, 2],
      tracking_error=torch.linalg.vector_norm(error[:, :2], dim=-1),
    )


class VelocityTrace:
  """Per-control-step commanded and measured base velocity.

  :class:`WalkMetrics` reduces a run to one row per environment, which is the
  right shape for a command grid and the wrong one for a *profile* run, where
  the command moves during the episode and the interesting quantity is how the
  robot follows it. This records the two side by side, step by step.
  """

  def __init__(self, dt: float) -> None:
    self.dt = dt
    self._command: list[torch.Tensor] = []
    self._achieved: list[torch.Tensor] = []
    self._upright: list[torch.Tensor] = []

  def record(self, command_b: torch.Tensor, state: EvalState) -> None:
    """Append one control step.

    Args:
      command_b: Shape ``(N, 3)`` command in force for this step.
      state: The robot state after the step.
    """
    self._command.append(command_b.detach().to("cpu", torch.float32).clone())
    self._achieved.append(_planar_twist(state).detach().to("cpu", torch.float32))
    self._upright.append(upright_from_quat(state.quaternion_w).detach().cpu())

  @property
  def num_steps(self) -> int:
    return len(self._command)

  def result(self) -> dict[str, torch.Tensor]:
    """Stacked traces.

    Returns:
      ``time`` shape ``(T,)``, ``command`` and ``achieved`` shape ``(T, N, 3)``
      ordered ``(vx, vy, wz)``, and ``upright`` shape ``(T, N)``.
    """
    if not self._command:
      raise RuntimeError("nothing recorded")
    steps = torch.arange(1, self.num_steps + 1, dtype=torch.float32)
    return {
      "time": steps * self.dt,
      "command": torch.stack(self._command),
      "achieved": torch.stack(self._achieved),
      "upright": torch.stack(self._upright),
    }


def write_trace_csv(path: Path, trace: VelocityTrace) -> None:
  """Write a profile run's traces, one row per step per environment."""
  data = trace.result()
  time = data["time"].tolist()
  command = data["command"].tolist()
  achieved = data["achieved"].tolist()
  upright = data["upright"].tolist()

  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "step",
        "time",
        "env",
        "command_vx",
        "command_vy",
        "command_wz",
        "achieved_vx",
        "achieved_vy",
        "achieved_wz",
        "upright",
      ]
    )
    for step, seconds in enumerate(time):
      for env, (cmd, ach, up) in enumerate(
        zip(command[step], achieved[step], upright[step], strict=True)
      ):
        writer.writerow([step, round(seconds, 6), env, *cmd, *ach, up])


def summarise(metrics: PerEnvMetrics) -> dict:
  """Survival and mean tracking error, over the environments that survived."""
  survived = metrics.survived > 0.5

  def mean(values: torch.Tensor) -> float:
    finite = values[survived & values.isfinite()]
    return float(finite.mean()) if finite.numel() else float("nan")

  return {
    "num_envs": int(metrics.survived.numel()),
    "num_survived": int(survived.sum()),
    "survival_rate": float(survived.float().mean()),
    "survivors_mean": {
      name: mean(getattr(metrics, name))
      for name in ("error_vx", "error_vy", "error_wz", "tracking_error")
    },
  }


def save_run(
  output_dir: Path,
  run: dict,
  metrics: PerEnvMetrics,
  summary: dict | None = None,
) -> dict:
  """Write ``per_env.csv`` and ``summary.json``, and return the summary.

  Args:
    output_dir: Directory to write into.
    run: The run's configuration, written as the summary's ``run`` block.
    metrics: The per-environment table; becomes ``per_env.csv``.
    summary: Aggregate to write, if the caller has one of its own. Defaults to
      :func:`summarise`.
  """
  summary = {"run": run, **(summary if summary is not None else summarise(metrics))}
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "per_env.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["env"] + metrics.column_names())
    for index, row in enumerate(metrics.rows()):
      writer.writerow([index] + row)
  with (output_dir / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
  return summary
