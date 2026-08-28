"""Walk metrics, computed from raw simulator state.

The point of this module is that both engines under comparison -- the ported
NUbots quintic walk and a learned policy -- are measured by *this* code and
nothing else. Neither engine's own notion of what it is doing enters: the input
is :class:`EvalState`, which is a plain readout of the robot's root link and
feet, and :meth:`EvalState.from_entity` is the single place either harness
obtains it.

Adding a metric is a three-line change, all in this file: a buffer in
:meth:`WalkMetrics.__init__`, its update in :meth:`WalkMetrics.record`, and a
field in :class:`PerEnvMetrics`. The CSV columns and the JSON summary are
derived from :class:`PerEnvMetrics`, so nothing downstream needs touching.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

FALL_UPRIGHT_THRESHOLD = 0.5
"""Torso up-axis component below which the robot counts as fallen.

``cos(60 degrees)``. Well past any pitch a walking gait produces -- the deployed
stance sits at 0.97 -- and reached long before the robot is horizontal, so the
fall time is the moment it stopped walking rather than the moment it landed.
"""

FOOT_HEIGHT_DEAD_BAND = 0.01
"""Foot height difference, in metres, that counts as a swap of the lower foot.

Matches the dead band ``SensorFilter`` uses for its ``Z_HEIGHT`` foot-down
method, so cadence counts the same transitions the robot's own foot-down
detector would.
"""


@dataclass(frozen=True)
class EvalState:
  """Raw simulator readout the metrics are computed from.

  Attributes:
    position_w: Shape ``(N, 3)`` root link position in the world frame.
    quaternion_w: Shape ``(N, 4)`` root link orientation, ``(w, x, y, z)``.
    lin_vel_b: Shape ``(N, 3)`` root link linear velocity in the body frame.
    ang_vel_b: Shape ``(N, 3)`` root link angular velocity in the body frame.
    foot_pos_w: Shape ``(N, 2, 3)`` left and right foot body positions in the
      world frame, or ``None`` to skip the cadence metric.
  """

  position_w: torch.Tensor
  quaternion_w: torch.Tensor
  lin_vel_b: torch.Tensor
  ang_vel_b: torch.Tensor
  foot_pos_w: torch.Tensor | None = None

  @classmethod
  def from_entity(cls, entity, foot_body_ids: torch.Tensor | None = None) -> EvalState:
    """Read the state out of an :class:`~mjlab.entity.Entity`.

    Both harnesses call this, so the two engines are measured from identical
    quantities by construction.

    Args:
      entity: The robot, already stepped and forwarded.
      foot_body_ids: Shape ``(2,)`` body indices of the left and right feet.
        Omit to skip the cadence metric.
    """
    data = entity.data
    foot_pos_w = None
    if foot_body_ids is not None:
      foot_pos_w = data.body_link_pos_w[:, foot_body_ids]
    return cls(
      position_w=data.root_link_pos_w,
      quaternion_w=data.root_link_quat_w,
      lin_vel_b=data.root_link_lin_vel_b,
      ang_vel_b=data.root_link_ang_vel_b,
      foot_pos_w=foot_pos_w,
    )


def upright_from_quat(quaternion_w: torch.Tensor) -> torch.Tensor:
  """Body up-axis dotted with world up, from a ``(w, x, y, z)`` quaternion.

  This is the ``[2, 2]`` element of the rotation matrix: 1.0 standing, 0.0 on
  its side, -1.0 upside down.
  """
  x = quaternion_w[:, 1]
  y = quaternion_w[:, 2]
  return 1.0 - 2.0 * (x * x + y * y)


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
  alive_time: torch.Tensor
  """Seconds of walking measured, i.e. the run length or the time to the fall."""
  achieved_vx: torch.Tensor
  """Mean body-frame forward velocity over the alive period, in m/s."""
  achieved_vy: torch.Tensor
  achieved_wz: torch.Tensor
  """Mean body-frame yaw rate over the alive period, in rad/s."""
  error_vx: torch.Tensor
  """Achieved minus commanded, per axis."""
  error_vy: torch.Tensor
  error_wz: torch.Tensor
  tracking_error: torch.Tensor
  """Norm of the planar velocity error, in m/s."""
  displacement_x: torch.Tensor
  """World-frame travel over the alive period, in metres."""
  displacement_y: torch.Tensor
  path_speed: torch.Tensor
  """Straight-line distance covered per second of alive time, in m/s."""
  rms_roll: torch.Tensor
  """Root-mean-square torso roll over the alive period, in radians."""
  rms_pitch: torch.Tensor
  min_upright: torch.Tensor
  """Smallest up-axis component reached."""
  cadence_hz: torch.Tensor
  """Foot swaps per second; NaN when foot positions were not recorded."""

  def column_names(self) -> list[str]:
    return [f.name for f in fields(self)]

  def rows(self) -> list[list[float]]:
    """Per-environment rows, in :meth:`column_names` order."""
    columns = [getattr(self, name).tolist() for name in self.column_names()]
    return [list(row) for row in zip(*columns, strict=True)]


class WalkMetrics:
  """Accumulates :class:`EvalState` samples into :class:`PerEnvMetrics`.

  Samples are accumulated only while an environment is upright. Everything a
  robot does after it has fallen is noise as far as walking quality goes, and
  averaging it in would make a robot that falls early and slides look slow
  rather than broken -- which is what ``fall_time`` is for.
  """

  def __init__(
    self,
    command_b: torch.Tensor,
    dt: float,
    fall_threshold: float = FALL_UPRIGHT_THRESHOLD,
    foot_dead_band: float = FOOT_HEIGHT_DEAD_BAND,
    warmup_s: float = 0.0,
  ) -> None:
    """
    Args:
      command_b: Shape ``(N, 3)`` commanded ``(vx, vy, wz)`` per environment.
      dt: Seconds between :meth:`record` calls.
      fall_threshold: See :data:`FALL_UPRIGHT_THRESHOLD`.
      foot_dead_band: See :data:`FOOT_HEIGHT_DEAD_BAND`.
      warmup_s: Seconds to discard from the front of the run before the walking
        quality metrics start accumulating. See :attr:`warmup_steps`.
    """
    self.command_b = command_b
    self.dt = dt
    self.fall_threshold = fall_threshold
    self.foot_dead_band = foot_dead_band
    self.warmup_steps = int(round(warmup_s / dt))
    """Control steps excluded from the averages.

    A robot starts from standing and takes a few seconds to reach the speed it
    was asked for, so a mean over the whole run reports the acceleration as
    well as the tracking: the quintic engine averages 0.179 m/s over 5 s of a
    0.3 m/s command, 0.199 over 10 s and 0.212 over 30 s, against a steady
    state of 0.219. Survival is *not* windowed -- ``fall_time``, ``survived``
    and ``alive_time`` are measured from the first step, because a robot that
    falls during the warm-up has not walked.
    """

    num_envs = command_b.shape[0]
    device = command_b.device
    zeros = torch.zeros(num_envs, device=device)

    self._steps = 0
    self._alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    self._alive_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._sample_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self._velocity_sum = torch.zeros(num_envs, 3, device=device)
    self._roll_sq_sum = zeros.clone()
    self._pitch_sq_sum = zeros.clone()
    self._min_upright = torch.full((num_envs,), math.inf, device=device)
    self._start_pos = torch.zeros(num_envs, 3, device=device)
    self._last_pos = torch.zeros(num_envs, 3, device=device)
    self._foot_sign = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._swaps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._has_feet = False
    self._started = False

  def start(self, state: EvalState) -> None:
    """Record the starting pose. Call once, after settling, before stepping."""
    self._start_pos = state.position_w.clone()
    self._last_pos = state.position_w.clone()
    if state.foot_pos_w is not None:
      self._has_feet = True
      self._foot_sign = self._lower_foot(state.foot_pos_w)
    self._started = True

  def record(self, state: EvalState) -> None:
    """Accumulate one control step."""
    if not self._started:
      self.start(state)

    upright = upright_from_quat(state.quaternion_w)
    still_up = upright >= self.fall_threshold
    # The sample on which the robot tips is still counted: it is the last one
    # belonging to the walk, and it is what dates the fall.
    counted = self._alive
    just_fell = counted & ~still_up

    self._steps += 1
    self._fall_step = torch.where(
      just_fell, torch.full_like(self._fall_step, self._steps), self._fall_step
    )

    self._alive_steps = self._alive_steps + counted.long()

    if self._steps == self.warmup_steps and self.warmup_steps > 0:
      # Last sample of the warm-up: rebase the displacement and the foot-swap
      # reference so both describe the measured window and not the run-up.
      self._start_pos = state.position_w.clone()
      self._last_pos = state.position_w.clone()
      if state.foot_pos_w is not None and self._has_feet:
        self._foot_sign = self._lower_foot(state.foot_pos_w)

    sampled = counted & (self._steps > self.warmup_steps)
    weight = sampled.float()
    self._sample_steps = self._sample_steps + sampled.long()
    self._velocity_sum = self._velocity_sum + weight.unsqueeze(-1) * torch.cat(
      (state.lin_vel_b[:, :2], state.ang_vel_b[:, 2:3]), dim=-1
    )

    roll, pitch, _ = euler_xyz_from_quat(state.quaternion_w)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)
    self._roll_sq_sum = self._roll_sq_sum + weight * roll * roll
    self._pitch_sq_sum = self._pitch_sq_sum + weight * pitch * pitch
    self._min_upright = torch.where(
      sampled, torch.minimum(self._min_upright, upright), self._min_upright
    )
    self._last_pos = torch.where(
      sampled.unsqueeze(-1), state.position_w, self._last_pos
    )

    if state.foot_pos_w is not None and self._has_feet:
      sign = self._lower_foot(state.foot_pos_w)
      swapped = (
        sampled & (sign != 0) & (self._foot_sign != 0) & (sign != self._foot_sign)
      )
      self._swaps = self._swaps + swapped.long()
      self._foot_sign = torch.where(sign != 0, sign, self._foot_sign)

    self._alive = counted & still_up

  def _lower_foot(self, foot_pos_w: torch.Tensor) -> torch.Tensor:
    """-1 if the left foot is lower, +1 if the right is, 0 inside the band."""
    difference = foot_pos_w[:, 0, 2] - foot_pos_w[:, 1, 2]
    sign = torch.zeros_like(difference, dtype=torch.long)
    sign = torch.where(difference < -self.foot_dead_band, -torch.ones_like(sign), sign)
    sign = torch.where(difference > self.foot_dead_band, torch.ones_like(sign), sign)
    return sign

  def result(self) -> PerEnvMetrics:
    """Reduce the accumulated samples. Safe to call more than once."""
    sample_steps = self._sample_steps.clamp(min=1).float()
    alive_time = self._alive_steps.float() * self.dt
    # The window the averages are taken over: the alive time less the warm-up,
    # and zero for an environment that never got out of it.
    safe_time = (self._sample_steps.float() * self.dt).clamp(min=self.dt)

    # An environment that fell inside the warm-up contributed no samples, so
    # its averages would be zeros rather than measurements. Say so instead.
    measured = self._sample_steps > 0
    nan = torch.full_like(alive_time, float("nan"))

    def only_measured(values: torch.Tensor) -> torch.Tensor:
      return torch.where(measured, values, nan)

    achieved = torch.where(
      measured.unsqueeze(-1),
      self._velocity_sum / sample_steps.unsqueeze(-1),
      nan.unsqueeze(-1),
    )
    error = achieved - self.command_b
    displacement = self._last_pos - self._start_pos

    fell = self._fall_step >= 0
    fall_time = torch.where(
      fell,
      self._fall_step.float() * self.dt,
      torch.full_like(alive_time, float("nan")),
    )
    cadence = only_measured(self._swaps.float() / safe_time) if self._has_feet else nan
    return PerEnvMetrics(
      command_vx=self.command_b[:, 0],
      command_vy=self.command_b[:, 1],
      command_wz=self.command_b[:, 2],
      survived=(~fell).float(),
      fall_time=fall_time,
      alive_time=alive_time,
      achieved_vx=achieved[:, 0],
      achieved_vy=achieved[:, 1],
      achieved_wz=achieved[:, 2],
      error_vx=error[:, 0],
      error_vy=error[:, 1],
      error_wz=error[:, 2],
      tracking_error=torch.linalg.vector_norm(error[:, :2], dim=-1),
      displacement_x=only_measured(displacement[:, 0]),
      displacement_y=only_measured(displacement[:, 1]),
      path_speed=only_measured(
        torch.linalg.vector_norm(displacement[:, :2], dim=-1) / safe_time
      ),
      rms_roll=only_measured(torch.sqrt(self._roll_sq_sum / sample_steps)),
      rms_pitch=only_measured(torch.sqrt(self._pitch_sq_sum / sample_steps)),
      min_upright=torch.where(self._min_upright.isinf(), nan, self._min_upright),
      cadence_hz=cadence,
    )


class VelocityTrace:
  """Per-control-step commanded and measured base velocity.

  :class:`WalkMetrics` reduces a run to one row per environment, which is the
  right shape for a command sweep and the wrong one for a *profile* run, where
  the command moves during the episode and the interesting quantity is how the
  robot follows it. This records the two side by side, step by step, so a run
  can be drawn as a time series.

  It is a recorder and nothing else: no metric is computed here, and a profile
  run also carries a :class:`WalkMetrics` so falls are dated the same way they
  are everywhere else.
  """

  def __init__(self, dt: float) -> None:
    """
    Args:
      dt: Seconds between :meth:`record` calls.
    """
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
    self._achieved.append(
      torch.cat((state.lin_vel_b[:, :2], state.ang_vel_b[:, 2:3]), dim=-1)
      .detach()
      .to("cpu", torch.float32)
    )
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
  """Write a profile run's traces, one row per step per environment.

  Long rather than wide: a profile run is a few tens of environments over a few
  thousand steps, and one row per sample keeps the file readable by anything
  without the column count depending on the batch size.
  """
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


def _stat(values: torch.Tensor) -> dict[str, float]:
  """Mean, standard deviation and median of a 1-D tensor, NaNs dropped."""
  finite = values[values.isfinite()]
  if finite.numel() == 0:
    return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
  return {
    "mean": float(finite.mean()),
    "std": float(finite.std(unbiased=False)),
    "median": float(finite.median()),
  }


WALK_QUALITY_METRICS: tuple[str, ...] = (
  "achieved_vx",
  "achieved_vy",
  "achieved_wz",
  "error_vx",
  "error_vy",
  "error_wz",
  "tracking_error",
  "path_speed",
  "rms_roll",
  "rms_pitch",
  "min_upright",
  "cadence_hz",
  "alive_time",
)
"""Metrics that describe *how* the robot walked, as opposed to whether it fell."""


def summarise(metrics: PerEnvMetrics) -> dict:
  """Aggregate per-environment metrics into a JSON-friendly summary.

  Walking quality is reported twice: over the environments that survived, and
  over all of them. The survivor figures are the ones to quote -- averaging a
  fallen robot's sliding into a mean speed describes neither population -- but
  when nothing survives they are all NaN, and the all-environment block is what
  is left to look at.
  """
  survived = metrics.survived > 0.5
  summary: dict = {
    "num_envs": int(metrics.survived.numel()),
    "num_survived": int(survived.sum()),
    "survival_rate": float(survived.float().mean()),
    "fall_time": _stat(metrics.fall_time),
    "survivors": {
      name: _stat(getattr(metrics, name)[survived]) for name in WALK_QUALITY_METRICS
    },
    "all_envs": {name: _stat(getattr(metrics, name)) for name in WALK_QUALITY_METRICS},
  }
  return summary


def save_run(output_dir: Path, run: dict, metrics: PerEnvMetrics) -> dict:
  """Write ``per_env.csv`` and ``summary.json``, and return the summary.

  Both entry points call this, so a quintic run and a policy run produce byte
  compatible outputs and can be concatenated without translation.
  """
  summary = {"run": run, **summarise(metrics)}
  write_per_env_csv(output_dir / "per_env.csv", metrics)
  write_summary_json(output_dir / "summary.json", summary)
  return summary


def write_per_env_csv(path: Path, metrics: PerEnvMetrics) -> None:
  """Write one row per environment.

  CSV rather than parquet: a run is one row per environment, so even a large
  sweep is thousands of rows, and CSV costs no dependency and can be read by
  anything.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["env"] + metrics.column_names())
    for index, row in enumerate(metrics.rows()):
      writer.writerow([index] + row)


def write_summary_json(path: Path, summary: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=False)
    handle.write("\n")


def format_summary(summary: dict) -> str:
  """One-screen rendering of :func:`summarise`."""
  survivors = summary["num_survived"]
  block = "survivors" if survivors else "all_envs"
  lines = [
    f"environments      : {summary['num_envs']}",
    f"survived          : {survivors} ({100.0 * summary['survival_rate']:.1f}%)",
  ]
  if survivors < summary["num_envs"]:
    fall = summary["fall_time"]
    lines.append(
      f"fall time (s)     : {fall['mean']:.2f} mean, {fall['median']:.2f} median"
    )
  if not survivors:
    lines.append("(no survivors; figures below are over all environments)")
  for label, key in (
    ("achieved vx (m/s)", "achieved_vx"),
    ("achieved vy (m/s)", "achieved_vy"),
    ("achieved wz (r/s)", "achieved_wz"),
    ("tracking error   ", "tracking_error"),
    ("path speed (m/s) ", "path_speed"),
    ("rms roll (rad)   ", "rms_roll"),
    ("rms pitch (rad)  ", "rms_pitch"),
    ("min upright      ", "min_upright"),
    ("cadence (Hz)     ", "cadence_hz"),
  ):
    stat = summary[block][key]
    lines.append(f"{label} : {stat['mean']:+.3f} +/- {stat['std']:.3f}")
  return "\n".join(lines)
