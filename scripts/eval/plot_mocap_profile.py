"""Velocity-profile figures from a motion-capture log of the real robot.

``eval_velocity_profile.py`` drives a simulated robot through a moving command
and writes ``trace.csv``; ``plot_comparison.py`` draws it. This script draws the
same figures from the *robot*, out of an NBS log exported to JSON, so a run on
the hardware can be laid next to the simulated one and read the same way.

  uv run python scripts/eval/plot_mocap_profile.py \\
    --log logs/eval/quintic-profilewalk-mocap.json

Only the profile figures. A sweep or a grid needs hundreds of runs at held
commands, which is a simulator's job; a mocap capture is one robot doing one
schedule once, and that is exactly what a profile is.

Input format
------------

JSON Lines -- one JSON object per line, as ``nbs2json`` writes -- with a
``type`` and a ``data``. Three message types are read and the rest ignored:

``message.behaviour.state.WalkState``
  ``data.velocityTarget`` is the command that was in force. This is the same
  quantity the simulated profile writes as ``command_vx``/``vy``/``wz``.
``message.input.MotionCapture``
  ``data.rigidBodies[0]`` is the tracked torso: position and orientation in the
  capture volume's frame, plus ``trackingValid``. This is the measurement.
  ``data.natnetTimestamp`` is the camera clock and is used for the time base --
  the log's own timestamps arrive in bursts and are useless for
  differentiating.
``message.input.Sensors``
  ``data.accelerometer`` and ``data.gyroscope``, used only to pin down the
  handedness of the capture frame. See below.

Frame calibration
-----------------

A capture volume's coordinate frame has nothing to do with the robot's. Motive
defines a rigid body's axes from whatever pose it happened to be in when the
body was created, and the floor is wherever the calibration square was put. So
before any velocity can be called "forward", four things have to be pinned
down. Three come out of physics and one out of the command:

**The floor plane** is fitted to the tracked position by a robust PCA. The
robot walks on a plane, so the smallest principal direction is the plane's
normal; on the log this file was written against, the residual is 6 mm against
1.3 m of in-plane travel.

**Which way is up** follows from the floor being a floor. Somebody picking the
robot up carries the torso far above the height it walks at and holds it there;
nothing holds it the same distance below, because the floor is in the way. Of
the two deep tails, the populated one is up.

This is the weakest of the four, and deliberately the one that matters least.
Flip it and the handedness test below flips with it, the two cancel, and every
velocity, yaw rate and ground track comes out the same; the only thing left
depending on the sign is which samples count as *off its feet*. On a capture
where nothing ever leaves the floor there is nothing to detect and nothing that
turns on the answer, and the run says as much.

**The handedness** is read off the robot's own gyroscope. Capture systems
differ on whether their frame is right- or left-handed, and a mirrored frame
leaves forward alone while quietly reversing left and right and the sense of
every rotation. The yaw rate measured about the floor normal is correlated
against the yaw rate the IMU measures about the torso's gravity axis; if they
disagree in sign, the frame is mirrored. Nothing in this test involves the
command, so it cannot be talked into agreeing with one.

**Which way is forward** is the one thing taken from the command: the direction
the robot travelled during the forward-commanded plateaus. It only has to be
right to within a quadrant -- a walk engine told to go forward does not go
sideways -- but it does mean the *heading* of the measured velocity relative to
the command is calibrated rather than measured. Everything else is measured:
every speed, every yaw rate, the whole time response, and the off-axis coupling
this run is full of.

All four are printed at the top of a run and every one can be pinned by hand:
``--up``, ``--forward``, ``--chirality``. Pin them if you know your capture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
  AXIS_COLOUR,
  AXIS_LABEL,
  AXIS_UNIT,
  BASELINE,
  GRID,
  INK,
  INK_2,
  MUTED,
  RED,
  SMOOTH_S,
  SURFACE,
  despine,
  save,
  use_house_style,
)
from matplotlib.lines import Line2D

import mjlab
from mjlab.evaluation.metrics import FALL_UPRIGHT_THRESHOLD

WALK_STATE = "message.behaviour.state.WalkState"
MOTION_CAPTURE = "message.input.MotionCapture"
SENSORS = "message.input.Sensors"

RAW_WINDOW_S = 0.10
"""Window the "raw" velocity trace is differentiated over, in seconds.

Motion capture measures position, not velocity, so there is no such thing as an
instantaneous reading: every velocity here is a local straight-line fit over
some window, and this is the shortest one worth drawing. At 120 Hz and a
millimetre of marker noise a two-frame difference is about 0.08 m/s of noise,
which would bury the stride it is meant to show.
"""

GAIT_BAND_HZ = (0.6, 6.0)
"""Frequencies a bipedal gait's stride can be looked for in."""

STRIDE_WINDOWS = 1.4303
"""Strides the velocity is fitted over, once the stride has been measured.

One stride is the wrong answer, and the reason is worth writing down. A moving
*mean* over exactly one period of a sinusoid cancels it; a moving least-squares
*slope* does not, because it weights each sample by its distance from the
middle of the window rather than equally. The slope's response to a sinusoid
vanishes where ``tan(x) = x`` with ``x = pi * W / T``, and the first non-zero
root of that puts the window at 1.43 periods instead of one.

It shows. On the walk engine's capture the residual sway in the lateral trace
is 0.29 m/s at half a stride, 0.18 at one, and 0.11 here. It keeps falling
slowly past this point, because a real stride is not a pure sinusoid, but so
does the lag -- and the plateau means the figures report move by under
0.02 m/s across the whole range, so nothing that is quoted turns on the choice.
"""

MIN_PLATEAU_S = 1.0
"""Shortest run of a constant command counted as a plateau, in seconds.

Long enough to skip the steps a ramp is quantised into, short enough to keep
every commanded hold in a profile.
"""

SETTLE_FRACTION = 0.5
"""Fraction of a plateau discarded before its mean is taken.

A plateau is preceded by a ramp and the robot arrives at the commanded speed
some way into it. Averaging from the first sample would report the acceleration
as well as the tracking, which is what ``--warmup`` exists for on the simulated
side.
"""

FALL_ATTRIBUTION_S = 2.0
"""Seconds after a held command within which a fall is still blamed on it.

A robot commanded into a fall goes down some way into the command and often
just after it, while the command is ramping back to zero. Attributing that to
the rest that follows would say the robot fell over standing still.
"""

LIFTED_HEIGHT_M = 0.08
"""Height above the walking plane past which the torso is not being walked."""

HANDLED_PAD_S = 0.5
"""Seconds either side of a handled span that are also discarded.

The samples on the way into somebody's hands and back out of them are not a
gait either.
"""


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass
class Log:
  """The three streams this script needs, on one clock.

  Attributes:
    command_t: Shape ``(W,)`` seconds since the start of the log.
    command: Shape ``(W, 3)`` commanded ``(vx, vy, wz)``.
    mocap_t: Shape ``(M,)`` seconds, on the capture system's own clock mapped
      onto the log's.
    position: Shape ``(M, 3)`` tracked torso position, capture frame.
    rotation: Shape ``(M, 3, 3)`` tracked torso orientation, capture frame.
    tracked: Shape ``(M,)`` whether the solver had a fix.
    imu_t: Shape ``(S,)`` seconds. Empty if the log carries no ``Sensors``.
    gyro: Shape ``(S, 3)`` angular velocity, torso frame, rad/s.
    accel: Shape ``(S, 3)`` specific force, torso frame, m/s^2.
  """

  command_t: np.ndarray
  command: np.ndarray
  mocap_t: np.ndarray
  position: np.ndarray
  rotation: np.ndarray
  tracked: np.ndarray
  imu_t: np.ndarray
  gyro: np.ndarray
  accel: np.ndarray
  clock_jitter: float
  frames_kept: float


def _xyz(value: dict, keys: str = "xyz") -> list[float]:
  return [float(value.get(key, 0.0)) for key in keys]


def _sane(vector: list[float], limit: float = 100.0) -> bool:
  """Reject the uninitialised memory a log opens with.

  The walk engine publishes its state before it has a command to publish, and
  the unset field comes out as whatever was on the stack -- denormals around
  1e-310, or a float too large to be a velocity. Both are recognisable, and
  neither is a command anybody gave.
  """
  return all(math.isfinite(v) and (v == 0.0 or 1e-9 < abs(v) < limit) for v in vector)


def read_log(path: Path) -> Log:
  """Parse an ``nbs2json`` export.

  Reads the file once, keeping only the fields the figures need. A capture of a
  three-minute profile is a few hundred megabytes, almost all of it marker
  clouds this script never looks at, and dropping them on the way past is what
  keeps the whole thing to a few seconds.
  """
  ct, cmd = [], []
  mlog, mnat, pos, quat, tracked = [], [], [], [], []
  it, gyro, accel = [], [], []

  with path.open() as handle:
    for line in handle:
      if '"type"' not in line:
        continue
      message = json.loads(line)
      kind = message.get("type")
      data = message.get("data", {})
      stamp = float(message.get("timestamp", 0.0)) * 1e-6
      if kind == WALK_STATE:
        target = _xyz(data.get("velocityTarget", {}))
        if _sane(target):
          ct.append(stamp)
          cmd.append(target)
      elif kind == MOTION_CAPTURE:
        bodies = data.get("rigidBodies") or []
        if not bodies:
          continue
        body = bodies[0]
        mlog.append(stamp)
        mnat.append(float(data.get("natnetTimestamp", stamp)))
        pos.append(_xyz(body["position"]))
        quat.append(_xyz(body["rotation"], "xyzt"))
        tracked.append(bool(body.get("trackingValid", True)))
      elif kind == SENSORS:
        it.append(stamp)
        gyro.append(_xyz(data.get("gyroscope", {})))
        accel.append(_xyz(data.get("accelerometer", {})))

  if not ct:
    raise SystemExit(f"{path}: no usable {WALK_STATE} messages")
  if len(mnat) < 100:
    raise SystemExit(f"{path}: only {len(mnat)} tracked motion-capture frames")

  origin = min(ct[0], mlog[0])
  native = np.asarray(mnat) - mnat[0]
  # The capture system's clock, shifted onto the log's. The log stamps a
  # motion-capture message when the batch it arrived in was unpacked, so its
  # spacing is nothing like the camera's; the offset between the two clocks is
  # constant and is all that is needed to line the command up with the motion.
  offset = float(np.median(np.asarray(mlog) - origin - native))
  mocap_t = native + offset
  return Log(
    command_t=np.asarray(ct) - origin,
    command=np.asarray(cmd),
    mocap_t=mocap_t,
    position=np.asarray(pos),
    rotation=_matrix_from_quat(np.asarray(quat)),
    tracked=np.asarray(tracked),
    imu_t=np.asarray(it) - origin,
    gyro=np.asarray(gyro).reshape(-1, 3),
    accel=np.asarray(accel).reshape(-1, 3),
    clock_jitter=float(np.std(np.asarray(mlog) - origin - mocap_t)),
    frames_kept=len(mocap_t) / max(1.0, (native[-1] - native[0]) * 120.0),
  )


def _matrix_from_quat(quat: np.ndarray) -> np.ndarray:
  """Rotation matrices from ``(x, y, z, w)`` quaternions, shape ``(N, 3, 3)``."""
  x, y, z, w = (quat / np.linalg.norm(quat, axis=-1, keepdims=True)).T
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ],
    -2,
  )


# --------------------------------------------------------------------------
# Differentiating an irregular track
# --------------------------------------------------------------------------


def local_slope(t: np.ndarray, x: np.ndarray, window: float) -> np.ndarray:
  """Slope of a straight line fitted to ``x`` over a sliding time window.

  The right tool for a dropped-frame motion-capture track: it takes the samples
  that are actually there rather than assuming a rate, it degrades gracefully
  across a gap instead of inventing motion through it, and it smooths and
  differentiates in one pass. Prefix sums make it linear in the number of
  samples however wide the window is.

  Args:
    t: Shape ``(N,)`` sample times, increasing.
    x: Shape ``(N,)`` or ``(N, k)`` samples.
    window: Full width of the fitting window, in seconds.

  Returns:
    The same shape as ``x``; NaN where the window held too few samples to fit.
  """
  flat = x.reshape(len(t), -1)
  lo = np.searchsorted(t, t - window / 2, side="left")
  hi = np.searchsorted(t, t + window / 2, side="right")

  def prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, 0)])

  n = (hi - lo).astype(float)
  st, stt = prefix(t), prefix(t * t)
  sx, stx = prefix(flat), prefix(flat * t[:, None])
  sum_t = st[hi] - st[lo]
  sum_tt = stt[hi] - stt[lo]
  sum_x = sx[hi] - sx[lo]
  sum_tx = stx[hi] - stx[lo]
  # The normal equations for a two-parameter fit, written out so the whole
  # track is one vectorised expression.
  denominator = n * sum_tt - sum_t * sum_t
  numerator = n[:, None] * sum_tx - sum_t[:, None] * sum_x
  slope = np.full_like(flat, np.nan)
  ok = (n >= 4) & (denominator > 1e-12)
  slope[ok] = numerator[ok] / denominator[ok, None]
  return slope.reshape(x.shape)


# --------------------------------------------------------------------------
# Frame calibration
# --------------------------------------------------------------------------


@dataclass
class Frame:
  """How the capture volume's axes relate to the robot's.

  Attributes:
    up_w: Floor normal in the capture frame, pointing up.
    up_b: The same direction in the tracked body's own frame.
    forward_b: Body-frame direction the robot walks in when told to go forward.
    chirality: ``+1`` if the capture frame is right-handed, ``-1`` if mirrored.
    plane_residual: RMS distance of the track from the fitted floor, in metres.
    tilt_deg: Per-sample lean from upright, in degrees.
    height: Per-sample height above the walking plane, in metres.
    notes: How each of the four was decided, for printing.
  """

  up_w: np.ndarray
  up_b: np.ndarray
  forward_b: np.ndarray
  chirality: float
  plane_residual: float
  tilt_deg: np.ndarray
  height: np.ndarray
  notes: list[str] = field(default_factory=list)

  @property
  def left_b(self) -> np.ndarray:
    """Body-frame direction of the robot's left.

    The cross product is right-handed; in a mirrored capture frame the physical
    left is the other way, which is the whole of what ``chirality`` buys.
    """
    return self.chirality * np.cross(self.up_b, self.forward_b)


def fit_floor(position: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, float]:
  """Robust plane fit; returns the unit normal and the RMS residual.

  Three rounds of trimming, because the samples where somebody is carrying the
  robot are metres off the plane and would drag an unweighted fit with them.
  """
  keep = usable.copy()
  normal = np.array([0.0, 0.0, 1.0])
  residual = float("nan")
  for _ in range(8):
    centre = position[keep].mean(0)
    _, singular, basis = np.linalg.svd(position[keep] - centre, full_matrices=False)
    normal = basis[-1]
    residual = float(singular[-1] / math.sqrt(keep.sum()))
    offset = (position - centre) @ normal
    spread = np.std(offset[keep])
    keep = usable & (np.abs(offset - np.median(offset[keep])) < 3 * spread)
  return normal, residual


def calibrate(log: Log, up: str | None, forward: str | None, chirality: int | None):
  """Work out the four unknowns of the capture frame. See the module docstring."""
  usable = log.tracked.copy()
  normal, residual = fit_floor(log.position, usable)
  height = (log.position - log.position[usable].mean(0)) @ normal
  height -= np.median(height[usable])

  notes = []
  if up is not None:
    normal = _parse_vector(up, "--up")
    height = (log.position - log.position[usable].mean(0)) @ normal
    height -= np.median(height[usable])
    notes.append("up: given on the command line")
  else:
    # The floor is a floor. Somebody picking the robot up carries the torso far
    # above the height it walks at and holds it there; nothing can hold it the
    # same distance below, because the floor is in the way. So of the two deep
    # tails, the populated one is up.
    deep = 0.6 * float(np.percentile(np.abs(height[usable]), 99.9))
    above = int((height[usable] > deep).sum())
    below = int((height[usable] < -deep).sum())
    if below > above:
      normal, height, above, below = -normal, -height, below, above
    notes.append(
      f"up: floor normal, sign from {above} samples more than {100 * deep:.0f} cm "
      f"clear of the walking plane against {below} that far under it"
    )
    if above <= below * 2:
      # Only the off-its-feet test reads this sign; see the module docstring.
      # Say so rather than letting a coin flip look like a measurement.
      notes.append(
        "  (weak: nothing left the walking plane by much, so the sign is a "
        "guess -- it only decides which samples count as off its feet)"
      )

  # The floor normal in the tracked body's own frame: constant while the robot
  # is upright, and the reference every tilt is measured from.
  normal_b = np.einsum("tji,j->ti", log.rotation, normal)
  up_b = np.median(normal_b[usable], 0)
  up_b /= np.linalg.norm(up_b)
  tilt = np.degrees(np.arccos(np.clip(normal_b @ up_b, -1.0, 1.0)))

  upright = usable & (tilt < 15.0) & (np.abs(height) < LIFTED_HEIGHT_M)
  yaw_rate = _yaw_rate(log, normal)

  if chirality is not None:
    sign = float(chirality)
    notes.append("handedness: given on the command line")
  else:
    sign, note = _chirality(log, yaw_rate, upright)
    notes.append(note)

  if forward is not None:
    forward_b = _parse_vector(forward, "--forward")
    notes.append("forward: given on the command line")
  else:
    velocity = local_slope(log.mocap_t, log.position, SMOOTH_S)
    velocity -= np.outer(velocity @ normal, normal)
    body = np.einsum("tji,tj->ti", log.rotation, velocity)
    driven = _interpolate(log.mocap_t, log.command_t, log.command)
    ahead = (
      upright
      & (driven[:, 0] > 0.5 * driven[:, 0].max())
      & (np.abs(driven[:, 1]) < 1e-6)
      & np.isfinite(body[:, 0])
    )
    if ahead.sum() < 20:
      raise SystemExit(
        "cannot find the robot's forward axis: the log has no plateau with a "
        "forward command and nothing else. Pass --forward."
      )
    forward_b = np.median(body[ahead], 0)
    notes.append(
      f"forward: direction of travel over {ahead.sum()} samples of forward-only command"
    )
  forward_b = forward_b - up_b * (forward_b @ up_b)
  forward_b /= np.linalg.norm(forward_b)

  frame = Frame(
    up_w=normal,
    up_b=up_b,
    forward_b=forward_b,
    chirality=sign,
    plane_residual=residual,
    tilt_deg=tilt,
    height=height,
    notes=notes,
  )
  return frame, yaw_rate


def _parse_vector(text: str, flag: str) -> np.ndarray:
  parts = [p for p in text.replace(",", " ").split() if p]
  if len(parts) != 3:
    raise SystemExit(f"{flag} wants three numbers, got {text!r}")
  vector = np.array([float(p) for p in parts])
  norm = np.linalg.norm(vector)
  if norm == 0.0:
    raise SystemExit(f"{flag} must not be zero")
  return vector / norm


def _yaw_rate(log: Log, normal: np.ndarray) -> np.ndarray:
  """Rate of turn about the floor normal, right-handed, rad/s.

  Read off a body axis projected into the floor plane rather than off the
  quaternion, so a torso that pitches and rolls through every stride does not
  leak into the number.
  """
  reference = log.rotation[:, :, 0]
  planar = reference - np.outer(reference @ normal, normal)
  first = np.cross(normal, [0.0, 0.0, 1.0])
  if np.linalg.norm(first) < 1e-6:
    first = np.cross(normal, [0.0, 1.0, 0.0])
  first /= np.linalg.norm(first)
  second = np.cross(normal, first)
  angle = np.unwrap(np.arctan2(planar @ second, planar @ first))
  return local_slope(log.mocap_t, angle, SMOOTH_S)


def _chirality(log: Log, yaw_rate: np.ndarray, upright: np.ndarray):
  """Is the capture frame right-handed, or mirrored?

  A mirrored frame leaves forward alone and reverses left, right and the sense
  of every rotation, so it cannot be seen in a position track at all. The
  gyroscope can see it: it measures a genuine right-handed yaw rate about the
  torso's own gravity axis, and the two must agree in sign.
  """
  if len(log.imu_t) < 100:
    return 1.0, "handedness: assumed right-handed (the log carries no IMU)"
  still = np.abs(np.linalg.norm(log.accel, axis=1) - 9.81) < 0.4
  if still.sum() < 50:
    return 1.0, "handedness: assumed right-handed (never still enough to read g)"
  gravity = np.median(log.accel[still], 0)
  gravity /= np.linalg.norm(gravity)
  # Both sides averaged over the same window before they are compared: the
  # gyroscope sees a stride the smoothed capture track has already lost, and an
  # unmatched pair would understate the agreement.
  imu = local_slope(
    log.imu_t, np.cumsum(log.gyro @ gravity * np.gradient(log.imu_t)), SMOOTH_S
  )
  paired = np.interp(log.mocap_t, log.imu_t, np.nan_to_num(imu))
  ok = upright & np.isfinite(yaw_rate)
  correlation = float(np.corrcoef(yaw_rate[ok], paired[ok])[0, 1])
  sign = 1.0 if correlation >= 0.0 else -1.0
  handed = "right-handed" if sign > 0 else "mirrored"
  return sign, (
    f"handedness: {handed}, from a {correlation:+.2f} correlation between the "
    f"captured yaw rate and the robot's gyroscope"
  )


def _interpolate(t: np.ndarray, source_t: np.ndarray, values: np.ndarray):
  return np.stack(
    [np.interp(t, source_t, values[:, k]) for k in range(values.shape[1])], -1
  )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass
class Plateau:
  """One held command, and what the robot did during it."""

  start: float
  end: float
  command: np.ndarray
  achieved: np.ndarray
  spread: np.ndarray
  samples: int
  fell: bool

  @property
  def axes(self) -> tuple[str, ...]:
    names = ("vx", "vy", "wz")
    return tuple(n for n, v in zip(names, self.command, strict=True) if abs(v) > 1e-6)

  @property
  def resting(self) -> bool:
    return not self.axes

  @property
  def label(self) -> str:
    if self.resting:
      return "rest"
    return " ".join(
      f"{AXIS_LABEL[n]}{self.command[i]:+.2f}"
      for i, n in enumerate(("vx", "vy", "wz"))
      if abs(self.command[i]) > 1e-6
    )


@dataclass
class Run:
  """A capture, reduced to the things the figures draw."""

  log: Log
  frame: Frame
  t: np.ndarray
  command: np.ndarray
  raw: np.ndarray
  smooth: np.ndarray
  upright: np.ndarray
  walked: np.ndarray
  handled: np.ndarray
  plateaus: list[Plateau]
  fall_t: float | None
  ground: np.ndarray
  smooth_s: float
  gait_s: float | None

  @property
  def blocks(self) -> list[Plateau]:
    return [p for p in self.plateaus if not p.resting]


def body_velocity(log: Log, frame: Frame, window: float) -> np.ndarray:
  """Forward, left and yaw rate in the robot's frame, over one window."""
  velocity = local_slope(log.mocap_t, log.position, window)
  velocity -= np.outer(velocity @ frame.up_w, frame.up_w)
  body = np.einsum("tji,tj->ti", log.rotation, velocity)
  yaw = frame.chirality * _yaw_rate_window(log, frame.up_w, window)
  return np.stack([body @ frame.forward_b, body @ frame.left_b, yaw], -1)


def _yaw_rate_window(log: Log, normal: np.ndarray, window: float) -> np.ndarray:
  reference = log.rotation[:, :, 0]
  planar = reference - np.outer(reference @ normal, normal)
  first = np.cross(normal, [0.0, 0.0, 1.0])
  if np.linalg.norm(first) < 1e-6:
    first = np.cross(normal, [0.0, 1.0, 0.0])
  first /= np.linalg.norm(first)
  second = np.cross(normal, first)
  angle = np.unwrap(np.arctan2(planar @ second, planar @ first))
  return local_slope(log.mocap_t, angle, window)


def gait_period(t: np.ndarray, lateral: np.ndarray, blocks) -> float | None:
  """The robot's stride period, measured off its own sideways sway.

  A walking torso sways sideways and counter-rotates once per stride, and on
  the hardware that sway is larger than any velocity a profile commands. It has
  to be averaged out before a trace shows tracking rather than gait, and the
  window that does that cleanly is one cycle -- so the cycle is measured rather
  than assumed. Assuming it would be assuming the robot walks at the cadence
  the engine was tuned for, which is one of the things a capture is for
  checking.

  Averaged over the commanded blocks Welch-style rather than taken from one
  spectrum of the whole run, so a block the robot spent falling over does not
  set the answer.

  Returns:
    Seconds per stride, or ``None`` if no clear peak was found.
  """
  rate, size = 120.0, 2048
  power = np.zeros(size // 2 + 1)
  used = 0
  for block in blocks:
    window = (t >= block.start) & (t <= block.end) & np.isfinite(lateral)
    if window.sum() < 30:
      continue
    grid = np.arange(t[window][0], t[window][-1], 1.0 / rate)
    if len(grid) < 64:
      continue
    piece = np.interp(grid, t[window], lateral[window])
    piece = (piece - piece.mean()) * np.hanning(len(piece))
    power += np.abs(np.fft.rfft(piece, size)) ** 2
    used += 1
  if not used:
    return None
  freq = np.fft.rfftfreq(size, 1.0 / rate)
  band = (freq >= GAIT_BAND_HZ[0]) & (freq <= GAIT_BAND_HZ[1])
  peak = float(freq[band][np.argmax(power[band])])
  return 1.0 / peak if peak > 0.0 else None


def find_plateaus(
  t: np.ndarray, command: np.ndarray, minimum: float
) -> list[tuple[float, float, np.ndarray]]:
  """Maximal runs of an unchanging command that lasted at least ``minimum``.

  Detected rather than assumed. A profile's schedule is a property of the run
  that produced the log, not of this script, and the same figure has to draw a
  capture whose amplitudes or timings were changed without being told.
  """
  rounded = np.round(command, 3)
  change = np.any(np.diff(rounded, axis=0) != 0.0, axis=1)
  edges = np.concatenate([[0], np.flatnonzero(change) + 1, [len(t)]])
  runs = []
  for a, b in zip(edges[:-1], edges[1:], strict=True):
    if t[b - 1] - t[a] >= minimum:
      runs.append((float(t[a]), float(t[b - 1]), rounded[a]))
  return runs


def build_run(log: Log, frame: Frame, smooth: float | None = None) -> Run:
  """Everything the figures read, in one pass over the capture.

  Args:
    log: The capture.
    frame: Its calibration.
    smooth: Window the drawn and averaged velocity is fitted over, in seconds.
      Defaults to one measured stride; see :func:`gait_period`.
  """
  raw = body_velocity(log, frame, RAW_WINDOW_S)
  command = _interpolate(log.mocap_t, log.command_t, log.command)
  upright = np.cos(np.radians(frame.tilt_deg))

  # Off the floor, on its side, or unsolved: not a gait, whatever the command
  # said. Grown by a little either side, because the samples on the way into
  # somebody's hands are not one either.
  suspect = (
    ~log.tracked | (frame.height > LIFTED_HEIGHT_M) | (upright < FALL_UPRIGHT_THRESHOLD)
  )
  handled = _grow(log.mocap_t, suspect, HANDLED_PAD_S)
  walked = ~handled

  fall_t = None
  fallen = np.flatnonzero(upright < FALL_UPRIGHT_THRESHOLD)
  if fallen.size:
    fall_t = float(log.mocap_t[fallen[0]])

  # The stride is measured off the commanded blocks, and the window that
  # averages it out is what everything downstream is drawn and averaged over.
  held = [
    Plateau(a, b, v, np.full(3, np.nan), np.full((2, 3), np.nan), 0, False)
    for a, b, v in find_plateaus(log.command_t, log.command, MIN_PLATEAU_S)
  ]
  gait = gait_period(log.mocap_t, raw[:, 1], [p for p in held if not p.resting])
  if smooth is not None:
    window_s = smooth
  else:
    window_s = gait * STRIDE_WINDOWS if gait is not None else SMOOTH_S
  smoothed = body_velocity(log, frame, window_s)
  walked = walked & np.isfinite(smoothed[:, 0])

  plateaus = []
  for start, end, value in find_plateaus(log.command_t, log.command, MIN_PLATEAU_S):
    settled = start + SETTLE_FRACTION * (end - start)
    window = (log.mocap_t >= settled) & (log.mocap_t <= end)
    measured = window & walked
    if measured.sum() >= 4:
      achieved = smoothed[measured].mean(0)
      # The middle eight tenths rather than the extremes: over a held command
      # the trace is a mean plus a stride, and min-to-max would report the
      # stride's worst excursion as though it were the uncertainty on the mean.
      spread = np.percentile(smoothed[measured], [10, 90], axis=0)
    else:
      achieved = np.full(3, np.nan)
      spread = np.full((2, 3), np.nan)
    plateaus.append(
      Plateau(
        start=start,
        end=end,
        command=value,
        achieved=achieved,
        spread=spread,
        samples=int(measured.sum()),
        fell=bool(fall_t is not None and start <= fall_t <= end + FALL_ATTRIBUTION_S),
      )
    )

  # The ground track, in metres along and across the heading the robot held
  # when the first command arrived. Anchored to the first *tracked* sample of
  # the run rather than the first walked one, so the origin does not move when
  # a sample somewhere else is reclassified as off its feet.
  opening = [p for p in plateaus if not p.resting]
  begins = opening[0].start if opening else float(log.mocap_t[0])
  anchor = int(np.flatnonzero(log.tracked & (log.mocap_t >= begins))[0])
  origin = log.position[anchor]
  heading = log.rotation[anchor] @ frame.forward_b
  heading -= frame.up_w * (heading @ frame.up_w)
  heading /= np.linalg.norm(heading)
  across = frame.chirality * np.cross(frame.up_w, heading)
  offset = log.position - origin
  ground = np.stack([offset @ heading, offset @ across], -1)

  return Run(
    log=log,
    frame=frame,
    t=log.mocap_t,
    command=command,
    raw=raw,
    smooth=smoothed,
    upright=upright,
    walked=walked,
    handled=handled,
    plateaus=plateaus,
    fall_t=fall_t,
    ground=ground,
    smooth_s=window_s,
    gait_s=gait,
  )


def _grow(t: np.ndarray, flag: np.ndarray, pad: float) -> np.ndarray:
  """Widen every true span by ``pad`` seconds on each side."""
  if not flag.any():
    return flag
  grown = flag.copy()
  edges = np.flatnonzero(np.diff(flag.astype(int)) != 0) + 1
  bounds = np.concatenate([[0], edges, [len(flag)]])
  for a, b in zip(bounds[:-1], bounds[1:], strict=True):
    if flag[a]:
      grown |= (t >= t[a] - pad) & (t <= t[b - 1] + pad)
  return grown


# --------------------------------------------------------------------------
# Figure 1: velocity tracking under a moving command
# --------------------------------------------------------------------------

LEAD_S = 3.0
"""Seconds of the standing-around either end of a capture that are drawn.

A capture starts when somebody presses record and ends when they stop, so it
opens and closes with a stationary robot. Enough is kept to show the trace
starting from rest; the rest of it is dead time.
"""


def figure_profile(run: Run, label: str, path: Path) -> None:
  """Command against response, for the whole capture, on one time axis.

  The counterpart of the simulated profile figure, with two differences that
  the data forces. The simulated one lays six independent robots' lanes end to
  end; this is one robot walking continuously for three minutes, so a boundary
  is a change of command and everything carries across it -- which is why a
  fall part way through is drawn where it happened rather than inside a lane.

  And it is three panels rather than one. Every axis is drawn through every
  block, because on a real robot the response to a command it was not given is
  rarely zero and is usually the interesting part; but the torso's yaw swings
  through a stride by more than any commanded speed, so sharing one scale
  between the three would flatten the two linear axes into the middle of it.
  """
  names = ("vx", "vy", "wz")
  fig, axes = plt.subplots(3, 1, figsize=(17, 8.4), sharex=True)

  blocks = run.blocks
  if not blocks:
    raise SystemExit("no commanded plateaus in this capture; nothing to draw")
  first, last = blocks[0].start - LEAD_S, blocks[-1].end + LEAD_S
  window = (run.t >= first) & (run.t <= last)
  t = run.t[window]
  trace = np.where(run.walked[window, None], run.smooth[window], np.nan)
  raw = np.where(run.walked[window, None], run.raw[window], np.nan)
  handled = [
    (max(a, first), min(b, last))
    for a, b in _spans(run.t, run.handled)
    if b >= first and a <= last
  ]

  for index, (ax, name) in enumerate(zip(axes, names, strict=True)):
    despine(ax)
    colour = AXIS_COLOUR[name]
    commanded = run.command[window, index]
    # Scaled to whichever is larger, the command or the response it produced.
    # A real robot overshoots, and a frame drawn to the command alone would cut
    # the overshoot off at exactly the moment worth looking at.
    span = max(
      0.15,
      1.6 * float(np.abs(commanded).max()),
      1.05 * float(np.nanpercentile(np.abs(trace[:, index]), 99.5)),
    )

    for order, block in enumerate(blocks):
      if order % 2:
        ax.axvspan(block.start, block.end, color=GRID, alpha=0.30, zorder=0)
    for a, b in handled:
      ax.axvspan(a, b, color=MUTED, alpha=0.22, zorder=1)

    ax.plot(
      t,
      np.clip(raw[:, index], -span, span),
      color=colour,
      linewidth=0.4,
      alpha=0.30,
      zorder=2,
    )
    ax.plot(
      t,
      commanded,
      color=INK_2,
      linewidth=1.4,
      linestyle=(0, (5, 3)),
      zorder=3,
    )
    ax.plot(
      t,
      np.clip(trace[:, index], -span, span),
      color=colour,
      linewidth=1.8,
      zorder=4,
      solid_capstyle="round",
    )
    # The number figure 2 reports, drawn where it was measured. On the lateral
    # axis especially the stride swings the trace by more than the command asks
    # for, and the mean is the only way to see the tracking through it.
    for plateau in run.plateaus:
      if not np.isfinite(plateau.achieved[index]):
        continue
      settled = plateau.start + SETTLE_FRACTION * (plateau.end - plateau.start)
      ax.plot(
        [settled, plateau.end],
        [plateau.achieved[index]] * 2,
        color=INK,
        linewidth=2.4,
        alpha=0.85,
        zorder=5,
        solid_capstyle="butt",
      )
    if run.fall_t is not None and first <= run.fall_t <= last:
      ax.axvline(run.fall_t, color=RED, linewidth=1.2, linestyle=":", zorder=6)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.set_ylim(-span, span)
    ax.set_xlim(first, last)
    ax.margins(x=0)
    ax.set_ylabel(f"{AXIS_LABEL[name]} ({AXIS_UNIT[name]})", color=colour)

  # The command each block asked for, written over the top panel only.
  for block in blocks:
    axes[0].annotate(
      block.label,
      xy=(0.5 * (block.start + block.end), 1.0),
      xycoords=("data", "axes fraction"),
      xytext=(0, 5),
      textcoords="offset points",
      ha="center",
      va="bottom",
      fontsize=8,
      fontweight="semibold",
      color=INK,
    )
  if run.fall_t is not None and first <= run.fall_t <= last:
    axes[0].annotate(
      f"fell at {run.fall_t:.0f} s",
      xy=(run.fall_t, 0.03),
      xycoords=("data", "axes fraction"),
      xytext=(4, 0),
      textcoords="offset points",
      color=RED,
      fontsize=7.5,
      fontweight="bold",
    )
  axes[-1].set_xlabel("time through the capture (s)")

  handles = [
    Line2D(
      [], [], color=INK_2, linewidth=1.6, linestyle=(0, (5, 3)), label="commanded"
    ),
    Line2D(
      [],
      [],
      color=INK_2,
      linewidth=2.0,
      label=f"measured ({run.smooth_s:.2f} s fit)",
    ),
    Line2D(
      [],
      [],
      color=INK_2,
      linewidth=0.8,
      alpha=0.45,
      label=f"measured ({RAW_WINDOW_S:.2f} s fit)",
    ),
    Line2D([], [], color=INK, linewidth=2.6, label="plateau mean"),
  ]
  if handled:
    handles.append(
      Line2D([], [], color=MUTED, linewidth=8, alpha=0.35, label="not on its feet")
    )
  fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    ncol=len(handles),
    columnspacing=1.6,
  )
  fig.suptitle(
    f"Velocity tracking under a moving command — {label}",
    x=0.006,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.text(
    0.006,
    0.010,
    "Motion capture of the robot, one continuous run: unlike the simulated "
    "figure, everything carries across a block boundary.\n"
    f"Velocity is a {run.smooth_s:.2f} s straight-line fit to the tracked "
    f"torso -- the window that cancels a measured stride -- drawn over the "
    f"same fit across {RAW_WINDOW_S:.2f} s.\n"
    "A stride swings the pale trace by more than any command asks for, which "
    "is the gait rather than the tracking, so each held command also carries "
    "its mean.",
    fontsize=7.5,
    color=MUTED,
    linespacing=1.5,
  )
  fig.tight_layout(rect=(0, 0.075, 1, 0.945))
  save(fig, path)


def _wrap(text: str, width: int) -> str:
  """Break a run-on key into lines no wider than the panels above it."""
  lines, current = [], ""
  for piece in text.split("  ·  "):
    candidate = piece if not current else f"{current}  ·  {piece}"
    if len(candidate) > width and current:
      lines.append(current)
      current = piece
    else:
      current = candidate
  if current:
    lines.append(current)
  return "\n".join(lines)


def _spans(t: np.ndarray, flag: np.ndarray) -> list[tuple[float, float]]:
  """Contiguous true spans of ``flag``, as ``(start, end)`` times."""
  if not flag.any():
    return []
  edges = np.flatnonzero(np.diff(flag.astype(int)) != 0) + 1
  bounds = np.concatenate([[0], edges, [len(flag)]])
  return [
    (float(t[a]), float(t[b - 1]))
    for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    if flag[a]
  ]


# --------------------------------------------------------------------------
# Figure 2: what each commanded plateau actually produced
# --------------------------------------------------------------------------


def figure_plateaus(run: Run, label: str, path: Path) -> None:
  """Achieved against commanded, one point per held command.

  Every plateau appears in every panel, not just in the panel for the axis it
  drives. A plateau that commanded only a yaw still has a forward speed, and
  putting it at ``x = 0`` in the forward panel is what makes the coupling
  visible instead of leaving it out of the figure.
  """
  names = ("vx", "vy", "wz")
  fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
  measured = [p for p in run.plateaus if np.isfinite(p.achieved).all()]

  for index, (ax, name) in enumerate(zip(axes, names, strict=True)):
    despine(ax)
    driving = [p for p in measured if name in p.axes]
    other = [p for p in measured if name not in p.axes]
    limit = max(
      0.05,
      float(np.abs([p.command[index] for p in measured] or [0.0]).max()),
      float(np.abs([p.achieved[index] for p in measured] or [0.0]).max()),
    )
    ax.plot(
      [-limit, limit],
      [-limit, limit],
      color=MUTED,
      linewidth=1.0,
      linestyle=(0, (4, 3)),
      zorder=2,
    )
    for group, style in ((other, "other"), (driving, "driving")):
      if not group:
        continue
      x = [p.command[index] for p in group]
      y = [p.achieved[index] for p in group]
      lo = [p.spread[0, index] for p in group]
      hi = [p.spread[1, index] for p in group]
      fell = np.array([p.fell for p in group])
      # Drawn as a segment from one percentile to the other rather than as an
      # error bar around the mean: over a held command the trace is a mean plus
      # a stride, and a skewed stride can leave the mean outside its own middle
      # eighty, which an error bar cannot express.
      ax.vlines(
        x,
        lo,
        hi,
        color=AXIS_COLOUR[name],
        linewidth=1.0,
        alpha=0.35 if style == "other" else 0.6,
        zorder=3,
      )
      common = dict(linestyle="none", marker="o", markersize=6.0, zorder=4)
      if style == "driving":
        ax.plot(
          np.array(x)[~fell],
          np.array(y)[~fell],
          color=AXIS_COLOUR[name],
          markeredgecolor=SURFACE,
          markeredgewidth=1.0,
          **common,
        )
        ax.plot(
          np.array(x)[fell],
          np.array(y)[fell],
          markerfacecolor=SURFACE,
          markeredgecolor=RED,
          markeredgewidth=1.6,
          **common,
        )
      else:
        ax.plot(
          x,
          y,
          markerfacecolor=SURFACE,
          markeredgecolor=AXIS_COLOUR[name],
          markeredgewidth=1.0,
          markersize=4.5,
          linestyle="none",
          marker="o",
          zorder=3,
        )
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(-1.15 * limit, 1.15 * limit)
    ax.set_ylim(-1.15 * limit, 1.15 * limit)
    ax.set_title(f"{AXIS_LABEL[name]}", loc="left", pad=6)
    ax.set_xlabel(f"commanded {AXIS_LABEL[name]} ({AXIS_UNIT[name]})")
    ax.set_ylabel(f"achieved ({AXIS_UNIT[name]})")

  handles = [
    Line2D(
      [],
      [],
      color=INK_2,
      marker="o",
      markersize=6,
      linestyle="none",
      markeredgecolor=SURFACE,
      label="plateau commanding this axis",
    ),
    Line2D(
      [],
      [],
      color=INK_2,
      marker="o",
      markersize=4.5,
      linestyle="none",
      markerfacecolor=SURFACE,
      label="plateau commanding another axis",
    ),
    Line2D(
      [],
      [],
      color=MUTED,
      linewidth=1.4,
      linestyle=(0, (4, 3)),
      label="perfect tracking",
    ),
  ]
  if any(p.fell for p in measured):
    handles.append(
      Line2D(
        [],
        [],
        color=RED,
        marker="o",
        markersize=6,
        linestyle="none",
        markerfacecolor=SURFACE,
        markeredgewidth=1.6,
        label="the robot went down",
      )
    )
  fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=len(handles),
    columnspacing=2.0,
  )
  fig.suptitle(
    f"Steady-state tracking, plateau by plateau — {label}",
    x=0.006,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.text(
    0.006,
    0.012,
    f"Mean over the last {100 * (1 - SETTLE_FRACTION):.0f}% of each held command, "
    "so the ramp onto it is not averaged in.\n"
    f"The bar is the middle 80% of the {run.smooth_s:.2f} s-averaged trace over "
    "that window, which on a real robot is mostly the stride.\n"
    "Samples where it was not on its feet are left out.",
    fontsize=7.5,
    color=MUTED,
    linespacing=1.5,
  )
  fig.tight_layout(rect=(0, 0.10, 1, 0.86))
  save(fig, path)


# --------------------------------------------------------------------------
# Figure 3: where the robot actually went
# --------------------------------------------------------------------------


def figure_ground_track(run: Run, label: str, path: Path) -> None:
  """The path over the floor, which is the thing only a capture can give.

  Drawn in the robot's initial heading: forward is up the page. Each commanded
  block is a solid coloured stretch and the rests between them are grey, so a
  command that was meant to move the robot in a straight line and did not shows
  as a curve rather than as a number.
  """
  fig, ax = plt.subplots(figsize=(7.6, 7.2))
  despine(ax)
  ax.set_aspect("equal")

  drawn = run.walked
  ax.plot(
    run.ground[drawn, 1],
    run.ground[drawn, 0],
    color=BASELINE,
    linewidth=1.0,
    zorder=2,
    solid_capstyle="round",
  )
  for a, b in _spans(run.t, run.handled):
    piece = (run.t >= a) & (run.t <= b)
    ax.plot(
      run.ground[piece, 1],
      run.ground[piece, 0],
      color=MUTED,
      linewidth=1.0,
      linestyle=(0, (2, 2)),
      zorder=2,
    )

  for order, block in enumerate(run.blocks, start=1):
    piece = (run.t >= block.start) & (run.t <= block.end) & drawn
    if piece.sum() < 2:
      continue
    colour = AXIS_COLOUR[block.axes[0]]
    ax.plot(
      run.ground[piece, 1],
      run.ground[piece, 0],
      color=colour,
      linewidth=2.6,
      alpha=0.9,
      zorder=3,
      solid_capstyle="round",
    )
    # Numbered rather than named. Twelve commands over four metres of floor
    # put their labels on top of each other; the key underneath says which is
    # which, and the numbers are the order they were commanded in.
    start = np.flatnonzero(piece)[0]
    ax.annotate(
      str(order),
      xy=(run.ground[start, 1], run.ground[start, 0]),
      xytext=(0, 0),
      textcoords="offset points",
      fontsize=8,
      color=SURFACE,
      fontweight="bold",
      ha="center",
      va="center",
      zorder=5,
      bbox={
        "boxstyle": "circle,pad=0.22",
        "facecolor": colour,
        "edgecolor": SURFACE,
        "linewidth": 1.0,
      },
    )

  ax.plot(
    0.0,
    0.0,
    marker="o",
    markersize=8,
    color=INK,
    markeredgecolor=SURFACE,
    markeredgewidth=1.5,
    zorder=5,
  )
  ax.annotate(
    "start",
    xy=(0.0, 0.0),
    xytext=(10, -12),
    textcoords="offset points",
    fontsize=8,
    color=INK,
    fontweight="bold",
  )
  if run.fall_t is not None:
    at = int(np.argmin(np.abs(run.t - run.fall_t)))
    ax.plot(
      run.ground[at, 1],
      run.ground[at, 0],
      marker="X",
      markersize=11,
      color=RED,
      markeredgecolor=SURFACE,
      markeredgewidth=1.2,
      zorder=6,
    )
    ax.annotate(
      "fell",
      xy=(run.ground[at, 1], run.ground[at, 0]),
      xytext=(10, 6),
      textcoords="offset points",
      fontsize=8,
      color=RED,
      fontweight="bold",
    )

  ax.set_xlabel("left of the starting heading (m)")
  ax.set_ylabel("along the starting heading (m)")
  ax.invert_xaxis()
  fig.suptitle(
    f"Where the robot went — {label}",
    x=0.01,
    y=0.99,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  key = _wrap(
    "  ·  ".join(
      f"{order} {block.label}" for order, block in enumerate(run.blocks, start=1)
    ),
    120,
  )
  fig.text(
    0.01,
    0.010,
    "Torso track over the floor plane, seen from above, rotated so the robot "
    "starts facing up the page.\nColoured by the first axis each block "
    "commands; grey is the rest between two of them, dashed where the robot "
    "was off its feet.\n" + key,
    fontsize=7.5,
    color=MUTED,
    linespacing=1.5,
  )
  fig.tight_layout(rect=(0, 0.055 + 0.02 * key.count("\n"), 1, 0.955))
  save(fig, path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclass
class Args:
  log: Path
  """NBS log exported to JSON, one message per line."""
  label: str | None = None
  """What the figures call this run. Defaults to the log's filename."""
  output_dir: Path | None = None
  """Where the figures go. Defaults to ``<log-directory>/<log-name>_figures``."""

  up: str | None = None
  """Pin the capture frame's up axis, as ``x,y,z``. Detected by default."""
  forward: str | None = None
  """Pin the robot's forward axis in the tracked body's frame, as ``x,y,z``.
  Detected by default from the forward-commanded plateaus."""
  chirality: int | None = None
  """Pin the capture frame's handedness: ``1`` right-handed, ``-1`` mirrored.
  Detected by default from the robot's gyroscope."""
  smooth: float | None = None
  """Window the drawn and averaged velocity is fitted over, in seconds.

  Defaults to one stride, measured from the robot's own sideways sway. Set it
  to the simulated figure's 0.6 s to put the two side by side, or longer to
  quieten a trace the gait dominates."""


def summarise(run: Run, label: str) -> dict:
  """The numbers on the figures, in a form something else can read."""
  frame = run.frame
  return {
    "label": label,
    "capture": {
      "duration_s": round(float(run.t[-1] - run.t[0]), 2),
      "mocap_frames": int(len(run.t)),
      "frames_kept": round(float(run.log.frames_kept), 4),
      "clock_jitter_s": round(float(run.log.clock_jitter), 4),
      "walked_fraction": round(float(run.walked.mean()), 4),
      "stride_s": None if run.gait_s is None else round(run.gait_s, 4),
      "velocity_window_s": round(run.smooth_s, 4),
      "fall_time_s": None if run.fall_t is None else round(run.fall_t, 2),
    },
    "frame": {
      "up_capture": [round(float(v), 6) for v in frame.up_w],
      "up_body": [round(float(v), 6) for v in frame.up_b],
      "forward_body": [round(float(v), 6) for v in frame.forward_b],
      "chirality": int(frame.chirality),
      "plane_residual_m": round(frame.plane_residual, 5),
      "notes": frame.notes,
    },
    "plateaus": [
      {
        "start_s": round(p.start, 2),
        "end_s": round(p.end, 2),
        "axes": list(p.axes),
        "command": [round(float(v), 4) for v in p.command],
        "achieved": [
          None if not np.isfinite(v) else round(float(v), 4) for v in p.achieved
        ],
        "samples": p.samples,
        "fell": p.fell,
      }
      for p in run.plateaus
    ],
  }


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()

  if not args.log.is_file():
    raise SystemExit(f"no such log: {args.log}")
  name = args.log.stem
  label = args.label or name.replace("-", " ").replace("_", " ")
  out = args.output_dir or args.log.parent / f"{name}_figures"

  log = read_log(args.log)
  frame, _ = calibrate(log, args.up, args.forward, args.chirality)
  run = build_run(log, frame, args.smooth)

  print(f"\n{args.log}")
  print(
    f"capture           : {run.t[-1] - run.t[0]:.1f} s, {len(run.t)} tracked "
    f"frames ({100 * log.frames_kept:.0f}% of a 120 Hz camera)"
  )
  print(
    f"clock jitter      : {log.clock_jitter * 1e3:.0f} ms between the capture "
    f"and the log"
  )
  print(f"plane residual    : {frame.plane_residual * 1e3:.1f} mm")
  for note in frame.notes:
    print(f"  {note}")
  if run.gait_s is not None:
    print(f"stride            : {run.gait_s:.3f} s, measured from the sideways sway")
  print(
    f"velocity window   : {run.smooth_s:.2f} s"
    + (
      f" ({run.smooth_s / run.gait_s:.2f} strides)"
      if run.gait_s is not None
      else " (no stride found)"
    )
  )
  print(
    f"commanded holds   : {len(run.blocks)} "
    f"({len(run.plateaus) - len(run.blocks)} rests)"
  )
  if run.fall_t is not None:
    print(f"fell              : at {run.fall_t:.1f} s")
  off = 1.0 - float(run.walked.mean())
  if off > 0.001:
    print(f"not on its feet   : {100 * off:.1f}% of the capture, left out")

  figure_profile(run, label, out / f"fig1_mocap_profile_{name}")
  figure_plateaus(run, label, out / f"fig2_mocap_plateaus_{name}")
  figure_ground_track(run, label, out / f"fig3_mocap_ground_track_{name}")

  summary_path = out / f"profile_{name}.json"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  with summary_path.open("w") as handle:
    json.dump(summarise(run, label), handle, indent=2)
    handle.write("\n")
  print(f"wrote             : {summary_path}")


if __name__ == "__main__":
  main()
