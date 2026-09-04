"""Velocity-profile figure from a motion-capture log of the real robot.

``eval_velocity_profile.py`` drives a simulated robot through a moving command
and writes ``trace.csv``; ``plot_comparison.py`` draws it. This script draws the
same figure from the *robot*, out of an NBS log exported to JSON, so a run on
the hardware can be laid next to the simulated one and read the same way.

  uv run python scripts/eval/plot_mocap_profile.py \\
    --log logs/eval/quintic-profilewalk-mocap.json

Input format
------------

JSON Lines -- one JSON object per line, as ``nbs2json`` writes -- with a
``type`` and a ``data``. Three message types are read and the rest ignored:

``message.behaviour.state.WalkState``
  ``data.velocityTarget``, the command in force. The same quantity the
  simulated profile writes as ``command_vx``/``vy``/``wz``.
``message.input.MotionCapture``
  ``data.rigidBodies[0]``, the tracked torso: position and orientation in the
  capture volume's frame, plus ``trackingValid``. ``data.natnetTimestamp`` is
  the camera clock and gives the time base -- the log's own timestamps arrive
  in bursts and cannot be differentiated.
``message.input.Sensors``
  ``data.accelerometer`` and ``data.gyroscope``, used only to fix the
  handedness of the capture frame.

Frame calibration
-----------------

A capture volume's frame has nothing to do with the robot's, so four things
have to be pinned down before any velocity can be called "forward". Three come
out of physics and one out of the command:

Floor plane
  Robust PCA fit to the tracked position; the smallest principal direction is
  the normal. 6 mm residual against 1.3 m of travel on the reference log.
Up
  The populated one of the two deep tails -- carrying the robot holds the torso
  well above the walking plane and the floor stops it going equally far below.
  The weakest of the four, and deliberately the least load-bearing: flipping it
  flips the handedness test too, the two cancel, and only which samples count
  as *off its feet* depends on the answer.
Handedness
  A mirrored frame leaves forward alone but reverses left, right and the sense
  of every rotation. Caught by correlating the yaw rate about the floor normal
  against the IMU's yaw rate about the torso's gravity axis. The command plays
  no part, so the test cannot be talked into agreeing with one.
Forward
  The direction travelled while a forward-only command was in force -- the one
  thing taken from the command, so the measured velocity's *heading* is
  calibrated rather than measured. Needs to be right only to within a quadrant.
  Everything else is measured: speeds, yaw rates, time response, coupling.

All four are printed at the top of a run, and each can be pinned with ``--up``,
``--forward`` or ``--chirality``.
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
  INK,
  INK_2,
  MUTED,
  RED,
  SMOOTH_S,
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

Capture measures position, so every velocity here is a local straight-line fit,
and this is the shortest window worth drawing: at 120 Hz and a millimetre of
marker noise a two-frame difference carries ~0.08 m/s of noise, enough to bury
the stride it is meant to show.
"""

GAIT_BAND_HZ = (0.6, 6.0)
"""Frequencies a bipedal gait's stride can be looked for in."""

STRIDE_WINDOWS = 1.4303
"""Strides the velocity is fitted over, once the stride has been measured.

Not one stride: a moving *mean* over one period cancels a sinusoid, but a
moving least-squares *slope* does not, since it weights samples by distance
from the window's centre. Its response vanishes at ``tan(x) = x``,
``x = pi * W / T``, whose first non-zero root is 1.43 periods.

Residual lateral sway on the reference capture: 0.29 m/s at half a stride,
0.18 at one, 0.11 here.
"""

LIFTED_HEIGHT_M = 0.08
"""Height above the walking plane past which the torso is not being walked."""

HANDLED_PAD_S = 0.5
"""Seconds either side of a handled span that are also discarded.

The samples on the way into somebody's hands and back out of them are not a
gait either.
"""

LEAD_S = 3.0
"""Seconds of the standing-around either end of a capture that are drawn.

Recording starts and stops by hand, so a capture opens and closes with a
stationary robot; this keeps just enough to show the trace leaving rest.
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

  The walk engine publishes its state before it has a command, so the unset
  field is whatever was on the stack: denormals near 1e-310, or a float far too
  large to be a velocity.
  """
  return all(math.isfinite(v) and (v == 0.0 or 1e-9 < abs(v) < limit) for v in vector)


def read_log(path: Path) -> Log:
  """Parse an ``nbs2json`` export.

  One pass, keeping only the fields the figure needs: a three-minute capture is
  a few hundred megabytes, almost all of it marker clouds nothing here reads.
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
  # The log stamps a capture message when its batch was unpacked, so its
  # spacing is nothing like the camera's. The offset between the two clocks is
  # constant, and is all that is needed to line command up with motion.
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
# Time-series helpers
# --------------------------------------------------------------------------


def local_slope(t: np.ndarray, x: np.ndarray, window: float) -> np.ndarray:
  """Slope of a straight line fitted to ``x`` over a sliding time window.

  Suits a dropped-frame capture track: it assumes no sample rate, degrades
  across a gap instead of inventing motion through it, and smooths and
  differentiates in one pass. Prefix sums keep it linear in the sample count
  whatever the window width.

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
  # Normal equations for a two-parameter fit, written out to vectorise.
  denominator = n * sum_tt - sum_t * sum_t
  numerator = n[:, None] * sum_tx - sum_t[:, None] * sum_x
  slope = np.full_like(flat, np.nan)
  ok = (n >= 4) & (denominator > 1e-12)
  slope[ok] = numerator[ok] / denominator[ok, None]
  return slope.reshape(x.shape)


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


def _grow(t: np.ndarray, flag: np.ndarray, pad: float) -> np.ndarray:
  """Widen every true span by ``pad`` seconds on each side."""
  grown = np.zeros_like(flag)
  for a, b in _spans(t, flag):
    grown |= (t >= a - pad) & (t <= b + pad)
  return grown


def _interpolate(t: np.ndarray, source_t: np.ndarray, values: np.ndarray):
  return np.stack(
    [np.interp(t, source_t, values[:, k]) for k in range(values.shape[1])], -1
  )


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

  Trimmed rather than plain least squares: the samples where somebody carries
  the robot sit metres off the plane and would drag the fit with them.
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
    # Of the two deep tails, the populated one is up: the floor is in the way
    # of the other.
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
      # Only the off-its-feet test reads this sign; say so rather than letting
      # a coin flip look like a measurement.
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
  yaw_rate = yaw_rate_about(log, normal, SMOOTH_S)

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
        "cannot find the robot's forward axis: the log never commands a "
        "forward velocity and nothing else. Pass --forward."
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


def yaw_rate_about(log: Log, normal: np.ndarray, window: float) -> np.ndarray:
  """Rate of turn about ``normal``, right-handed, rad/s, fitted over ``window``.

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
  return local_slope(log.mocap_t, angle, window)


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
  # Both sides averaged over the same window first: the gyroscope still sees a
  # stride the smoothed capture track has lost, and would understate the match.
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


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass
class Run:
  """A capture, reduced to the things the figure draws.

  Attributes:
    t: Shape ``(M,)`` capture times, the clock everything below is on.
    command: Shape ``(M, 3)`` command, interpolated onto ``t``.
    raw: Shape ``(M, 3)`` body velocity over ``RAW_WINDOW_S``.
    smooth: Shape ``(M, 3)`` body velocity over ``smooth_s``.
    upright: Shape ``(M,)`` cosine of the torso's lean from vertical.
    walked: Shape ``(M,)`` samples that are a gait and have a usable fit.
    handled: Shape ``(M,)`` samples off the floor, on its side, or unsolved.
    driven: Stretches of ``t`` during which a non-zero command was in force.
    fall_t: When the torso first went past the fall threshold, if it did.
    smooth_s: Width of the fit behind ``smooth``, in seconds.
    gait_s: Measured stride period, or ``None`` if no clear peak was found.
  """

  log: Log
  frame: Frame
  t: np.ndarray
  command: np.ndarray
  raw: np.ndarray
  smooth: np.ndarray
  upright: np.ndarray
  walked: np.ndarray
  handled: np.ndarray
  driven: list[tuple[float, float]]
  fall_t: float | None
  smooth_s: float
  gait_s: float | None


def body_velocity(log: Log, frame: Frame, window: float) -> np.ndarray:
  """Forward, left and yaw rate in the robot's frame, over one window."""
  velocity = local_slope(log.mocap_t, log.position, window)
  velocity -= np.outer(velocity @ frame.up_w, frame.up_w)
  body = np.einsum("tji,tj->ti", log.rotation, velocity)
  yaw = frame.chirality * yaw_rate_about(log, frame.up_w, window)
  return np.stack([body @ frame.forward_b, body @ frame.left_b, yaw], -1)


def gait_period(
  t: np.ndarray, lateral: np.ndarray, spans: list[tuple[float, float]]
) -> float | None:
  """The robot's stride period, measured off its own sideways sway.

  The sway is larger than any velocity a profile commands and has to be
  averaged out before a trace shows tracking rather than gait. Measured rather
  than assumed, since assuming it would assume the robot walks at the cadence
  the engine was tuned for -- one of the things a capture is for checking.
  Welch-averaged over the commanded stretches, so a stretch spent falling over
  cannot set the answer.

  Returns:
    Seconds per stride, or ``None`` if no clear peak was found.
  """
  rate, size = 120.0, 2048
  power = np.zeros(size // 2 + 1)
  used = 0
  for start, end in spans:
    window = (t >= start) & (t <= end) & np.isfinite(lateral)
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


def build_run(log: Log, frame: Frame, smooth: float | None = None) -> Run:
  """Everything the figure reads, in one pass over the capture.

  Args:
    log: The capture.
    frame: Its calibration.
    smooth: Window the drawn velocity is fitted over, in seconds. Defaults to
      one measured stride; see :func:`gait_period`.
  """
  raw = body_velocity(log, frame, RAW_WINDOW_S)
  command = _interpolate(log.mocap_t, log.command_t, log.command)
  upright = np.cos(np.radians(frame.tilt_deg))

  # Off the floor, on its side, or unsolved: not a gait, whatever the command
  # said. Padded either side; see HANDLED_PAD_S.
  suspect = (
    ~log.tracked | (frame.height > LIFTED_HEIGHT_M) | (upright < FALL_UPRIGHT_THRESHOLD)
  )
  handled = _grow(log.mocap_t, suspect, HANDLED_PAD_S)

  fall_t = None
  fallen = np.flatnonzero(upright < FALL_UPRIGHT_THRESHOLD)
  if fallen.size:
    fall_t = float(log.mocap_t[fallen[0]])

  # The stride sets the window everything drawn is fitted over, so measure it
  # first, over the stretches that were commanded to move.
  driven = _spans(log.mocap_t, np.any(np.abs(command) > 1e-6, axis=1))
  gait = gait_period(log.mocap_t, raw[:, 1], driven)
  if smooth is not None:
    window_s = smooth
  elif gait is not None:
    window_s = gait * STRIDE_WINDOWS
  else:
    window_s = SMOOTH_S
  smoothed = body_velocity(log, frame, window_s)

  return Run(
    log=log,
    frame=frame,
    t=log.mocap_t,
    command=command,
    raw=raw,
    smooth=smoothed,
    upright=upright,
    walked=~handled & np.isfinite(smoothed[:, 0]),
    handled=handled,
    driven=driven,
    fall_t=fall_t,
    smooth_s=window_s,
    gait_s=gait,
  )


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------


def figure_profile(run: Run, label: str, path: Path) -> None:
  """Command against response, for the whole capture, on one time axis.

  Two differences from the simulated profile figure, both forced by the data.
  It is one robot walking continuously rather than six independent lanes laid
  end to end, so a fall is drawn where it happened. And it is three panels
  rather than one: every axis is drawn throughout, because the response to a
  command the robot was not given is rarely zero, but the torso's yaw swings
  through a stride by more than any commanded speed, so one shared scale would
  flatten the two linear axes.
  """
  if not run.driven:
    raise SystemExit("no commanded motion in this capture; nothing to draw")

  names = ("vx", "vy", "wz")
  fig, axes = plt.subplots(3, 1, figsize=(17, 8.4), sharex=True)

  first = run.driven[0][0] - LEAD_S
  last = run.driven[-1][1] + LEAD_S
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
    # Whichever is larger, the command or the response: a real robot
    # overshoots, and scaling to the command alone would clip it off.
    span = max(
      0.15,
      1.6 * float(np.abs(commanded).max()),
      1.05 * float(np.nanpercentile(np.abs(trace[:, index]), 99.5)),
    )

    for a, b in handled:
      ax.axvspan(a, b, color=MUTED, alpha=0.22, zorder=1)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.plot(
      t,
      np.clip(raw[:, index], -span, span),
      color=colour,
      linewidth=0.4,
      alpha=0.30,
      zorder=2,
    )
    ax.plot(t, commanded, color=INK_2, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    ax.plot(
      t,
      np.clip(trace[:, index], -span, span),
      color=colour,
      linewidth=1.8,
      zorder=4,
      solid_capstyle="round",
    )
    if run.fall_t is not None and first <= run.fall_t <= last:
      ax.axvline(run.fall_t, color=RED, linewidth=1.2, linestyle=":", zorder=5)
    ax.set_ylim(-span, span)
    ax.set_xlim(first, last)
    ax.margins(x=0)
    ax.set_ylabel(f"{AXIS_LABEL[name]} ({AXIS_UNIT[name]})", color=colour)

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
      [], [], color=INK_2, linewidth=2.0, label=f"measured ({run.smooth_s:.2f} s fit)"
    ),
    Line2D(
      [],
      [],
      color=INK_2,
      linewidth=0.8,
      alpha=0.45,
      label=f"measured ({RAW_WINDOW_S:.2f} s fit)",
    ),
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
    "figure, everything carries across a change of command.\n"
    f"Velocity is a {run.smooth_s:.2f} s straight-line fit to the tracked "
    f"torso -- the window that cancels a measured stride -- drawn over the "
    f"same fit across {RAW_WINDOW_S:.2f} s.",
    fontsize=7.5,
    color=MUTED,
    linespacing=1.5,
  )
  fig.tight_layout(rect=(0, 0.06, 1, 0.945))
  save(fig, path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclass
class Args:
  log: Path
  """NBS log exported to JSON, one message per line."""
  label: str | None = None
  """What the figure calls this run. Defaults to the log's filename."""
  output_dir: Path | None = None
  """Where the figure goes. Defaults to ``<log-directory>/<log-name>_figures``."""

  up: str | None = None
  """Pin the capture frame's up axis, as ``x,y,z``. Detected by default."""
  forward: str | None = None
  """Pin the robot's forward axis in the tracked body's frame, as ``x,y,z``.
  Detected by default from the forward-only stretches of the command."""
  chirality: int | None = None
  """Pin the capture frame's handedness: ``1`` right-handed, ``-1`` mirrored.
  Detected by default from the robot's gyroscope."""
  smooth: float | None = None
  """Window the drawn velocity is fitted over, in seconds.

  Defaults to one stride, measured from the robot's own sideways sway. Set it
  to the simulated figure's 0.6 s to put the two side by side, or longer to
  quieten a trace the gait dominates."""


def summarise(run: Run, label: str) -> dict:
  """The numbers behind the figure, in a form something else can read."""
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
  strides = (
    f" ({run.smooth_s / run.gait_s:.2f} strides)"
    if run.gait_s is not None
    else " (no stride found)"
  )
  print(f"velocity window   : {run.smooth_s:.2f} s{strides}")
  print(f"commanded motion  : {len(run.driven)} stretches")
  if run.fall_t is not None:
    print(f"fell              : at {run.fall_t:.1f} s")
  off = 1.0 - float(run.walked.mean())
  if off > 0.001:
    print(f"not on its feet   : {100 * off:.1f}% of the capture, left out")

  figure_profile(run, label, out / f"fig1_mocap_profile_{name}")

  summary_path = out / f"profile_{name}.json"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  with summary_path.open("w") as handle:
    json.dump(summarise(run, label), handle, indent=2)
    handle.write("\n")
  print(f"wrote             : {summary_path}")


if __name__ == "__main__":
  main()
