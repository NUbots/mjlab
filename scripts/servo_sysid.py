"""Servo system identification from a NUClear sensor log.

Reproduces the full doc-17 pipeline against any JSON-lines log containing
paired ``message.platform.RawSensors`` / ``message.input.Sensors`` records
(walking data preferred - the fits need motion, and the noise calibration
is only meaningful under gait vibration):

  uv run python scripts/servo_sysid.py extract <log.json> [--cache out.npz]
  uv run python scripts/servo_sysid.py report <cache.npz>

``report`` prints five sections; where each number lands in the sim is
noted inline and summarized at the end:

1. data-integrity checks: presentVelocity unit (rev/s vs rad/s), the
   per-servo direction-map sign bug, and register latency, all measured
   against the position derivative (never trust the velocity register).
2. electrical fits ``duty*V = R*I + K*w`` per joint (K is the back-EMF
   constant; a stall-spec ratio is NOT comparable - it bakes in gear
   losses).
3. mechanical fits ``K*I = J*w' + b*w + tc*sign(w)`` on swing-phase
   samples. J (reflected inertia -> ARMATURE_*) pools robustly; b/tc are
   gravity-confounded in walking data - use a BAM-style pendulum bench
   for those (github.com/Rhoban/bam).
4. observation-noise calibration: high-frequency residual std per
   channel (gyro/gravity/joint pos/joint vel/current) -> the actor
   Gnoise values. Bench-static noise under-reads walking reality ~5x;
   always calibrate from gait data.
5. power-network fit: ``V = Voc - rate*t - R_src*I_total`` plus
   per-servo residual resistance (daisy-chain position) -> the bus
   voltage model constants and DR ranges.

Quality gates (tighten, never loosen, when pooling): electrical R^2 >
0.9, mechanical R^2 > 0.4, and a joint must have >800 moving swing
samples to contribute.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

FIELDS = (
  "presentPosition",
  "presentVelocity",
  "presentCurrent",
  "presentPWM",
  "voltage",
  "temperature",
)
POS, VEL, CUR, PWM, VOLT, TEMP = range(6)
PWM_FULL_SCALE = 885.0  # Dynamixel PWM register ticks at 100% duty.
LEG_JOINTS = ("HipYaw", "HipRoll", "HipPitch", "Knee", "AnklePitch", "AnkleRoll")


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
  k = np.hanning(w)
  k /= k.sum()
  if x.ndim == 1:
    return np.convolve(x, k, mode="same")
  return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, x)


def kurtosis(x: np.ndarray) -> float:
  x = x - x.mean()
  s2 = float((x**2).mean())
  return float((x**4).mean() / max(s2**2, 1e-30) - 3.0)


def extract(log_path: str, cache_path: str) -> None:
  """Stream the JSON-lines log into numpy arrays (one pass, ~1 min/GB)."""
  servo_names: list[str] | None = None
  raw_t, raw_rows, feet_rows, grav_rows = [], [], [], []
  gyro_rows, accel_rows = [], []
  with open(log_path) as f:
    for line in f:
      is_raw = '"message.platform.RawSensors"' in line[:60]
      rec = json.loads(line)
      d = rec["data"]
      if is_raw:
        sv = d["servo"]
        if servo_names is None:
          servo_names = list(sv.keys())
        raw_t.append(rec["timestamp"])
        raw_rows.append([[s.get(k, 0.0) for k in FIELDS] for s in sv.values()])
        g, a = d["gyroscope"], d["accelerometer"]
        gyro_rows.append((g["x"], g["y"], g["z"]))
        accel_rows.append((a["x"], a["y"], a["z"]))
      else:
        feet_rows.append([bool(ft.get("down", False)) for ft in d.get("feet", [])])
        H = d.get("Htw")
        grav_rows.append(
          (H["x"]["z"], H["y"]["z"], H["z"]["z"]) if H else (0.0, 0.0, 1.0)
        )
  np.savez_compressed(
    cache_path,
    t=np.array(raw_t, dtype=np.float64) / 1e6,
    arr=np.array(raw_rows, dtype=np.float32),
    names=np.array(servo_names),
    fd=np.array(feet_rows, dtype=bool),
    grav=np.array(grav_rows, dtype=np.float32),
    gyro=np.array(gyro_rows, dtype=np.float32),
    accel=np.array(accel_rows, dtype=np.float32),
  )
  print(f"cached {len(raw_t)} records -> {cache_path}")


def _frame_flip(vel_raw: np.ndarray, dpos: np.ndarray) -> float:
  """Sign relating the raw servo frame to the position convention.

  The NUbots position converter applies the per-servo direction map;
  the velocity/current/PWM path does not, so raw-signed channels are
  mirrored on half the joints. Fits run in the RAW frame (flip the
  position-derived kinematics), keeping I and PWM consistent.
  """
  m = np.abs(dpos) > 0.5
  if m.sum() < 200:
    return 1.0
  return float(np.sign(np.corrcoef(smooth(vel_raw)[m], dpos[m])[0, 1]))


def report(cache_path: str) -> None:
  z = np.load(cache_path, allow_pickle=True)
  t, arr, names, fd = z["t"], z["arr"], list(z["names"]), z["fd"]

  print("=== 1. data integrity (vs position derivative) ===")
  print(f"{'joint':<16}{'corr':>7}{'amp_ratio':>11}{'lag_ms':>8}   verdict")
  for j, nm in enumerate(names):
    pos = arr[:, j, POS].astype(np.float64)
    vel = arr[:, j, VEL].astype(np.float64)
    dpos = smooth(np.gradient(pos, t))
    m = np.abs(dpos) > 0.5
    if m.sum() < 500:
      continue
    best = (0, 0.0)
    vs = smooth(vel)
    for lag in range(-10, 11):
      a = vs[max(0, lag) : len(vs) + min(0, lag)]
      b = dpos[max(0, -lag) : len(dpos) - max(0, lag)]
      mm = np.abs(b) > 0.5
      c = np.corrcoef(a[mm], b[mm])[0, 1]
      if abs(c) > abs(best[1]):
        best = (lag, c)
    lag, c = best
    a = vs[max(0, lag) : len(vs) + min(0, lag)]
    b = dpos[max(0, -lag) : len(dpos) - max(0, lag)]
    mm = np.abs(b) > 0.5
    ratio = float(np.std(a[mm]) / np.std(b[mm]))
    dt_ms = float(np.median(np.diff(t)) * 1000)
    unit = "rev/s" if abs(ratio - 1 / (2 * np.pi)) < 0.05 else "rad/s?"
    sign = "SIGN-FLIPPED" if c < 0 else "ok"
    print(f"{nm:<16}{c:>+7.3f}{ratio:>11.5f}{lag * dt_ms:>+8.0f}   {unit}, {sign}")

  print("\n=== 2+3. electrical / mechanical fits (swing phase, raw frame) ===")
  legs = [s + j for j in LEG_JOINTS for s in ("r", "l")]
  print(
    f"{'joint':<14}{'R[ohm]':>8}{'K[Nm/A]':>9}{'eR2':>7}"
    f"{'J':>9}{'b':>8}{'tc':>8}{'mR2':>7}{'n':>7}"
  )
  pooled = []
  for nm in legs:
    if nm not in names:
      continue
    j = names.index(nm)
    pos = arr[:, j, POS].astype(np.float64)
    cur = smooth(arr[:, j, CUR].astype(np.float64), 5)
    duty = arr[:, j, PWM].astype(np.float64) / PWM_FULL_SCALE
    volt = arr[:, j, VOLT].astype(np.float64)
    w = smooth(np.gradient(pos, t))
    w = w * _frame_flip(arr[:, j, VEL].astype(np.float64), w)
    wd = smooth(np.gradient(w, t), 13)
    side = 0 if nm[0] == "l" else 1
    moving = (np.abs(w) > 0.3) & ~fd[:, side]
    if moving.sum() < 800:
      continue
    A = np.stack([cur[moving], w[moving]], 1)
    y = (duty * volt)[moving]
    (R_, K_), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2e = float(1 - res[0] / np.sum((y - y.mean()) ** 2))
    Am = np.stack([wd[moving], w[moving], np.sign(w[moving])], 1)
    ym = (K_ * cur)[moving]
    (J_, b_, tc_), resm, *_ = np.linalg.lstsq(Am, ym, rcond=None)
    r2m = float(1 - resm[0] / np.sum((ym - ym.mean()) ** 2))
    print(
      f"{nm:<14}{R_:>8.3f}{K_:>9.3f}{r2e:>7.3f}"
      f"{J_:>9.4f}{b_:>8.3f}{tc_:>8.3f}{r2m:>7.3f}{moving.sum():>7}"
    )
    if r2e > 0.9 and r2m > 0.4:
      pooled.append((R_, K_, J_, b_, tc_, float(moving.sum())))
  if pooled:
    v = np.array(pooled)
    n = v[:, 5]
    wm = (v[:, :5] * n[:, None]).sum(0) / n.sum()
    sd = np.sqrt(((v[:, :5] - wm) ** 2 * n[:, None]).sum(0) / n.sum())
    print(f"\npooled legs ({len(pooled)} joints passing gates):")
    for name, m, s in zip(("R", "K", "J", "b", "tc"), wm, sd, strict=True):
      print(f"  {name:<3}= {m:.4f} +/- {s:.4f}")
    print("  -> J = ARMATURE_<servo>; K = _NUGUS_CURRENT_KT")
    print("  -> b/tc: gravity-confounded here; prefer a BAM pendulum bench")

  print("\n=== 4. observation-noise calibration (hf residual std) ===")
  gyro, grav = z["gyro"].astype(np.float64), z["grav"].astype(np.float64)
  ghf = gyro - smooth(gyro, 11)
  vhf = grav - smooth(grav, 11)
  print(
    f"gyro std {ghf.std(axis=0).round(4)} kurt {[round(kurtosis(c), 1) for c in ghf.T]}"
  )
  print(f"gravity std {vhf.std(axis=0).round(5)}")
  print("  -> base_ang_vel / projected_gravity actor Gnoise std")
  for label, idx in (("joint_pos", POS), ("current", CUR)):
    hfs = []
    for j in range(arr.shape[1]):
      x = arr[:, j, idx].astype(np.float64)
      hfs.append(float(np.std(x - smooth(x, 11))))
    print(f"{label}: per-servo hf std p50={np.median(hfs):.4f} max={max(hfs):.4f}")
  dvs = []
  for j in range(arr.shape[1]):
    dv = np.gradient(arr[:, j, POS].astype(np.float64), t)
    dvs.append(float(np.std(dv - smooth(dv, 11))))
  print(f"joint_vel (pos-derivative): p50={np.median(dvs):.3f} max={max(dvs):.3f}")
  print("  -> joint_pos / joint_vel / actuator_current actor Gnoise std")

  print("\n=== 5. power network ===")
  V = arr[:, :, VOLT].astype(np.float64)
  current = arr[:, :, CUR].astype(np.float64)
  Itot = np.abs(current).sum(axis=1)
  Vmean = V.mean(axis=1)
  T = t - t[0]
  A = np.stack([np.ones_like(Itot), T, Itot], 1)
  coef, _, *_ = np.linalg.lstsq(A, Vmean, rcond=None)
  bus = A @ coef
  r2 = 1 - np.sum((Vmean - bus) ** 2) / np.sum((Vmean - Vmean.mean()) ** 2)
  print(
    f"V = {coef[0]:.2f} {coef[1] * 60:+.4f} V/min {coef[2] * 1000:+.1f} mOhm*I_tot"
    f"  (R^2={r2:.2f})"
  )
  print(f"fleet I: p50={np.median(Itot):.1f} p95={np.percentile(Itot, 95):.1f}A")
  P = (V * np.abs(current)).sum(axis=1)
  print(
    f"fleet P: p50={np.median(P):.0f} p95={np.percentile(P, 95):.0f}W max={P.max():.0f}W"
  )
  print("per-servo residual resistance (chain position; -> R_local DR range):")
  for j, nm in enumerate(names):
    resid = V[:, j] - bus
    own = np.abs(current[:, j])
    if own.std() < 0.05:
      continue
    c2, *_ = np.linalg.lstsq(np.stack([np.ones_like(own), own], 1), resid, rcond=None)
    print(
      f"  {nm:<16} R_local={-c2[1] * 1000:>7.1f} mOhm  offset={resid.mean():>+.3f}V"
    )
  print("  -> Voc/R_src/discharge + R_local ranges for the bus-voltage model")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest="cmd", required=True)
  pe = sub.add_parser("extract")
  pe.add_argument("log")
  pe.add_argument("--cache", default="sysid_cache.npz")
  pr = sub.add_parser("report")
  pr.add_argument("cache")
  args = p.parse_args()
  if args.cmd == "extract":
    extract(args.log, args.cache)
  else:
    report(args.cache)


if __name__ == "__main__":
  main()
