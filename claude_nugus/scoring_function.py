"""Scoring function for evaluating a completed (or partially-completed) run.

Takes the compressed metrics summary and returns a scalar score plus a rich
breakdown of the components that produced it. The score is the orchestrator's
automated proxy for "natural gait quality"; human feedback remains the ground
truth.

Design priorities:

1.  **No hard-zero gates.** Earlier versions of this function collapsed every
    failed run to 0.0, which made the experiment history useless for ranking
    "almost passing" vs "catastrophic" failures. The new function uses smooth
    saturating signals so the score always carries information.

2.  **Distinguish failure severity.** A run that fell over twice per env still
    scores higher than one that fell five times, and a run with a downward
    fell_over slope scores higher than one that's stuck. This lets the planner
    move toward better regions even before any run actually passes.

3.  **Penalise early aborts.** Runs that die at iter 750 are not the same as
    runs that completed 3000 iters with the same final-window stats. The
    scoring takes ``n_iterations`` into account when available.

4.  **Keep passing-vs-failing easy to read.** Scores in [0.0, 0.30) mean at
    least one hard quality gate failed; [0.30, 1.0] means all gates passed
    and the score reflects gait + tracking quality.

Priority ordering (per project goals):
    Stability  >  Gait quality  >  Velocity tracking
"""

from __future__ import annotations

from typing import Any, TypedDict, cast


class CompressedMetrics(TypedDict):
  """Late-training-window averages from a completed run.

  Computed over the final ~20% of iterations of the run, NOT the whole run,
  because early training values are noise from a partly-trained policy.
  See metrics_compression.py.
  """

  # Per-component reward terms (averaged over final window)
  track_linear_velocity: float
  track_angular_velocity: float
  upright: float
  pose: float
  action_rate_l2: float  # negative
  cot_proxy: float  # negative
  limb_symmetry: float  # negative
  air_time: float
  gait_phase_regularity: float  # negative
  foot_clearance: float  # negative
  foot_swing_height: float  # negative
  foot_slip: float  # negative

  # Termination rates (averaged over final window)
  fell_over_rate: float
  time_out_rate: float

  # Raw metrics (averaged over final window)
  gait_air_cv_mean: float
  gait_contact_cv_mean: float
  slip_velocity_mean: float
  air_time_mean: float
  landing_force_mean: float
  locomotion_speed_mean: float
  symmetry_pos_cost_mean: float
  symmetry_vel_cost_mean: float

  # Training-side health signals
  mean_episode_length: float
  mean_action_std: float
  entropy_loss: float


# Hard-gate thresholds. Failing any of these caps the score at GATE_FAIL_CEILING.
STABILITY_GATE_FELL_OVER = 0.5
MIN_EP_LENGTH = 500.0

# Anything that fails a gate sits in [0, GATE_FAIL_CEILING]; anything that
# passes sits in [GATE_FAIL_CEILING, 1.0]. This keeps the boundary obvious
# while still ranking failed runs against each other.
GATE_FAIL_CEILING = 0.30

# Default planned iteration count. The orchestrator passes the real value;
# this is just a fallback for the smoke test and standalone calls.
DEFAULT_PLANNED_ITERATIONS = 3000


def normalize_negative_penalty(value: float, scale: float) -> float:
  """Map a negative penalty (closer to 0 = better) to [0, 1] (1 = better).

  Uses 1 / (1 + |value| / scale) — soft, monotonic, no hard clipping.
  ``scale`` should be the rough magnitude where the penalty starts to matter.
  """
  return 1.0 / (1.0 + abs(value) / scale)


def normalize_positive_reward(value: float, scale: float) -> float:
  """Map a positive reward (higher = better) to [0, 1]."""
  if value <= 0:
    return 0.0
  return value / (value + scale)


def stability_quality(fell_over_rate: float, fell_over_slope: float) -> float:
  """Smooth stability signal in [0, 1].

  Saturating in fell_over_rate so a run with fell_over=2.0 still scores above
  a run with fell_over=4.0. A negative slope (improving over the run) earns a
  small bonus — useful for partial-run promise checks.
  """
  base = 1.0 / (1.0 + (fell_over_rate / 0.5) ** 2)
  if fell_over_slope < 0:
    # Slope is per-iteration. -5e-4/iter over a 3k probe = -1.5 over the run,
    # which is a serious recovery — worth ~+0.10. Clip the bonus.
    bonus = min(0.10, abs(fell_over_slope) * 200.0)
    base = min(1.0, base + bonus)
  return base


def episode_length_quality(ep_length: float) -> float:
  """Soft saturating signal on mean episode length, in [0, 1]."""
  if ep_length <= 0:
    return 0.0
  return ep_length / (ep_length + 300.0)


def completion_quality(n_iterations: int, planned_iterations: int) -> float:
  """Fraction of planned iterations actually reached, in [0, 1]."""
  if planned_iterations <= 0:
    return 1.0
  return min(1.0, n_iterations / planned_iterations)


def gait_quality_score(m: CompressedMetrics) -> float:
  """Weighted sum of gait-quality signals, mapped to [0, 1]."""
  components = [
    (normalize_negative_penalty(m["gait_phase_regularity"], scale=0.05), 0.20),
    (normalize_negative_penalty(m["gait_air_cv_mean"], scale=0.3), 0.10),
    (normalize_negative_penalty(m["gait_contact_cv_mean"], scale=0.3), 0.10),
    (normalize_negative_penalty(m["limb_symmetry"], scale=0.005), 0.15),
    (normalize_negative_penalty(m["action_rate_l2"], scale=0.3), 0.15),
    (normalize_negative_penalty(m["foot_slip"], scale=0.005), 0.10),
    (normalize_negative_penalty(m["slip_velocity_mean"], scale=0.05), 0.05),
    (normalize_positive_reward(m["air_time"], scale=0.001), 0.10),
    (normalize_negative_penalty(m["landing_force_mean"], scale=80.0), 0.05),
  ]
  total_weight = sum(w for _, w in components)
  return sum(v * w for v, w in components) / total_weight


def velocity_tracking_score(m: CompressedMetrics) -> float:
  """Weighted sum of tracking signals, mapped to [0, 1]."""
  components = [
    (normalize_positive_reward(m["track_linear_velocity"], scale=0.15), 0.60),
    (normalize_positive_reward(m["track_angular_velocity"], scale=0.10), 0.40),
  ]
  return sum(v * w for v, w in components)


def _extract_fell_over_slope(summary: dict[str, Any]) -> float:
  """Pull the late-trend slope out of the compressed-summary trends block."""
  trends = summary.get("trends", {}) or {}
  entry = trends.get("Episode_Termination/fell_over", {}) or {}
  return float(entry.get("slope_per_iter", 0.0))


def score_run(
  summary: dict[str, Any],
  planned_iterations: int = DEFAULT_PLANNED_ITERATIONS,
) -> dict:
  """Compute the overall score for a run, with a diagnostic breakdown.

  Accepts either a full compressed summary (with ``final_window``,
  ``trends``, ``n_iterations``) or — for backwards compatibility — a bare
  metrics dict, in which case slope and iteration count default to neutral
  values.
  """
  if "final_window" in summary:
    m = cast(CompressedMetrics, summary["final_window"])
    n_iterations = int(summary.get("n_iterations", planned_iterations))
    fell_over_slope = _extract_fell_over_slope(summary)
  else:
    m = cast(CompressedMetrics, summary)
    n_iterations = planned_iterations
    fell_over_slope = 0.0

  gates_failed: list[str] = []
  if m["fell_over_rate"] > STABILITY_GATE_FELL_OVER:
    gates_failed.append("stability")
  if m["mean_episode_length"] < MIN_EP_LENGTH:
    gates_failed.append("ep_length")

  components = {
    "stability": stability_quality(m["fell_over_rate"], fell_over_slope),
    "episode_length": episode_length_quality(m["mean_episode_length"]),
    "completion": completion_quality(n_iterations, planned_iterations),
    "gait_quality": gait_quality_score(m),
    "velocity_tracking": velocity_tracking_score(m),
  }

  if gates_failed:
    # Failing run: rank by stability + completion + ep_length, with gait /
    # tracking only contributing a thin sliver. Compressed into [0, 0.30].
    raw = (
      0.55 * components["stability"]
      + 0.20 * components["episode_length"]
      + 0.15 * components["completion"]
      + 0.07 * components["gait_quality"]
      + 0.03 * components["velocity_tracking"]
    )
    score = GATE_FAIL_CEILING * raw
  else:
    # Passing run: gait > tracking > stability margin. Lives in
    # [GATE_FAIL_CEILING, 1.0] so passing always beats failing.
    quality = (
      0.55 * components["gait_quality"]
      + 0.20 * components["velocity_tracking"]
      + 0.15 * components["stability"]
      + 0.05 * components["episode_length"]
      + 0.05 * components["completion"]
    )
    score = GATE_FAIL_CEILING + (1.0 - GATE_FAIL_CEILING) * quality

  reason = None
  if gates_failed:
    parts = []
    if "stability" in gates_failed:
      parts.append(
        f"fell_over_rate={m['fell_over_rate']:.2f} > {STABILITY_GATE_FELL_OVER}"
      )
    if "ep_length" in gates_failed:
      parts.append(
        f"mean_ep_length={m['mean_episode_length']:.0f} < {MIN_EP_LENGTH:.0f}"
      )
    reason = "; ".join(parts)

  return {
    "score": score,
    "gates_failed": gates_failed,
    "reason": reason,
    "components": components,
    "n_iterations": n_iterations,
    "planned_iterations": planned_iterations,
    "fell_over_slope": fell_over_slope,
    # Legacy keys preserved so older readers and the analyst's prompt still
    # see familiar fields. Prefer ``components`` and ``gates_failed`` going
    # forward.
    "gate_failed": gates_failed[0] if gates_failed else None,
    "gait_quality": components["gait_quality"],
    "velocity_tracking": components["velocity_tracking"],
  }


if __name__ == "__main__":
  # Smoke test against a real compressed summary.
  import json
  import sys
  from pathlib import Path

  if len(sys.argv) < 2:
    print("Usage: python scoring_function.py /path/to/compressed.json")
    sys.exit(1)

  with open(Path(sys.argv[1])) as f:
    summary = json.load(f)
  print(json.dumps(score_run(summary), indent=2))
