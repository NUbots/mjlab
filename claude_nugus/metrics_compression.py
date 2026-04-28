"""Compress a full training_log.json into the summary used for scoring + planning.

The raw log has ~50 keys × N iterations. For a 2000-iter probe that's 100,000
numbers — too much to hand to an LLM. This module produces:

  1. A `final_window` — averaged values over the last 20% of iterations,
     used for the scoring function and for "what does this run look like
     when it has settled".

  2. A `trends` summary — for each key metric, fit a simple linear slope
     across the run and classify as 'improving' / 'flat' / 'worsening'.
     Lets the planner reason about whether longer training would help.

  3. A `health_flags` list — simple rule-based detection of pathologies
     (entropy collapse, runaway action_rate, etc.).

The output is what the orchestrator passes to Planner and Analyst agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Keys we care about for the score + summary. Everything else is dropped.
SCORING_KEYS = {
    "Episode_Reward/track_linear_velocity": "track_linear_velocity",
    "Episode_Reward/track_angular_velocity": "track_angular_velocity",
    "Episode_Reward/upright": "upright",
    "Episode_Reward/pose": "pose",
    "Episode_Reward/action_rate_l2": "action_rate_l2",
    "Episode_Reward/cot_proxy": "cot_proxy",
    "Episode_Reward/limb_symmetry": "limb_symmetry",
    "Episode_Reward/air_time": "air_time",
    "Episode_Reward/gait_phase_regularity": "gait_phase_regularity",
    "Episode_Reward/foot_clearance": "foot_clearance",
    "Episode_Reward/foot_swing_height": "foot_swing_height",
    "Episode_Reward/foot_slip": "foot_slip",
    "Episode_Termination/fell_over": "fell_over_rate",
    "Episode_Termination/time_out": "time_out_rate",
    "Metrics/gait_air_cv_mean": "gait_air_cv_mean",
    "Metrics/gait_contact_cv_mean": "gait_contact_cv_mean",
    "Metrics/slip_velocity_mean": "slip_velocity_mean",
    "Metrics/air_time_mean": "air_time_mean",
    "Metrics/landing_force_mean": "landing_force_mean",
    "Metrics/locomotion_speed_mean": "locomotion_speed_mean",
    "Metrics/symmetry_pos_cost_mean": "symmetry_pos_cost_mean",
    "Metrics/symmetry_vel_cost_mean": "symmetry_vel_cost_mean",
    "mean_ep_len": "mean_episode_length",
    "mean_action_std": "mean_action_std",
    "loss/entropy": "entropy_loss",
}

# Trend keys — subset where directional change matters.
TREND_KEYS = [
    "mean_reward",
    "mean_ep_len",
    "loss/entropy",
    "mean_action_std",
    "Episode_Termination/fell_over",
    "Metrics/gait_air_cv_mean",
    "Metrics/gait_contact_cv_mean",
    "Metrics/slip_velocity_mean",
    "Metrics/landing_force_mean",
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _linear_slope(ys: list[float]) -> float:
    """Slope of a least-squares fit y = m*x + b, with x = [0, 1, 2, ...].

    Hand-rolled (no numpy) so this stays light. Returns slope per iteration.
    """
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _classify_trend(slope: float, scale: float) -> str:
    """Classify a slope as 'improving', 'worsening', or 'flat'.

    `scale` is the rough magnitude of the metric — slopes smaller than
    ~1% of scale per iteration are treated as flat.
    """
    threshold = 0.01 * scale
    if abs(slope) < threshold:
        return "flat"
    return "improving" if slope > 0 else "worsening"


# Direction map: which way is "improving" for each trend key?
# +1 means higher = better, -1 means lower = better.
TREND_DIRECTION = {
    "mean_reward": +1,
    "mean_ep_len": +1,
    "loss/entropy": -1,         # lower entropy = more confident, but watch for collapse
    "mean_action_std": -1,
    "Episode_Termination/fell_over": -1,
    "Metrics/gait_air_cv_mean": -1,
    "Metrics/gait_contact_cv_mean": -1,
    "Metrics/slip_velocity_mean": -1,
    "Metrics/landing_force_mean": -1,
}


def compress_log(log_path: str | Path, final_window_frac: float = 0.20) -> dict[str, Any]:
    """Read a full training_log.json and return the compressed summary."""
    with open(log_path) as f:
        log = json.load(f)

    n_iters = len(log["iteration"])
    if n_iters == 0:
        raise ValueError(f"Empty training log at {log_path}")

    window_size = max(1, int(n_iters * final_window_frac))
    window_start = n_iters - window_size

    # 1. Final window averages
    final_window = {}
    for raw_key, friendly_key in SCORING_KEYS.items():
        if raw_key not in log:
            continue
        values = log[raw_key][window_start:]
        final_window[friendly_key] = _mean(values)

    # 2. Trends across the full run
    trends = {}
    for key in TREND_KEYS:
        if key not in log:
            continue
        ys = log[key]
        slope = _linear_slope(ys)
        scale = max(abs(_mean(ys)), 1e-6)
        raw_class = _classify_trend(slope, scale)
        # Re-orient based on what "improving" means for this metric.
        direction = TREND_DIRECTION.get(key, +1)
        if direction == -1 and raw_class != "flat":
            raw_class = "improving" if raw_class == "worsening" else "worsening"
        trends[key] = {
            "slope_per_iter": slope,
            "trend": raw_class,
            "first": ys[0],
            "last": ys[-1],
        }

    # 3. Health flags
    health_flags = []
    if "loss/entropy" in log:
        ent_first = log["loss/entropy"][0]
        ent_last = log["loss/entropy"][-1]
        if ent_last < 0.3 * ent_first:
            health_flags.append(
                f"entropy_collapse: dropped from {ent_first:.2f} to {ent_last:.2f}"
            )
    if "mean_action_std" in log and log["mean_action_std"][-1] < 0.1:
        health_flags.append(
            f"action_std_low: {log['mean_action_std'][-1]:.3f} (under-exploration risk)"
        )
    if "Episode_Termination/fell_over" in log:
        late_fall = _mean(log["Episode_Termination/fell_over"][window_start:])
        if late_fall > 0.5:
            health_flags.append(
                f"unstable: late-window fell_over={late_fall:.2f}"
            )

    return {
        "n_iterations": n_iters,
        "window_iterations": window_size,
        "final_window": final_window,
        "trends": trends,
        "health_flags": health_flags,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metrics_compression.py /path/to/training_log.json")
        sys.exit(1)
    summary = compress_log(sys.argv[1])
    print(json.dumps(summary, indent=2))