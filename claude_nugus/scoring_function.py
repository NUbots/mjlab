"""Scoring function for evaluating a completed training run.

Takes the compressed metrics summary from a training run and returns a scalar
score. This is the orchestrator's automated proxy for "natural gait quality".
Human feedback is the ground truth; this score is just for ranking experiments
between feedback checkpoints.

Priority ordering (per project goals):
    Stability (gate)  >  Gait quality  >  Velocity tracking

Stability is treated as a hard gate — if the policy is not stable, the score
is 0 regardless of how good other metrics look. Otherwise the score is a
weighted sum of gait quality and velocity tracking components.
"""

from __future__ import annotations

from typing import TypedDict


class CompressedMetrics(TypedDict):
  """Late-training-window averages from a completed run.

  These are computed over the final ~20% of iterations of the run, NOT the
  whole run, because early training values are noise from a partly-trained
  policy. See metrics_compression.py.
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
  fell_over_rate: float  # episodes terminated by falling per env
  time_out_rate: float

  # Raw metrics (averaged over final window)
  gait_air_cv_mean: float  # coefficient of variation of swing duration
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


# Tunable: stability gate threshold. If fell_over rate at end of run is above
# this, the run is treated as a failure (score = 0). Currently set to 0.5
# which is generous — failed policies usually have fell_over > 2.0.
STABILITY_GATE_FELL_OVER = 0.5

# Tunable: minimum episode length to consider a run "real". Below this the
# robot is barely moving and the per-component rewards aren't comparable.
MIN_EP_LENGTH = 50.0


def normalize_negative_penalty(value: float, scale: float) -> float:
  """Map a negative penalty (closer to 0 = better) to [0, 1] (1 = better).

  Uses 1 / (1 + |value| / scale) — soft, monotonic, no hard clipping.
  `scale` should be the rough magnitude where the penalty starts to matter.
  """
  return 1.0 / (1.0 + abs(value) / scale)


def normalize_positive_reward(value: float, scale: float) -> float:
  """Map a positive reward (higher = better) to [0, 1].

  Uses value / (value + scale). Tracking rewards are bounded [0,1] per step
  by their exp(-error/std^2) shape, but they get accumulated over an episode,
  so the post-episode value is unbounded. This squashes safely.
  """
  if value <= 0:
    return 0.0
  return value / (value + scale)


def gait_quality_score(m: CompressedMetrics) -> float:
  """Weighted sum of gait-quality signals, mapped to [0, 1]."""
  # Each component is normalized to [0,1] where 1 = good. Weights sum to 1.
  components = [
    # Phase regularity — most important gait shape signal
    (normalize_negative_penalty(m["gait_phase_regularity"], scale=0.05), 0.20),
    # Direct CV measurements — independent of weight tuning
    (normalize_negative_penalty(m["gait_air_cv_mean"], scale=0.3), 0.10),
    (normalize_negative_penalty(m["gait_contact_cv_mean"], scale=0.3), 0.10),
    # Limb symmetry
    (normalize_negative_penalty(m["limb_symmetry"], scale=0.005), 0.15),
    # Smoothness — too high a magnitude here means twitchy actions
    (normalize_negative_penalty(m["action_rate_l2"], scale=0.3), 0.15),
    # Clean foot contact
    (normalize_negative_penalty(m["foot_slip"], scale=0.005), 0.10),
    (normalize_negative_penalty(m["slip_velocity_mean"], scale=0.05), 0.05),
    # Stepping (not shuffling)
    (normalize_positive_reward(m["air_time"], scale=0.001), 0.10),
    # Soft landing implicit via landing force
    (normalize_negative_penalty(m["landing_force_mean"], scale=80.0), 0.05),
  ]

  score = sum(value * weight for value, weight in components)
  total_weight = sum(weight for _, weight in components)
  return score / total_weight  # safety: normalise in case weights drift


def velocity_tracking_score(m: CompressedMetrics) -> float:
  """Weighted sum of tracking signals, mapped to [0, 1]."""
  components = [
    (normalize_positive_reward(m["track_linear_velocity"], scale=0.15), 0.60),
    (normalize_positive_reward(m["track_angular_velocity"], scale=0.10), 0.40),
  ]
  return sum(value * weight for value, weight in components)


def score_run(m: CompressedMetrics) -> dict:
  """Compute the overall score for a run, with diagnostic breakdown.

  Returns a dict with the final score and the component sub-scores so the
  orchestrator and Analyst can reason about why the score is what it is.
  """
  # Hard gates first — these short-circuit the rest of the score.
  if m["fell_over_rate"] > STABILITY_GATE_FELL_OVER:
    return {
      "score": 0.0,
      "gate_failed": "stability",
      "reason": (
        f"fell_over_rate={m['fell_over_rate']:.2f} "
        f"exceeds gate of {STABILITY_GATE_FELL_OVER}"
      ),
      "gait_quality": None,
      "velocity_tracking": None,
    }

  if m["mean_episode_length"] < MIN_EP_LENGTH:
    return {
      "score": 0.0,
      "gate_failed": "ep_length",
      "reason": (
        f"mean_episode_length={m['mean_episode_length']:.1f} "
        f"below minimum of {MIN_EP_LENGTH}"
      ),
      "gait_quality": None,
      "velocity_tracking": None,
    }

  gait = gait_quality_score(m)
  velocity = velocity_tracking_score(m)

  # 70/30 split per project priority: gait > velocity tracking.
  # Stability is already gated above — no separate term needed.
  combined = 0.70 * gait + 0.30 * velocity

  return {
    "score": combined,
    "gate_failed": None,
    "reason": None,
    "gait_quality": gait,
    "velocity_tracking": velocity,
  }


if __name__ == "__main__":
  # Smoke test against the sample log.
  import json
  import sys
  from pathlib import Path

  log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  if log_path is None or not log_path.exists():
    print("Usage: python scoring_function.py /path/to/training_log.json")
    sys.exit(1)

  with open(log_path) as f:
    log = json.load(f)

  # Build a CompressedMetrics from the last entry in each list.
  # NOTE: this is a smoke test only — real use goes through metrics_compression.py
  last = lambda k: log[k][-1]
  m: CompressedMetrics = {
    "track_linear_velocity": last("Episode_Reward/track_linear_velocity"),
    "track_angular_velocity": last("Episode_Reward/track_angular_velocity"),
    "upright": last("Episode_Reward/upright"),
    "pose": last("Episode_Reward/pose"),
    "action_rate_l2": last("Episode_Reward/action_rate_l2"),
    "cot_proxy": last("Episode_Reward/cot_proxy"),
    "limb_symmetry": last("Episode_Reward/limb_symmetry"),
    "air_time": last("Episode_Reward/air_time"),
    "gait_phase_regularity": last("Episode_Reward/gait_phase_regularity"),
    "foot_clearance": last("Episode_Reward/foot_clearance"),
    "foot_swing_height": last("Episode_Reward/foot_swing_height"),
    "foot_slip": last("Episode_Reward/foot_slip"),
    "fell_over_rate": last("Episode_Termination/fell_over"),
    "time_out_rate": last("Episode_Termination/time_out"),
    "gait_air_cv_mean": last("Metrics/gait_air_cv_mean"),
    "gait_contact_cv_mean": last("Metrics/gait_contact_cv_mean"),
    "slip_velocity_mean": last("Metrics/slip_velocity_mean"),
    "air_time_mean": last("Metrics/air_time_mean"),
    "landing_force_mean": last("Metrics/landing_force_mean"),
    "locomotion_speed_mean": last("Metrics/locomotion_speed_mean"),
    "symmetry_pos_cost_mean": last("Metrics/symmetry_pos_cost_mean"),
    "symmetry_vel_cost_mean": last("Metrics/symmetry_vel_cost_mean"),
    "mean_episode_length": last("mean_ep_len"),
    "mean_action_std": last("mean_action_std"),
    "entropy_loss": last("loss/entropy"),
  }
  print(json.dumps(score_run(m), indent=2))
