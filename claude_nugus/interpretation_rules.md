# Interpretation Rules for Training Metrics

This document encodes domain knowledge that an LLM looking at training metrics
in isolation would not have. The Planner and Analyst agents read this file as
part of their context.

When a rule below conflicts with a metric pattern you observe, the rule wins —
metrics in isolation are easy to misinterpret.

---

## 1. Mean reward going more negative is not necessarily bad

If `mean_reward` becomes more negative while `mean_ep_len` is increasing, the
policy is **surviving longer and accumulating bounded penalties over more
steps**. This is normal early-training behaviour and is NOT a regression.

Rule: when comparing reward across runs or iterations, compare **per-step
reward rate** (`mean_reward / mean_ep_len`), not raw reward.

---

## 2. `fell_over` rate spiking early is expected

In the first ~500 iterations, `Episode_Termination/fell_over` typically rises
from 0 (the robot was standing still) to several per env (it's learning to
move and that destabilizes it). It then drops back toward 0 as the policy
masters balance.

Rule: only treat sustained `fell_over > 0.5` past iteration ~750 as a
stability failure. Early spikes are not meaningful.

---

## 3. Entropy declining slowly is good; sharp drops are bad

`loss/entropy` is the action distribution entropy. A slow monotonic decline
indicates the policy is becoming appropriately confident as it learns. A
sharp drop (e.g. losing >50% over 100 iterations) indicates **exploration
collapse** — the policy has prematurely committed to a behaviour and stopped
exploring alternatives.

Rule: classify entropy trend as "healthy" if slope is negative but small.
Flag "entropy_collapse" if slope is steeply negative or final value is below
~30% of starting value.

---

## 4. `mean_action_std` is the second exploration signal

Starting value is typically ~1.0 (Gaussian policy initialization). Healthy
endpoint is around 0.2-0.4. Below 0.1 means the policy is over-deterministic
and may be stuck in a local optimum.

---

## 5. Curriculum stage matters for what metrics mean

`Curriculum/command_vel/lin_vel_x_max` reveals the current curriculum stage.
Early in training, commanded velocities are small (e.g. ±0.5 m/s). Tracking
rewards saturate easily at low speeds — high tracking values early do NOT
mean the policy can track high speeds.

Rule: only compare tracking rewards across runs that are at the same
curriculum stage. If comparing across stages, be explicit about it.

---

## 6. Reward components compete — interpret them as a system

Increasing one reward weight will usually decrease the achieved value of
another reward component, because the policy's behaviour is a compromise
across the whole reward landscape. For example:

- Increasing `gait_phase_regularity` weight (more negative) → expect
  `track_linear_velocity` to decrease (the policy now spends learning capacity
  on rhythm instead of speed).
- Increasing `action_rate_l2` weight (more negative) → expect smoother but
  slower-reacting policy; tracking metrics may degrade.
- Increasing `pose` weight → expect reduced range of motion in joints with
  small std values; can flatten gait.

Rule: when proposing a change, predict and check at least one *other* metric
that should move as a side effect. If it doesn't, something unexpected is
happening and the result is suspect.

---

## 7. Foot height metrics: clearance vs swing height

`foot_clearance` is a continuous penalty on deviation from target height,
weighted by foot velocity. `foot_swing_height` is evaluated at landing and
penalises peak swing height error. They serve different purposes:

- `foot_clearance` shapes the trajectory throughout swing
- `foot_swing_height` shapes the apex of the swing

Tune them together if both target heights match. Diverging target heights is
unusual and probably a mistake.

---

## 8. Landing force interpretation

`Metrics/landing_force_mean` is in Newtons. For a humanoid Nugus-sized robot (7.65kg ~ 76 Newtons in weight),
typical values:

- Below ~30 N: very soft, possibly unrealistic / floaty
- 30-90 N: healthy range
- Above 100 N: hard impacts, will cause hardware concerns and unnatural gait

Rule: do not aggressively reduce `soft_landing` weight unless
`landing_force_mean` is clearly above ~90 N.

---

## 9. Symmetry metrics: position vs velocity

`Metrics/symmetry_pos_cost_mean` measures positional asymmetry (how
differently the left and right legs are bent). `Metrics/symmetry_vel_cost_mean`
measures velocity asymmetry (how differently they're moving).

Position symmetry can be near-zero with poor velocity symmetry — this looks
like a "mirror-pose limp": the legs hold the same shape but at different
phases of the gait. If `symmetry_pos_cost_mean` is small but
`symmetry_vel_cost_mean` is large, that is the diagnostic signal.

---

## 10. The CV (coefficient of variation) gait metrics

`Metrics/gait_air_cv_mean` and `Metrics/gait_contact_cv_mean` measure the
spread of swing/stance durations across feet. Low CV = regular periodic gait.

- Below ~0.15: very regular, healthy
- 0.15-0.4: somewhat regular
- Above 0.4: irregular, suggests unstable or limping gait

These are **direct measurements** independent of reward weights — they're the
most trustworthy gait quality signal we have without human review.

---

## 11. Don't draw conclusions from < 1000 iterations of training

The early phase of training is dominated by curriculum effects, exploration
noise, and the policy "finding" basic locomotion. Any metric values from the
first ~1000 iterations are not predictive of where the run will end up.

Rule: when scoring a run, use only the late window (we use the final 20% of
iterations). When deciding whether to abort a run early, only use stability
gates (`fell_over` past iter 500, NaN losses), not gait quality metrics.

---

## 12. Limit yourself to 2-3 parameter changes per experiment

Even though many parameters could be tuned simultaneously, change only 2-3
at a time. Otherwise you cannot attribute observed changes to any single
parameter, which makes the experiment history useless for future planning.

If multiple changes seem coupled (e.g. `foot_clearance_target_height` and
`foot_swing_height_target_height`), it is fine to count them as one
"semantic change" and adjust both together.