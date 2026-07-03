# Context and findings

Evidence base for the plan. Facts here were verified against the repo at
commit `adf2023` and the W&B project
`vincenttumm-the-university-of-newcastle/mjlab` on 2026-07-03.

## 1. Current setup (one-paragraph orientation)

Fork of mjlab (MuJoCo Warp + RSL-RL PPO). Task
`Mjlab-Velocity-Flat-Nubots-Nugus`, built by
`src/mjlab/tasks/velocity/config/nugus/env_cfgs.py` mutating the base velocity
env. 20 actuated DOF + 20 passive backlash joints; Dynamixel servos modeled as
`BuiltinPositionActuator` (MuJoCo native position servo, kp/kd/effort/armature
from `nugus_constants.py`). 50 Hz control (200 Hz physics, decimation 4), 20 s
episodes, 8192 envs/GPU × 4 GPUs on the Volcano cluster. Asymmetric
actor-critic MLPs (512,256,128); actor sees ang-vel, gravity, joint pos/vel,
last action, command, gait clock (sin/cos), with measured IMU/servo noise and
sensor delays; critic additionally sees base lin-vel, foot states, optional
height scan. Gait shaping is clock-based (`feet_swing_height_clock`, period
0.7 s, swing ratio 0.45, target 0.08 m). Full inventory:
`references/codebase-map.md`.

Two gait paradigms exist as `MJLAB_VARIANT`s:

- `clock_anneal` (default): fixed external clock; swing-height reward annealed
  away over phases P1/P2/P3 and replaced by sparse landing/air-time terms.
- `clock_learned`: policy owns the phase via an extra `phase_delta` action
  (action dim 21), regularized to nominal cadence by a staged
  `phase_delta_nominal` penalty.

A `hard_continue` curriculum ("base→hard") ramps, from `CONT_BASE_STEP`
(iter 2000), over ~1000 iters: commands x±0.5→±0.75 / y±0.3→±0.45 /
yaw±0.5→±0.80, push scale ×1→×2, upright weight 0.5→0.25 (std widened),
body_ang_vel and angular_momentum penalties relaxed. Independently, "Phase-C"
ramps in joule (−3e-4 default), joint_acc (−1e-4), torque_rate (−1e-3),
soft_landing (−0.01) and base_height (+0.3) between p2 and p3.

## 2. W&B findings (F1–F5)

Raw data and per-run curves: `references/wandb-run-history.md`.

### F1 — The base→hard "regression" is partly reward accounting; the real signal is falls

The hard stage simultaneously removes reward mass (foot_swing_height 0.75→0,
upright 0.5→0.25) and adds penalties, so `Train/mean_reward` before/after the
ramp measures different MDPs. Decomposed:

- clock_anneal (v14, v15): `track_linear_velocity` holds (1.25–1.3 vs 1.5 at
  peak) but `fell_over` climbs ~0.05 → 0.5–1.1 per episode. Falls are the
  problem.
- clock_learned (v11): tracking itself collapses (1.4 → 0.68) in lockstep
  with the phase-delta ratio decaying (see F3).

Consequence: **no fixed-condition evaluation exists**, so no cross-stage or
cross-variant comparison in v8–v15 is clean. Fixing this is E0.1 and is a
prerequisite for interpreting every experiment in this plan.

### F2 — v15 (20k iters) flatlined; more compute does not help

Both seeds sat at reward 38–40 from iter ~4k through ~15k, fell_over
oscillating 0.5–1.1. Recommendation: stop these runs; record final numbers in
the experiments doc.

### F3 — clock_learned's phase delta always collapses to the nominal clock

Phase-delta usage ratio (`Metrics/phase_delta_nominal_ratio_mean`) at
start → end of every base→hard run: v11 s1 1.03→0.04, s2 1.04→0.32,
s3 1.03→0.25, s4 1.03→0.06; v12 tail-0.2 1.04→0.22; v13 best 1.05→0.05.
The mechanism costs an action dim, a 5-stage penalty anneal, and checkpoint
compatibility, and has never beaten fixed-clock like-for-like. v8's headline
100.5 reward was the degenerate case (ratio ≈ 0 from the start, easy task).
→ Standing recommendation B4: retire the variant until hardware walking works.

### F4 — Privileged critic is a large stability win, and the mechanism points at DR-parameter obs

v10: adding height_scan to the critic → fell_over 0 — on FLAT terrain, where
the scan is nearly constant. The gain is a richer critic (lower value
variance), not terrain info. Prediction: feeding the critic the actually
informative privileged signals — the sampled DR parameters — is a cheaper,
larger win. That is C1. Also: v14/v15 ran `CRITIC_HEIGHT_SCAN=false` and
regressed (fell_over 1.08 vs v10's 0). Keep it true.

### F5 — Smaller confirmed facts

- Joule 3e-4 over-penalizes: v12 tail-0.1 collapsed (28.9, fell 2.75/ep);
  identical config with joule 1e-5 (v13) reached 69.0 / fell 0.25.
- Angular-velocity tracking saturates (~1.6/2.0) while linear tracking is the
  binding constraint in the hard stage.
- pd-tail −0.2 ≫ −0.1 at joule 3e-4 (v12: 67.4 vs 28.9). Interaction with
  joule confirmed by v13.

## 3. Already tried and rejected — DO NOT RE-PROPOSE

| What | Where | Outcome |
|---|---|---|
| Halving upright to 0.5 as a standalone change | v9 | Tanked reward; clock_anneal baseline collapsed (peak 59 → 14.5) |
| Resume-based hard continuation (checkpoint reload into harder cfg) | v10-A | Crashed repeatedly; curriculum desync; phase collapse. Abandoned for single-run base→hard |
| clock_learned pd-tail −0.1 at joule 3e-4 | v12 | Worst run of the sweep (fell 2.75/ep). Only viable at joule 1e-5 |
| Extending base→hard to 20k iters (legacy critic) | v15 | Plateau high-30s/40s; never recovers pre-ramp peak |
| CURRENT_OBS on/off at 2000 iters | v8 | Negligible difference |
| Annealing phase_delta penalty to zero (no tail) | v11 | Gait degrades through hard stage; tail hold added in v12 |

## 4. Known bug history (pattern to respect)

Two *silent no-op* bugs have already cost whole batches:

1. `PROGRESS_BACKSLIDE_W` defaulted 0.0 and was never exported through the
   k8s template until `938b771` — the v9 "backslide penalty" runs didn't have
   it.
2. **Still live at `adf2023`:** the `joint_friction` DR event scales
   `dof_frictionloss`, whose baseline is 0 for every motor joint → the event
   does nothing (see gap G2 below).

Also: checkpoint-resume bugs (fixed `040378d`, `1ef0870`, `a04d83b`), pinned
SHA fetch (`62c809b`, `617dfaa`), curriculum stage ordering on short runs
(`05bc2bd`). Dimension changes (clock_learned action, CURRENT_OBS,
obs-history) break checkpoint resume by design.

## 5. Gap analysis vs 2024–2026 practice

**Already aligned or ahead:** backlash joints in MJCF (ahead of most
published work), measured IMU/servo noise, sensor delays (20–60 ms) and
actuator command delays (5–15 ms), asymmetric actor-critic, per-episode servo
DR incl. intra-episode effort drift, no base_lin_vel in actor, stop/settle
command tails, clock-based gait reward family.

**Gaps, ranked by expected sim-to-real impact:**

| # | Gap | Evidence / source | Fixed by |
|---|---|---|---|
| G1 | **No servo speed saturation.** `velocity_limit=30` in `nugus_constants.py` is ignored by `BuiltinPositionActuator` (only `DcMotorActuator` uses it). Real no-load speeds ≈ 6.6 rad/s (MX-64), 4.7 (MX-106), ~3.2 (XH540-W270) — policy can command 5–10× hardware capability | Verified in `src/mjlab/actuator/builtin_actuator.py` vs `dc_actuator.py`; consequence pattern in Bez thesis (failed transfer) | E0.2a |
| G2 | **Zero gearbox friction; `joint_friction` DR is a no-op** (scales a 0 baseline) | Verified: no `frictionloss` in `nugus_constants.py` or XML motor joints; BAM shows friction dominates cheap servos | E0.2b, A1 |
| G3 | **No fixed evaluation protocol** | F1 | E0.1 |
| G4 | **No mass/inertia DR** (only torso COM offset ±2.5 cm) | CTS uses ±20% link mass; universal in successful recipes | A3 |
| G5 | **Hard stage changes ~6 things at once on a step schedule** | F1/F2; modern recipes keep reward stacks constant | B1, B2, B3 |
| G6 | **No symmetry mechanism** (limb_symmetry wired but disabled) | In modern minimal recipes (FastSAC paper, equivariant-policy work) | C2 |
| G7 | **Critic lacks DR-parameter obs** | F4 | C1 |
| G8 | Single-frame actor, no adaptation module | Literature: not needed for first flat-ground transfer; useful later | C4 (deferred) |
| G9 | **No hardware data has ever constrained the sim** | — | D2, A4 |

## 6. Uncertainty register (honest state of consensus)

- Identification-vs-randomization: the systematic study claims zero-DR
  transfer after good ID; the safe synthesis used in this plan is *identify,
  then randomize modestly around identified values*.
- Off-policy (FastSAC/FastTD3): large wall-clock wins reported, but young;
  this plan does NOT migrate the stack.
- CTS/adaptation: gains demonstrated mostly on quadrupeds/rough terrain;
  deferred until hardware evidence demands it.
- Exact Dynamixel speed/friction numbers in this plan are datasheet/BAM-repo
  starting points, flagged VERIFY where they must be checked against the
  actual robot.
- v15 was still running when analyzed (iter ~15k); conclusions about it are
  provisional but the 11k-iteration plateau makes reversal unlikely.
