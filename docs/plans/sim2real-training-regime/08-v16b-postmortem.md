# v16b post-mortem (2026-07-03) and v16c spec

v16b (`5eccb3d`, seeds `obp8h6ol`/`zf7ad97s`) never learned to stand: episode
length stuck at 22–30 control steps (~0.5 s) for 1,400+ iterations, both
seeds identically. Observed policy behavior: throws itself backwards and
falls — i.e. **learned fast termination**, not physical inability.

## Evidence (local A/B tests + W&B)

1. **Static stand test** (zero actions, PD holds keyframe, no DR/pushes):
   the robot STANDS for 3 s under the v16b actuator params AND under the
   restored pre-cut efforts. Physics does not prevent standing.
   (Test-harness gotcha for future agents: `reset_base` and
   `reset_robot_joints` are events — stripping "all events" for a clean test
   also removes state reset and produces garbage.)
2. **Random-action survival** (N(0,1) actions ≈ untrained policy):
   v16b efforts mean 43 steps / max 58; restored efforts mean 54 / max 131.
   The effort cut truncates the exploration tail — the long lucky episodes
   that bootstrap learning — but is not alone fatal.
3. **Per-term W&B table at the v16b equilibrium**: every term tiny; net
   ≈ +0.02/step. No single dominating penalty *at that equilibrium* — the
   trap is counterfactual: any movement toward standing/walking passes
   through a flail region that the now-always-on penalties price steeply.
4. **Calibration fact from v13** (healthy walker, old physics):
   `joint_acc_l2` at −1e-4 cost **−1.07/step** — the largest penalty in the
   entire stack, ~⅓ of gross tracking reward. During untrained flailing it
   is several times larger. v16 escaped the flail valley because Phase-C
   penalties were ZERO for its first ~1000 iters; v16b priced the valley
   from step 0 and both seeds settled on "minimize movement, tip backward,
   die" (backward = the cheap direction: ankle_pitch range is asymmetric,
   (−0.6, 1.0)).

## Root causes (two, compounding)

**R1 — Un-specced effort-limit cut.** `5eccb3d` reinterpreted
`effort_limit` as *rated continuous* torque (MX-106 11.086→5.43, XH540
11.086→6.66, MX-64 6.16→3.70 Nm) on top of the DcMotor speed-derating curve
and the (0.7,1.2) effort DR. The continuous rating is a thermal average,
not an instantaneous cap — Dynamixel firmware by default allows near-stall
torque transiently, and the DC-motor curve already handles derating.
Doc 06 §V5's wording ("effort_limit = rated/firmware torque cap") was
ambiguous and caused this; the correct reading is the FIRMWARE cap
(≈ stall at bus voltage), not the continuous rating.

**R2 — "Flatten Phase-C from iteration 0" was a flawed recommendation
(doc 06 spec item 5 — superseded here).** The diagnosis (anneal-of-shaping
handoff is bad) stands; the fix overshot. Movement penalties calibrated for
a trained gait must not price early exploration: they create a penalty
valley between "die fast" and "walk" that PPO cannot cross, especially with
no alive bonus making survival intrinsically worthwhile.

## v16c spec (minimal change-set from v16b)

1. **Restore effort limits**: `effort_limit = saturation_effort = stall
   torque at bus voltage` (MX-106 10.0, XH540 11.7, MX-64 7.3 — the stall
   values already in `nugus_constants.py`), i.e. delete the RATED_TORQUE_*
   usage. Keep the corrected velocity limits and BAM friction. (If a
   distinct firmware current-limit is ever measured on the real servos,
   set `effort_limit` to that and keep saturation at stall.)
2. **Alive bonus**: new reward term, +0.5/step while not terminated (a
   plain constant; gate on nothing). Rationale: makes death never optimal,
   removes the fast-termination equilibrium class entirely; present in the
   FastSAC minimal recipe (`references/fastsac-15min-recipe.md`) and most
   legged-gym-family configs. This also would have blunted the v16 p3
   collapse.
3. **Warm up movement penalties instead of step-0-full**: keep them
   PERMANENT once on (no anneal-away of shaping — doc 06's core finding
   stands) but ramp joule/joint_acc/torque_rate/soft_landing weights
   0 → final over iters 500–1000 (simple `reward_curriculum` stages), or
   gate on `mean_episode_length > 500` via the existing
   `staged_on_plateau`. Success shaping (swing height, air time, base
   height) stays on from step 0 as in clock_persist.
4. Everything else identical to v16b (clock_persist, GAIT_PERIOD 1.0,
   swing 0.05, corrected velocity limits, trimmed DR, hs-critic,
   joule 1e-5).

**Ablation cell worth adding (1 run):** `joint_acc_l2` weight −3e-5 instead
of −1e-4 — v13 data shows −1e-4 consumes ~⅓ of gross tracking reward in a
healthy gait, and the explicit-torque DcMotor actuator likely amplifies
accelerations vs the old builtin actuator it was tuned on.

**Success signal:** mean_episode_length > 800 by iter ~600 (v16 reached 956
by 570 on harsher shaping); then fixed eval falls_per_min per doc 06. If
v16c with restored efforts + alive bonus + warmup STILL cannot stand by
iter 1000, suspect the DcMotor explicit-PD/backlash interaction next
(diagnose with the static+noise test scripts pattern above, sweeping kp).

## Bookkeeping

- v16b vcjobs stopped 2026-07-03 (`mj-gs-v16b-cp-hs-joule-1e5{,-s2}`),
  ~iter 1400/2000 — no path to recovery, both seeds identical.
- Doc 06 spec item 5 ("flatten Phase-C") and §V5's effort_limit wording are
  superseded by this doc.
- The stand/noise test scripts used here follow the pattern in the
  session scratchpad (`stand_test.py`, `noise_test.py`); worth promoting
  into `scripts/tools/` as a pre-launch smoke check: (a) zero-action stand,
  (b) random-action survival distribution — cheap, would have caught both
  v16 and v16b failure modes before burning cluster time.
