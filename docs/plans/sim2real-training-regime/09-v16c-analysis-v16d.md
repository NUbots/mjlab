# v16c analysis and v16d change-set (2026-07-03)

v16c walks — the alive bonus fixed the bootstrap. Three observed defects
(user-reported from watching the policy, all confirmed in metrics): feet
drag along the ground / insufficient lift, a one-sided "limp", and linear
velocity tracking that peaks mid-training then declines.

## Evidence from the two v16c cells

Cells: `3mgdsxzo` (s1, joint_acc −1e-4, ran to ~2000) and `pl1auixy`
(joint_acc −3e-5, infra-crashed at iter 1599 while healthy).

**D1 — The joint_acc warm-up is what kills velocity tracking.** In s1,
`track_linear_velocity` peaks 1.26 at iter 796 — exactly as the penalty
warm-up (iters 500–1000) brings `joint_acc_l2` to −1.69/step — then
declines monotonically: 0.88 @ 1217, 0.69 @ 1643, 0.49 @ 1928;
`twist/error_vel_x` rises 0.21 → 0.52 m/s. Meanwhile **total reward RISES
to 80.1** while tracking collapses: the policy is optimizing the
alive (+0.5) + upright + pose + base_height + swing cocktail and shedding
the risky, acceleration-expensive behavior (actually walking at commanded
speed). Textbook reward hacking, priced in by joint_acc.

The −3e-5 cell confirms causality: trkLin 1.26 @ 865 holding 1.22 @ 1217,
mild decline to 1.08 @ 1583 (vs s1's 0.69 at the same iter); err_x ~0.20–0.25
vs 0.52; fell_over 1.42 vs s1's late 3.75; air_time_mean 0.085–0.094 s vs
s1's 0.038. Even at −3e-5 the term still costs ~−0.8/step. For calibration,
IsaacLab's tuned G1 configs use dof_acc weights in the ~1e-7 range —
−1e-4 was orders of magnitude off; NUgus's small links have naturally huge
qdd, so the raw sum(qdd²) is ~25k in a healthy gait.

**D2 — Foot dragging is incentive-limited, NOT actuator-limited.**
`vel_sat_frac_legs` ≤ 0.8% throughout — the corrected velocity limits are
nowhere near binding. Yet air_time_mean is 0.04–0.09 s against a 0.45 s
swing window. The swing target of 0.05 m plus a small air_time weight
(0.08), against joint_acc/torque penalties and speed-gated foot_clearance
(slow dragging is cheap), makes shuffling the optimum.

**D3 — The limp is unpenalized asymmetry.** `gait_air_cv_mean` ~0.44–0.50
in the healthy cell (coefficient of variation of swing timing across
feet — a direct limp measure). No symmetry mechanism was active: MIRROR_AUG
defaults off AND was missing from the k8s template entirely (as were
LR_CAP/ENTROPY_DECAY/GAMMA) — the C2/C3 hooks were unlaunchable until this
change-set. (s1's late CV of 0.05 is not health — it's the symmetric
shuffle.)

**D4 — s1 destabilized late** (fell 0.25 → 3.75 over iters 800→1928) with
adaptive LR free to sit at 1e-3; the −3e-5 cell was milder. LR_CAP now
applied.

## v16d change-set (implemented in this working tree)

| Change | Value | Fixes | Mechanism |
|---|---|---|---|
| `JOINT_ACC_W` | −1e-5 (main), −3e-5 (bridge cell) | D1 | stop pricing fast leg motion; tracking stays profitable |
| `TRACK_LIN_W` (new knob) | 3.0 (was 2.0) | D1 | raise the marginal value of true speed vs the alive/posture cocktail |
| `SWING_TARGET_HEIGHT` | 0.065 (was 0.05) | D2 | actuators have headroom (vel_sat ~0); demand real lift |
| `AIR_TIME_W` (new knob) | 0.15 (was 0.08) | D2 | reward airborne swings directly |
| `MIRROR_AUG` | 1 (+ template plumbing) | D3 | symmetric policy via data augmentation; `nomirror` cell isolates the effect |
| `LR_CAP`/`LR_CAP_START_ITER` | 3e-4 from iter 1200 (+ template plumbing) | D4 | stop late-training LR excursions |

Unchanged: clock_persist, GAIT_PERIOD 1.0, alive 0.5, warm-up 500–1000
(kept for the OTHER penalties — joule/torque_rate/soft_landing — which are
innocent per the term tables), stall efforts, trimmed DR, hs-critic.

Files: `env_cfgs.py` (TRACK_LIN_W/TRACK_ANG_W/AIR_TIME_W knobs),
`volcano-train-job.template.yaml` (9 new env passthroughs),
`gen-gridsearch.sh` (`gen_v16d`, 3 cells), `scripts/k8s/gen_v16d/`
(manifests).

## Launch checklist (for the implementing agent)

1. Commit + push this change-set. **The generated manifests pin
   `GIT_COMMIT=547c67d` (pre-change) — after pushing, update
   `configmap.yaml`'s pin to the new SHA and REGENERATE:**
   `BATCH=v16d bash scripts/k8s/gen-gridsearch.sh -o scripts/k8s/gen_v16d`
2. Launch all 3 cells (main / nomirror / jacc-3e5), 2000 iters each.
3. v16c s1 can finish (it was ~1950/2000); the crashed jacc-3e5 cell does
   not need a rerun — v16d-jacc-3e5 supersedes it.

## Success signals (fixed thresholds, check ~iter 1500–2000)

- `track_linear_velocity` does NOT decline after the warm-up: ≥1.2 at end
  (v16c s1: 0.49; jacc-3e-5: 1.08).
- `twist/error_vel_x` ≤ 0.15 m/s (v16c: 0.25–0.52).
- `air_time_mean` ≥ 0.15 s (v16c: 0.04–0.09).
- `gait_air_cv_mean` ≤ 0.2 in the mirror cell, AND clearly below the
  nomirror cell (limp fixed by mechanism, not coincidence).
  ⚠️ The mirror map's physics-consistency test is SKIPPED (backlash/delay
  buffers not mirrored) — only involution + slice audits ran. If the mirror
  cell trains WORSE than nomirror, suspect a sign error in the map before
  anything else.
- `fell_over` ≤ 0.5/ep at end.
- Then: `nugus_eval` (≥256 envs/command) + `sim2sim_eval` gate before any
  hardware talk.

## Known telemetry gap

clock_persist does not log a peak-swing-height metric (`peak_height_mean`
existed only on the clock_anneal path). Add it to the persist path — it is
the direct measure of D2 and of the 0.065 target being met; air_time is a
proxy.

## If v16d-main still shuffles

Next levers, in order: raise `AIR_TIME_W` to 0.25; wire `foot_clearance`
`target_height` to `SWING_TARGET_HEIGHT` (currently fixed 0.08 — one-sided
below-target, speed-gated; raising foot speed makes dragging costlier);
drop joint_acc_l2 to 0 entirely and let torque_rate + action_rate carry
smoothness (they were near-zero cost in all tables, i.e. not binding).
