# Overnight post-mortem (2026-07-04): one disease, many misdiagnoses

## The finding

**Every run that trained past ~iter 1500 collapsed, regardless of config, from
entropy-driven policy-noise regrowth.** The fixed entropy bonus (0.01) pushes
the action std back up once task reward saturates; rising noise on a
marginally stable biped raises falls; falls degrade the value function; the
spiral compounds.

Traces (Policy/mean_std vs fell_over/ep):

| Run | std @1077 | std @1979 | fell @1077 | fell @1979 |
|---|---|---|---|---|
| v16d-main (2k) | 0.146 | 0.286 | 0.54 | 22.7 |
| base2-ref s1 (2k) | 0.149 | 0.328 | 0.42 | 4.6 |
| v16c s1 (2k, pre-mirror, pre-v16d) | 0.087 | 0.115 | 0.46 | 5.8 |
| final 4k @2531/3476 | 0.184 → 0.080 | — | 245 → 361 | (degenerate fallen attractor; ep_len 23) |

Std bottoms at ~iter 1000–1100 in every run, then regrows; fell_over rises in
lockstep. The 4k run shows the endgame: noise destroys the value function
(~iter 2000–2500), after which the policy converges onto a fallen attractor.

**Eliminated suspects (checked, innocent):** LR (cap engaged and worked —
traces ≤3e-4 after iter 1200); mirror augmentation (v16c s1 pre-dates it and
shows the identical disease); the clock-silencing schedule (gated to
clock_anneal + SILENCE_CLOCK, inactive); any p3-scheduled reward change
(clock_persist has none; warm-up stages end at iter 1000).

**The fix existed all night and was never enabled:** `ENTROPY_DECAY=1`
(runner hook, linear 0.01 → 0.001 over the run; `ENTROPY_START`/`ENTROPY_END`
knobs) defaulted OFF in every overnight generator.

## Corrected hypothesis verdicts

The overnight decision trail compared healthy short runs (1000–1400 it,
pre-collapse) against poisoned long runs (2000+ it, post-collapse). Verdicts
re-graded:

| Overnight verdict | Corrected status |
|---|---|
| D0 "v16d-main only tolerable finisher (fell 5.4)" | All wave-0 cells were post-collapse; ranking reflects collapse depth, not config quality. BASE1 choice happened to be harmless because wave-1 runs were short. |
| H1 partially rejected (R5 heel-toe: foot_tilt flat) | **Still open.** R5 was the most stable run of the night (fell 0.17 @999) with best-in-wave err_x; 1000 iters may be too short for heel-toe to emerge, and per-corner clearance weight was possibly unrescaled (÷4 note in doc 10). Re-test long, with entropy decay. |
| H2 cadence binds tracking — supported (R7) | **VALID** (short-run comparison, pre-collapse). Period 0.7 promoted rightly. |
| H3 joint_acc neutral at 0 vs −1e-5 | Valid short-run result; keep −1e-5. |
| H4 alive cut lost to air bump | Valid (both 1399 it, clean comparison). |
| H5 "seed noise dominates — R10/R11 collapsed" | **WRONG.** R10/R11 collapsed from the entropy disease (they ran 2000 it), not seed variance. Seed spread is still unmeasured post-v16c. |
| H6 wide DR regressed | **UNSOUND** — 2000-it run, confounded by collapse. Re-test. |
| H7 self_paced failed | **UNSOUND** — same confound. Re-test. |
| γ=0.97 regressed | Collapsed EARLIER (onset ~1182) — plausibly a real bad interaction with adaptive-KL; deprioritize rather than re-test. |
| R18/R19 wave-4 results | Void. |

**What survives as real progress (all pre-collapse comparisons):**
R4 FOOT_FLAT_ONESIDED (+air_time), R7 GAIT_PERIOD=0.7/swing 0.05 (best
tracking), R13 AIR_TIME_W=0.25 + clearance-from-swing (air 0.115, fell 1.12
@1399). That stack — "R13 config" — is the legitimate base.

## Gate recalibration (why "nothing passed" overstates failure)

Doc-09 thresholds were calibrated at GAIT_PERIOD=1.0. air_time scales with
the swing window (0.45 × period): at period 0.7 the window is 0.315 s, so the
threshold should be ~0.35 × window ≈ **0.11 s**, which R13 (0.115) passes;
err_x 0.134 ≤ 0.15 passes. Only fell ≤ 0.5 is genuinely unmet — and fell was
measured at 1399 on a run WITHOUT entropy decay; the true steady-state number
is unknown until v16e. Updated gate: err_x ≤ 0.15, air_time ≥ 0.35×swing
window, gait_cv ≤ 0.2, fell ≤ 0.5, measured at ≥2000 it WITH entropy decay,
plus fixed eval.

## v16e spec (next batch)

1. **Cells 1–2:** R13 stack + `ENTROPY_DECAY=1` (`ENTROPY_START=0.01`,
   `ENTROPY_END=0.001`), seeds 1–2, 2000 it. This is the single-variable test
   of the disease fix on the best-known config.
2. **Cell 3:** same, 4000 it (does decay hold the policy at its optimum
   through a long horizon — the R18 question, done right).
3. **After cells 1–2 pass:** re-run the three voided experiments on the
   entropy-fixed base: wide DR, self_paced, R5 heel-toe long (2000 it, and
   check the per-corner clearance magnitude rescale first).
4. Standing change: `ENTROPY_DECAY=1` becomes a default in every future
   generator (add to the standing recommendations in README).

## New process rules (add to all future autonomous sessions)

- **Baseline guard:** a new base must beat the previous batch's best at the
  SAME iteration count. v16d-main (fell 5.4 @2000) vs v16c jacc-3e5 (1.42
  @1583) should have halted the night at wave 0.
- **Health telemetry:** monitor `Policy/mean_std`; alarm when std rises >30%
  above its post-convergence minimum — that is the collapse leading
  indicator (~500 iters of warning before fell_over shows it).
- Long-vs-short comparisons are invalid across a collapse boundary; deltas
  are only meaningful between runs of equal length and equal health.
