# Overnight autonomous plan — ~20 runs / 10 hours / 2 slots (2026-07-03)

Budget: 2 concurrent cluster slots × ~10 h ≈ **20 run-hours**. 2000 iters
≈ 1 h; 1000 iters ≈ 30 min; 1400 iters ≈ 45 min. Goal: maximum information
about what makes a good NUgus walk, organized as adaptive waves — each wave's
cells are chosen from the previous wave's results, so read the decision
rules, not just the run list. v16d (wave 0) is already in flight.

## Operating rules (apply to every run)

- **Judge on Metrics, never `Train/mean_reward`** (weights differ across
  cells): `twist/error_vel_x|y|yaw`, `air_time_mean`, `foot_tilt_mean`,
  `gait_air_cv_mean`, `landing_force_mean`, `slip_velocity_mean`,
  `vel_sat_frac_*`, `Episode_Termination/fell_over`, ep_len.
- **Kill-early rule:** if `mean_episode_length` < 600 by iter 500, kill the
  run and log it as a bootstrap failure (v16c reached ep_len ~970 by iter
  389 — anything far off that is broken, don't burn the hour).
- One changed axis per cell. Integrated configs get their own integrator
  cell before being promoted to a new base.
- Keep both slots busy; when a slot would idle, run `nugus_eval` (≥256
  envs/command) on the latest winner checkpoint — the per-command tracking
  breakdown is itself a key experiment (does error concentrate at high |vx|?
  That confirms the stride-length ceiling, H2 below).
- Record every run in `docs/experiments/2026-07-nugus-gridsearch-summary.md`
  as you go; update the README checklist at the end.
- Every new env knob MUST be added to `volcano-train-job.template.yaml` and
  the config-audit test must pass before launching (the silent-default bug
  class has burned us twice).

## Hypotheses being tested (info targets)

- **H1 (user priority): the reward stack forces flat-foot walking.**
  The intended semantics (user-specified, and what `feet_clearance` already
  ALMOST computes) is: **penalize horizontal translation while close to the
  ground** — `(target − h)²₊ × ‖v_xy‖`. A heel-toe gait is fine under that
  rule: the toe planted as a pivot during toe-off is low but not
  translating (≈ zero penalty); the heel lifts mostly in z first, then
  translates once high. Three flat-foot biases break this today:
  (a) `feet_clearance` measures height as the MIN corner but velocity as
  the FOOT-CENTER site — so a pivoting toe (min corner ≈ 0) gets charged
  with the center's pivot velocity. Fix: per-corner — each corner's own
  below-target clearance × its own xy velocity (existing function, pointed
  at an ungrouped 8-corner height sensor + the 8 corner sites);
  (b) the swing-height reward reads min-corner height, so a pitched foot
  counts as unlifted → move it to foot-center height;
  (c) `foot_flat` penalizes ANY swing-foot sole tilt → make the pitch
  component ONE-SIDED (penalize toe-below-heel only — the digging hazard;
  allow toe-up heel-strike and heel-up toe-off attitudes), keep roll
  two-sided. NOTE: `foot_flat` is already contact-gated to swing feet only,
  and standing flatness comes from physics + the stand-still terms —
  standing behavior needs NO change and must be visually re-verified in the
  winner cell.
- **H2: stride length, not lift, is the tracking ceiling.** At period 1.0 s,
  0.5 m/s ⇒ 0.25 m steps on ~0.45 m hip height (real kid-size steps are
  0.06–0.15 m). Faster cadence (shorter period) may fix tracking better
  than any reward change. vel_sat_frac ~0 says the actuators have headroom
  for higher cadence at moderate lift.
- **H3: joint_acc_l2 may be unnecessary entirely** (v16c showed even −3e-5
  costs −0.8/step; torque_rate/action_rate were never binding).
- **H4: the alive+posture income still suppresses speed** (alive 0.5 may be
  more than needed post-bootstrap).
- **H5: seed noise could be masquerading as effects** (v11 seed spread was
  huge; everything since v16 is seed 1).
- **H6: the trimmed DR is hiding robustness debt** (payload ±0.2 / mass
  ±10% vs the plan's eventual ±0.3–0.5 / ±15%).
- **H7 (stretch): an unclocked (`self_paced`) policy can now bootstrap**
  given the alive bonus + air_time economy (doc 07 Stage 4).

## Code tasks BEFORE wave 1 (~30–45 min, do while wave 0 runs)

1. `FOOT_FLAT_W` env knob (default −0.5) on `cfg.rewards["foot_flat"].weight`,
   plus a one-sided pitch option in `feet_flat_orientation`: new param
   `one_sided_pitch: bool` — penalize the fore-aft tangent component only
   when the toe is below the heel (sign it from the foot local axes; write a
   unit test with a toe-down and a toe-up foot pose); roll stays two-sided.
   Knob `FOOT_FLAT_ONESIDED=1`.
2. **Per-corner clearance (H1a):** add `foot_corner_height_scan` —
   a `TerrainHeightSensorCfg` on the same 8 corner sites with
   `group_size=1` (8 ungrouped heights); switch `foot_clearance`'s
   `height_sensor_name` to it and its `asset_cfg.site_names` to the 8
   corner sites, so each corner is charged with its OWN xy velocity
   (`feet_clearance` already sums element-wise — this is config-only apart
   from the sensor block). Knob `CLEARANCE_PER_CORNER=1`. Weight may need
   ÷4 rescale (8 low corners vs 2 low feet when flat-dragging — check the
   logged magnitude against v16d baseline before choosing).
3. **Foot-centre height for the swing reward (H1b):** second sensor
   `foot_center_height_scan` (frames = `left_foot`/`right_foot` sites,
   group_size 1); knob `SWING_HEIGHT_SOURCE=min_corner|center` switches
   `foot_swing_height` and `foot_swing_height_landing`.
4. Template entries for all knobs + regenerate + config-audit test green.
5. (If time) `Metrics/peak_height_mean` on the clock_persist path (doc 09
   telemetry gap) — makes H1 cells directly measurable.

## Waves

### Wave 0 — in flight (R1–R3, ~2 h)
R1 `v16d-main`, R2 `v16d-jacc-3e5`, R3 `v16d-nomirror` (2000 iters each).
**D0:** BASE1 = best cell on Metrics. If mirror < nomirror, drop MIRROR_AUG
from all subsequent cells and file a mirror-map bug note (its physics test
is skipped — doc 09 caveat). If jacc-3e-5 ≈ jacc-1e-5, keep −1e-5.

### Wave 1 — heel-toe + cadence (R4–R9, 1000–1400 iters, ~3.5 h)
All single-axis deltas on BASE1:
- R4 `FOOT_FLAT_ONESIDED=1` at weight −0.5 (1000 it) — H1c isolated: does
  allowing heel-strike/toe-off attitudes (while still punishing toe-digging)
  change the gait?
- R5 full heel-toe package: `CLEARANCE_PER_CORNER=1` +
  `SWING_HEIGHT_SOURCE=center` + `FOOT_FLAT_ONESIDED=1` (1000 it) —
  H1a+b+c (the components only jointly express heel-toe, so R5 is the real
  test; R4 isolates the cheapest single term). Verify standing is still
  flat-footed in the viewer/eval before promoting.
- R6 `GAIT_PERIOD=0.85`, swing 0.065 (1000 it) — H2
- R7 `GAIT_PERIOD=0.7`, swing 0.05 (1000 it) — H2 far point
- R8 `JOINT_ACC_W=0` (1400 it — needs post-warm-up window) — H3
- R9 integrator: best-of{R4,R5} × best-of{R6,R7} combined (1400 it)
**D1:** BASE2 = BASE1 + validated H1/H2/H3 winners (via R9; if R9
underperforms its parents, promote only the single best axis).
Signals: `foot_tilt_mean` (should RISE with heel-toe — it becomes a
descriptive metric, not a target), `air_time_mean` ≥0.15, `landing_force`
not exploding (>2× baseline = slap-landing regression), twist errors,
per-command eval if slot free.

### Wave 2 — economy + variance (R10–R13, ~3 h)
- R10 BASE2, 2000 it, seed 1 (the new reference run)
- R11 BASE2, 2000 it, **seed 2** — H5; if R10 vs R11 spread is large,
  demand ≥2-seed agreement before trusting any wave-1-sized delta
- R12 `ALIVE_W=0.25` (1400 it) — H4
- R13 conditional: if air_time still <0.12 → `AIR_TIME_W=0.25` +
  `foot_clearance` target wired to swing target; else γ=0.97 cell (1400 it)
**D2:** BASE3 = BASE2 ± R12/R13.

### Wave 3 — robustness + reach (R14–R17, ~4 h)
- R14 BASE3 + restored wide DR (payload −0.3/+0.5 kg, link mass ±15%), 2000 it — H6
- R15 BASE3 + push interval scale ×1.5, 2000 it — first step back toward
  the hard task, single-axis this time (v17's decoupling question, answered
  incrementally)
- R16 `MJLAB_VARIANT=self_paced` on BASE3 economy (2000 it) — H7; expect
  high variance; kill-early rule applies; even a mediocre walk is a
  strong signal for doc 07 Stage 4
- R17 backfill: whichever of {γ=0.97, ALIVE 0.25, AIR_TIME 0.25} wasn't run
**D3:** pick FINAL config.

### Wave 4 — consolidation (R18–R20, ~3 h)
- R18 FINAL, 4000 iters (~2 h) — late-training stability (the v15/v16c-s1
  failure mode) with LR_CAP active
- R19 FINAL seed 3, 2000 it — variance bracket
- R20 backfill/stretch: gait-period-as-command prototype (doc 07 Stage 2)
  ONLY if implemented with a unit test proving obs and reward read the same
  per-env period; otherwise a BASE3 rough-terrain taste run
  (`Mjlab-Velocity-Rough-Nubots-Nugus`, 2000 it) — first-ever rough data
  point, informs whether the terrain curriculum is usable at all.

Endgame: run `nugus_eval` + `sim2sim_eval` on R18's final checkpoint;
write the summary; leave a ranked "what mattered" table in the experiments
doc — that table is the whole point of the night.

## What NOT to do overnight

- No hard_continue / TRAINING_REGIME changes beyond R15's single push axis.
- No clock_learned. No CTS/history. No effort/velocity-limit changes (the
  actuator model is finally credible — leave it fixed so tonight's data is
  comparable).
- Don't stack >1 unvalidated axis into a base without an integrator cell.
- Don't trust any delta smaller than the R10-vs-R11 seed spread.
