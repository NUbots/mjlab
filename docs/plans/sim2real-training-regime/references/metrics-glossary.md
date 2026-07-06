# Metrics glossary — every logged key, what it means, healthy bands

Reference for reading W&B runs from v27 onward. Healthy bands and alarm
values come from measured v20–v27 history, not theory. Organized by
prefix as they appear in W&B.

## Curriculum/aimd_difficulty/* (the difficulty controller, R8)

- `difficulty` — the AIMD scalar d ∈ [0, 1]. 0 = easiest commands
  (±0.20 m/s x) and 0.75× pushes; 1 = the full lerp target (L5 table ×
  `envelope_scale`) and 2.0× pushes. Healthy: climbing early, then a
  sawtooth around capacity. Alarm: parked flat at 1.0 for hundreds of
  iters (saturation churn risk, killed v27) or pinned near 0 late.
- `ssthresh` — TCP-style high-water mark: 0.85 × the d where congestion
  last fired. Above it, d climbs at 1/3 rate (probing a known wall).
  1.0 means no congestion yet.
- `lin_vel_x_max`, `ang_vel_z_max` — the actual command range bounds d
  currently implies (m/s, rad/s). What the robot is being asked to do.
- `push_scale` — push magnitude multiplier implied by d_push (0.75–2.0).
- `difficulty_push` — the push axis's own AIMD scalar (v29+: control is
  split per axis; v28 died of a cohort-blind single scalar — the pushed
  cohort burned above the bar for ~1000 iters while the blended
  population rate read healthy). Congestion signal is the EXCESS rate
  (fast_fall_pushed − fast_fall_clean, bar 0.30); commands and pushes
  now retreat independently, population emergency arrests both.
- `ssthresh_push` — high-water mark for the push axis.
- `landing_factor` — the optimizer-anneal multiplier (R14, LANDING_ANNEAL
  runs). 1.0 = full learning; once the run is at capacity (d >= 0.95
  sustained 200 iters) AND plateaued (attain within 2% of trailing max
  for 150 iters) it decays 0.995/iter (0.9/iter if the attain-slide
  fires anyway), and the runner scales desired_kl by it — the adaptive
  schedule then walks the LR to its floor. Monotone within a run.
  Watch: should reach ~0.1 within ~450 iters of plateau, BEFORE the
  historical ignition window; a run dying with factor still at 1.0
  means the trigger conditions never aged (check d and attain flatness).
- `attain_trailing_max` — sticky running max of clean attainment
  (~14k-iter half-life decay). The attain-slide congestion reference
  (v30+): clean_attain below 95% of this fires a d_cmd cut even with
  falls quiet — under ellipsoid geometry, over-capacity commands
  under-track instead of falling, so the churn signature (attain slide)
  is the command axis's second congestion signal.
- `competence_*` — copies of the tracker population means (see
  diagnostics below) snapshotted by this term.

## Curriculum/competence_diagnostics/* (stratified signals, R9)

Cohort prefix: `clean_` = the never-pushed 70% of envs (uncontaminated
tracking signal, deployment-matched); `pushed_` = the 30% receiving
pushes. Bare `fast_fall_*` names carry the same split.

- `clean_attain` / `pushed_attain` — attainment: achieved velocity
  projected on the commanded direction, as a signed true fraction of
  commanded speed, on steps with |cmd| ≥ 0.15, per-env EMA then cohort
  mean. 1.0 = perfect, 0 = standing still (sandbagging), negative =
  moving AGAINST commands (v23's shattered policy read −0.68). Healthy:
  0.55–0.75; the AIMD increase gate needs > 0.40. Watch the gap between
  clean and pushed: a large gap means pushes are costing tracking.
- `clean_attain_x`, `clean_attain_y` (and `pushed_`) — per-axis
  attainment (R12): achieved_x/commanded_x etc., each sample weighted by
  that axis's share of command energy. Answers whether the table's
  0.75:0.45 x:y ratio matches the robot's real anatomy. Expect x > y
  for a humanoid; a persistent y deficit at equal normalized demand
  means the ellipsoid shape is wrong, not the policy.
- `clean_wobble` / `pushed_wobble` — fraction of steps with torso tilt
  > 25°. The graded near-fall precursor. Healthy: < 0.05; the levels
  controller demoted at > 0.25.
- `clean_fell_ema` / `pushed_fell_ema` — per-env EMA (α=0.1 per episode
  end) of "episode ended in a fall". A RATE in [0, 1] but SLOW: at
  900-step episodes it lags reality by ~200 iterations (the v23
  lesson). Use the fast channels for anything time-critical.
- `fast_fall_clean` / `fast_fall_pushed` — the fast channel: falls ÷
  episode-ends in ~1-iteration windows, EMA α=0.2 (~5–11 iter
  response). THE congestion/crash signal. Healthy: < 0.20; congestion
  bar 0.35 (recoverable, proven twice); 0.5+ = spiral (proven fatal).
- `push_excess_fall` — `fast_fall_pushed − fast_fall_clean`. The fall
  rate attributable to pushes alone (both cohorts share the command
  sampler, so command difficulty cancels). Healthy: small positive.
  Rising while clean stays flat = push difficulty is the binding axis.
- `hazard_cmd_0..7` — clean-cohort fall hazard bucketed by commanded
  SPEED |v_xy|: bucket i covers [0.1·i, 0.1·(i+1)) m/s. Value = falls
  per exposure step in that bucket (EMA over ~50-iter windows).
  5e-4 ≈ one fall per 2000 steps ≈ one per ~40 s walking at that speed.
- `frontier_speed` — midpoint of the fastest speed bucket still under
  the hazard bar (default 5e-4), scanning up from slow. The measured
  "how fast can it reliably go". Cross-validate against where the AIMD
  sawtooth settles; agreement is what earns the estimator control.
- `hazard_rho_0..7` — same hazard, bucketed by Mahalanobis radius
  ρ = √((vx/Rx)² + (vy/Ry)² + (ω/Rω)²) with R the CURRENT per-axis
  range maxima: bucket i covers [0.2·i, 0.2·(i+1)). ρ ≈ 1 is the
  ellipsoid surface; under box sampling ρ > 1 exists (the corners, up
  to ~1.7) and those buckets measure the corner cost directly (R11).
  If hazard_rho rises with ρ while hazard_cmd is flat across speed,
  falls are driven by axis COMBINATIONS, not raw speed.
- `frontier_rho` — the ρ-version of frontier_speed.
- `frontier/hazard_by_speed`, `frontier/hazard_by_rho`,
  `frontier/push_fall_dt` — W&B Histogram renderings of the same bucket
  data (heatmap over training steps; the human view). The per-bucket
  scalars remain the programmatic source of truth.
- `push_fall_within_{0.5,1.0,2.0,4.0,8.0}s` — cumulative fraction of
  pushed-cohort falls that happened within that many seconds of the
  last push. The empirical recovery-time distribution (answers the
  "how long is a recovery horizon" question; v27 measured only 20%
  within 1 s — a 1 s horizon would misattribute 80%).

## Curriculum/track_reward_watchdog/* (fail-fast, user rule)

- `ema` — 0.05/iter EMA of the exact value logged as
  Episode_Reward/track_linear_velocity. Healthy: 1.3–2.4.
- `armed` — 1.0 once ema has exceeded 2.0 (every healthy run arms by
  ~iter 800).
- `below_count` — consecutive iterations with ema < 1.0 while armed.
  At 60 the run RAISES and fails fast (rotted policy). Bounce
  transients (≤45 iters) never reach it.

## Curriculum/joule_lambda_shadow/* (Lagrangian pilot, R10 — log-only)

- `lambda` — the energy multiplier the constrained controller WOULD
  apply. Rises (additively, ~1000-iter full ramp) only while all style
  gates hold; retreats ×0.8 on style collapse.
- `live_weight` — the staged joule weight actually in force (for
  comparison; shadow validation = lambda behaves at least as sanely).
- `peak_ema` / `peak_frac` — slow EMA of Metrics/peak_height_mean and
  its fraction of a slowly-decaying trailing max. peak_frac is the
  foot-lift-collapse gate: rise requires ≥ 0.85, retreat at < 0.70
  (v25-slow slid 0.0128→0.0072 = 0.56 — would have retreated in time).
- `gates_ok` — 1.0 when all rise conditions hold (peak_frac ≥ 0.85,
  clean_attain ≥ 0.50, fast_fall_clean < 0.20).
- `style_broken` — 1.0 when peak_frac < 0.70 (retreat firing).

## Curriculum/adaptive_command_level/*, adaptive_push_level/* (legacy ladder)

Used by CURRICULUM_STYLE=levels runs (v22b–v25). `level` = table index
(commands 0–5, pushes 0–5 scale table); `*_min/_max` = the range bounds
at that level; `competence_*` = tracker means as above.

## Curriculum/{joule_heating,joint_acc_l2,soft_landing,torque_rate}_competence/*

Staged penalty gates: `stage_idx` = which weight stage is active,
`weight` = the current penalty weight. Demote-on-instability drops
stages when competence is badly lost (weights retreat toward 0 during
crashes — that is by design, not a bug).

## Metrics/* (reward-term instrumentation)

- `peak_height_mean` — mean swing-foot apex height (m) over feet that
  landed this step. Healthy walking: 0.010–0.014. The foot-drag /
  penalty-overpressure leading indicator (T4 watches its decay from
  run max; v25-slow fired at 44% below).
- `min_foot_lateral_distance_mean` — center-to-center lateral foot
  separation (m). Healthy: 0.17–0.28. Gate ≥ 0.13 (≈5 cm edge gap);
  monitor trigger < 0.115 (≈3 cm edge gap = self-tripping hazard).
- `feet_min_sep_violation_mean` — mean one-sided violation below the
  0.13 wall. Healthy: ~1e-6 (i.e., never).
- `phase_delta_dev_p95` — 95th percentile |policy phase delta| for
  clock_owned runs. 1.0–1.25 = the policy actively uses its phase
  freedom (good; it PEAKS under stress — that is the escape hatch
  firing). Near 0 = tether too strong.
- `swing_clock_error_mean` — timing mismatch between actual swing and
  the nominal clock.

## Observation normalizer (R15 — no logged key, but read this)

rsl-rl's EmpiricalNormalization updates its running mean/var every step
forever by default, at a 1/count-lagged rate — a learning-rate-immune
feedback loop between the policy's behavior and its own input scaling,
implicated as the root driver of the ~1600-iteration ignition (v32:
identical attainment-slide rate at LR 1e-5 and 2e-4 — a frozen policy
cannot slide; its input normalization can). From R15 on, normalizers
freeze after OBS_NORM_FREEZE_ITERS (default 500) iterations' worth of
samples. If a post-R15 run still ignites at ~1600, this hypothesis is
dead and the next suspect list opens.

## Core rsl-rl keys (the confusing ones)

- `Episode_Termination/fell_over` — falls per STEP averaged over the
  logging window, NOT a rate in [0,1]: multiply by 24 (steps/iter) for
  falls per iteration. Healthy at hard difficulty ≈ 1–2 (≈ 25–50
  falls/iter across the whole population); 5+ = crash in progress;
  100+ = destroyed. Same units for time_out (episodes reaching the cap
  — higher is better).
- `Episode_Reward/<term>` — mean episodic sum ÷ max episode length in
  seconds. For track_linear_velocity: 2.0+ = easy-range mastery,
  1.3–1.9 = healthy hard-range operation, < 1.0 sustained = rot (the
  watchdog bar).
- `Train/mean_episode_length` — steps (cap 1000). Healthy: > 850 at
  any difficulty. 500s = heavy falling; < 300 = shattered.
- `Policy/mean_std` — action noise σ. Floor-clamped at STD_MIN=0.13.
  Healthy equilibrium 0.13–0.25. Collapse below 0.13 is impossible by
  construction; sustained RISE late (v24b: 0.24→0.26 while degrading)
  reads as churn, not exploration.
- `Loss/value` — critic loss. Slow doubling late in a run co-moves
  with policy drift (critic chasing a wandering policy).
- `Perf/total_fps` — environment steps/sec across all ranks. 8-GPU
  multinode at 6144/GPU ≈ 536k; a value matching the 4-GPU baseline
  (~278k at 8192) historically meant the gang silently split into
  independent runs (check for duplicate W&B runs).

## Reading a run in 30 seconds

1. `difficulty` sawtoothing, not parked → curriculum alive.
2. `fast_fall_clean` < 0.2 → not burning.
3. `clean_attain` 0.55+ and flat/rising → actually tracking.
4. `watchdog ema` > 1.3 → not rotting.
5. `peak_height_mean` ≥ 0.010 → not shuffling.
6. `hazard_rho_4..7` vs `hazard_cmd_4..7` → what difficulty actually
   binds (combinations vs speed).
