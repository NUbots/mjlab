# Competence-based ("edge of competence") curriculum — implementation plan

Replace time-scheduled curricula with controllers that adjust difficulty from
measured competence. Motivations, in order:

1. **Training speed.** Time schedules waste compute on both sides of the
   frontier: too-easy phases run to their timer even when the policy was
   competent hundreds of iterations earlier, and too-hard jumps destroy
   progress that must be re-learned (v11–v16: every collapse in this
   project's history is a scheduled change landing on a policy that wasn't
   ready, or shaping being removed on a timer). Post-Phase-0 physics made
   the task harder, so the calibration of every hand-tuned timer is stale —
   competence gating self-calibrates.
2. **It replaces `hard_continue` entirely.** The B-track question ("which
   hard-stage component destabilizes?") dissolves: each axis ramps at its
   own measured pace instead of four axes jumping on a shared timer.
3. **Evidence.** Grow-on-success is standard for terrain
   (`terrain_levels_vel` is already in-tree); [ADR](https://arxiv.org/abs/1910.07113)
   scaled it to DR widths; [CHRL](https://arxiv.org/pdf/2310.15583) adapts
   commands + DR + reward coefficients from policy performance;
   [TransCurriculum (2026)](https://arxiv.org/abs/2603.14156) shows
   multi-dimensional adaptive curricula beat velocity-only ones (18.8%
   faster peak velocity on hardware, sim-to-real transfer loss 27%→18%).
   We implement the simple rule-based version, not learned teachers.

**Throughput caveat (check first, separately):** "training got slower" has
two components — sample efficiency (this plan) and raw steps/sec. Compare
`Perf/*` fps between a v13-era run and a current run: Phase-0 physics
(frictionloss constraints, DcMotor), extra sensors (8 corner rays), and
MIRROR_AUG (doubles effective batch) all cost wall-clock per iteration. If
fps regressed >30%, file that as its own issue — no curriculum fixes it.

## Design

### Competence signals (per-env, aggregated to population)

Computed at episode end (reset), EMA'd over the last ~10 episodes per env:

- `track_err`: mean ‖cmd_xy − v_xy‖ over the episode's commanded-walking
  steps, **normalized by command magnitude** (‖err‖ / max(‖cmd‖, 0.2)) so
  small commands can't game the signal by standing still.
- `fell`: 1 if the episode ended in fell_over, else 0.
- `ep_len_frac`: episode length / max length.

Population competence = mean over envs. All three logged under
`Curriculum/competence_*`.

### The controller (one reusable class, three appliers)

A generic `CompetenceController` with the stability features that make
these loops safe:

- **Promote** one level when competence has been above the promote
  threshold for `cooldown` consecutive checks.
- **Demote** one level when below the demote threshold (single check —
  react fast to breakage).
- **Hysteresis:** promote threshold strictly harder than demote (e.g.
  promote if `track_err_norm < 0.25` AND `fell_ema < 0.3`; demote if
  `track_err_norm > 0.45` OR `fell_ema > 1.0`).
- **Cooldown:** ≥50 iterations between changes (let PPO respond before
  re-measuring).
- **Bounded, monotone-mapped levels:** level `L ∈ {0..L_max}` maps to
  difficulty via fixed tables; one step per change; never skips.
- **State in checkpoints:** level + EMAs go through the existing
  curriculum-snapshot machinery (`MjlabOnPolicyRunner` already persists
  curriculum state — the resume bugs of `040378d` taught us this).
- **Logged:** `Curriculum/<axis>_level` — these curves replace "which iter
  are we at" as the training-progress readout, and are the primary
  cross-run comparison (level-vs-iteration = learning speed).

### Axis 1 — commands (`adaptive_command_level`)

Level table (example, tune endpoints to the v16e winner):

| L | lin_x | lin_y | ang_z |
|---|---|---|---|
| 0 | ±0.20 | ±0.05 | ±0.10 |
| 1 | ±0.30 | ±0.10 | ±0.25 |
| 2 | ±0.40 | ±0.20 | ±0.40 |
| 3 | ±0.50 | ±0.30 | ±0.50 | ← current "base" endpoint
| 4 | ±0.60 | ±0.35 | ±0.65 |
| 5 | ±0.75 | ±0.45 | ±0.80 | ← old "hard" endpoint

Implementation: a curriculum term that mutates the command term's ranges
exactly as `commands_vel` stages do today — same mutation, different
trigger. Replaces the 3-stage step schedule. **Phase 2 (only if needed):**
per-env levels à la `terrain_levels_vel`, requiring per-env range support
in `UniformVelocityCommand`; skip until population-level saturates.

### Axis 2 — pushes, then DR widths (`adaptive_push_level`, ADR-lite)

Same controller on the push event: level maps to push velocity scale
{0.75, 1.0, 1.25, 1.5, 1.75, 2.0} (mutation code exists in
`push_robot_curriculum`). Gate on `fell_ema` only (pushes are about
stability, not tracking). Once proven, add DR-width axes the same way:
payload range, link-mass range, friction range each get a level — this is
ADR-lite and subsumes the "restore wide DR" TODO.

### Axis 3 — movement-penalty gating (replaces the Phase-C time warm-up)

Penalties (joint_acc, torque_rate, soft_landing, joule) turn on and ramp
only when stability competence holds (`ep_len_frac > 0.8` AND
`fell_ema < 0.5`), stepping 25% of final weight per cooldown window while
competence persists; freeze (don't ramp further) if competence dips. Never
ramp DOWN automatically — a penalty appearing and disappearing is
non-stationarity of the worst kind; freeze is the safe failure mode.
Reuse/extend `staged_on_plateau` (in-tree, tested) into `staged_on_metric`:
stage advance gated on a metric predicate instead of a reward plateau.

### Explicitly out of scope

- Learned curriculum teachers (TransCurriculum-style) — rule-based first.
- Adapting the gait-shaping rewards (swing height/air time) — those define
  the task; adapting them re-opens the anneal-collapse class.
- Entropy schedule stays time-based for now (it works); optional later:
  hold `ENTROPY_END` until command level saturates.

## Implementation steps

1. `src/mjlab/envs/mdp/curriculums.py` (or a nugus-local module first):
   `CompetenceTracker` (per-env episode stats at reset → EMAs) +
   `CompetenceController` (thresholds, hysteresis, cooldown, level state,
   snapshot dict). ~150 lines.
2. `adaptive_command_level` + `adaptive_push_level` curriculum terms using
   the controller; `staged_on_metric` for penalties.
3. Env knobs: `ADAPTIVE_COMMANDS=1`, `ADAPTIVE_PUSHES=1`,
   `PENALTY_GATE=competence|time` (time = current warm-up, kept as
   control), threshold overrides. Template entries + config-audit test.
4. Unit tests (synthetic competence streams): promotes after cooldown,
   demotes immediately, hysteresis band holds level, freeze-not-reverse for
   penalties, snapshot round-trip, and the audit that levels map to the
   intended ranges.
5. Wire into the v16e-winner config behind the knobs; leave time-scheduled
   path intact as the A/B control.

## Experiment: v20 A/B

All cells: v16e winner base, ENTROPY_DECAY=1, 2000 it, then winner ×2 seeds.

| Cell | Config |
|---|---|
| v20-control | time-scheduled (v16e as-is) |
| v20-cmd | adaptive commands only |
| v20-full | adaptive commands + pushes + penalty gating |
| v20-hard | v20-full with L_max at the old hard endpoints (cmd L5, push 2.0) |

**Primary metric — time-to-competence:** iterations until (normalized
track_err ≤ 0.3 AND fell_ema ≤ 0.5) *at command level ≥3*. Secondary:
final fixed-eval gate (doc 12 recalibrated), level-vs-iteration curves,
and for v20-hard: does it reach L5 stably at all (v11–v15 never did — this
single cell retires the entire hard_continue question if it works).
**Expected outcome honestly stated:** modest speedup on the base task
(maybe 20–40% fewer iterations to gate), large gain on v20-hard (where
time schedules always failed), and better *reliability* (no scheduled
collapses) — the reliability is worth more than the raw speed.

## Risks / gotchas

- **Signal gaming:** normalize tracking error by command magnitude (above);
  exclude standing envs from `track_err`.
- **Oscillation:** hysteresis + cooldown are load-bearing; test them.
- **Non-stationarity for PPO:** one bounded level step per cooldown keeps
  the MDP drift small; the adaptive LR (capped) tolerates it.
- **Comparability:** with adaptive difficulty, training reward is even less
  comparable across runs — fixed eval (E0.1) and level-vs-iteration curves
  are the only valid comparisons. Log both prominently.
- **Interaction with entropy decay:** if difficulty is still rising when
  entropy reaches its floor, late levels train with little exploration —
  watch `Policy/mean_std` vs `Curriculum/*_level`; if levels stall late,
  consider holding entropy at 0.003 until level saturation.
- **Seed variance in level trajectories:** compare time-to-competence
  across ≥2 seeds before believing speedups.
