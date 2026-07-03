# Track B — Curriculum & reward restructure

Target failure mode: every base→hard run (v11, v14, v15) degrades through the
hard ramp and never recovers — falls climb to 0.5–1.1/ep (clock_anneal) or
tracking collapses (clock_learned). The hard stage changes ~6 things at once
(commands, pushes, upright weight, four Phase-C penalties), so the record
cannot attribute blame. B1 diagnoses; B2/B3 are the redesigns; B4 is a
standing recommendation.

All comparisons use the fixed eval (E0.1), primarily
`falls_per_min` and per-command `lin_vel_rmse` at (0.75,0,0) vs (0.3,0,0).

Curriculum machinery reference: `_add_hard_continue_curriculum`,
`_add_phase_c_curriculum`, `_add_gait_curriculum` in
`src/mjlab/tasks/velocity/config/nugus/env_cfgs.py`; engine in
`src/mjlab/envs/mdp/curriculums.py` (`reward_curriculum` step-scheduler and
the unwired `staged_on_plateau`).

---

## B1 — Hard-stage decoupling grid (v17) — RUN THIS FIRST

**What:** five runs on the v16 base, each enabling exactly one component of
the hard stage (plus one all-components control):

| Cell | Hard component enabled at CONT_BASE_STEP | Everything else |
|---|---|---|
| v17-cmd | command widening only (x±0.75, y±0.45, yaw±0.80) | base values |
| v17-push | push scale ×2 only | base values |
| v17-upright | upright weight 0.5→0.25 (+std widening) only | base values |
| v17-phasec | Phase-C penalty ramp only (joule 1e-5, joint_acc, torque_rate, soft_landing, base_height) | base values |
| v17-all | all of the above (v14 replica on new physics) | — |

Config: clock_anneal, `CRITIC_HEIGHT_SCAN=true`, `JOULE_W=1e-5`,
`STAND_W=0.15`, `PHASE_C_FRAC=0.5`, `PHASE_ITERATIONS=2000`,
`CONT_BASE_STEP=48000`, `MAX_ITERATIONS=4000`, seed 1. (Seeds 1+2 if cluster
capacity allows — v11's seed variance was large.)

**Implementation:** refactor `_add_hard_continue_curriculum` to take an
enabled-components set from a new env var, e.g.
`HARD_COMPONENTS=commands,pushes,upright,phasec` (default: all, preserving
current behavior). Phase-C is currently a separate function on the p2/p3
schedule — for the grid, gate it behind the same flag mechanism so
"phasec off" removes those stages. Export the new var through
`gen-gridsearch.sh` AND `volcano-train-job.template.yaml` (the audit test
from Phase 0 item 5 should catch it if forgotten — that bug class has
happened before).

Add a `gen_v17` case to `scripts/k8s/gen-gridsearch.sh` emitting the five
cells, tags `batch-v17`, component name in the run name.

**Signal:**
- The cell(s) whose falls_per_min approaches v17-all's identifies the
  destabilizer.
- Expected (not assumed): v17-push and/or v17-upright are the likely
  culprits — upright-weight cuts were already implicated in v9; pushes
  double while the stabilizing reward halves.
- If NO single cell reproduces v17-all's degradation, the interaction is the
  problem → strengthens the case for B2 (don't stage at all).

**Cost:** 5–10 × 4k-iter runs. **Depends:** Phase 0 complete.

---

## B2 — Single-stage "hard-from-start" (v18)

**What:** train with final difficulty and a constant reward stack from
iteration 0. No hard_continue, no Phase-C reward rewrites. Modern recipes
(FastSAC paper; most 2025 velocity-tracking work) do not stage difficulty
via reward-weight rewrites — the termination penalty plus episode-length
growth is curriculum enough.

**Config:**
- Commands: full ranges from the start (x±0.75, y±0.45, yaw±0.80) — keep
  the existing early command-vel curriculum's first ~250-iter narrow stage
  if training fails to lift off, otherwise drop it too.
- Pushes ×2 from the start.
- Reward weights fixed at final values for the whole run: upright 0.25 …
  or 0.5 — decide from B1's v17-upright result: if the upright cut alone
  destabilizes, keep 0.5 permanently (accepting a slightly stiffer gait) and
  note that the "hard" target weights were never validated against hardware
  anyway.
- Phase-C penalties ON from the start at final values (joule 1e-5,
  joint_acc −1e-4, torque_rate −1e-3, soft_landing −0.01, base_height +0.3).
- Keep ONLY the gait anneal (foot_swing_height 0.75→0 over P1–P3 with the
  landing/air_time ramp-in) — it exists to hand over from shaping to
  self-organized gait and did not show pathology in the record.
- clock_anneal, hs-critic true, 4000 iters, seeds 1–2.

**Implementation:** new `TRAINING_REGIME=hard_from_start` branch in
`env_cfgs.py` next to the existing regime switch: sets command ranges and
push scale statically, sets final reward weights statically, skips
`_add_hard_continue_curriculum` and `_add_phase_c_curriculum`.

**Signal:** beat the best B1 cell and v17-all on fixed-eval falls_per_min
and hard-command tracking at equal iterations. Early training will look
worse (harder task from step 0) — judge at ≥3k iters, not 1k. If it can't
learn to walk at all in 2k iters, re-add the 250-iter narrow-command stage
before concluding failure.

**Cost:** 2–4 runs. **Depends:** B1 informs the upright decision but B2 can
launch in parallel if capacity allows (use upright 0.25 and 0.5 as two arms).
**Conflicts:** if B2 wins, `hard_continue`/`Phase-C` become dead code —
delete rather than letting them rot half-wired.

---

## B3 — Plateau-triggered progression (contingency)

Only if B2 loses to a staged variant: replace the step-scheduled
`reward_curriculum` with the existing, tested-but-unwired
`staged_on_plateau` (`src/mjlab/envs/mdp/curriculums.py`; tests in
`tests/test_envs_curriculums.py`) so each difficulty stage advances only when
the reward EMA flattens, decoupling stage boundaries from iteration guesses.
Wire it for whichever component B1 exonerated as "fine when ramped."
Cost: low-medium. This also removes the `PHASE_ITERATIONS`-vs-`MAX_ITERATIONS`
bookkeeping (`_coalesce_reward_curriculum_stages`) fragility class.

---

## B4 — Retire clock_learned (standing recommendation, no run needed)

Data (context doc F3): the phase-delta action collapses to nominal in every
base→hard run; the best "clock_learned" policy is functionally fixed-clock;
the variant costs an action dim, a 5-stage penalty schedule, and checkpoint
compatibility. Do not launch further clock_learned cells. Keep the code (it
is substantial and tested); revisit after a fixed-clock policy walks on
hardware, where variable cadence has a concrete payoff (push recovery,
speed-dependent step frequency). If revisited, the collapse mechanism to fix
first: the nominal-delta penalty makes deviation strictly costly while the
clock-annealed rewards make deviation useless — there is no reward channel
through which a non-nominal phase can pay for itself on flat ground.
