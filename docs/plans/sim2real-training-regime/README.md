# NUgus sim-to-real training regime — runbook

Plan authored 2026-07-03 from: repo state at commit `adf2023` (branch
`add-phase-clock`), the v8–v15 W&B gridsearch history (project
`vincenttumm-the-university-of-newcastle/mjlab`), and a literature review of
2024–2026 sim-to-real bipedal locomotion RL. All source material is distilled
into `references/` — **do not re-research; look it up there first.**

## Goal

Produce a walking policy that transfers to the physical NUgus (NUbots RoboCup
kid-size humanoid, 20× Dynamixel servos). No policy has been deployed to
hardware yet. The two highest-risk gaps are actuator-model fidelity (the sim
currently has **no servo speed saturation and no gearbox friction**) and the
destabilizing base→hard curriculum (every v11–v15 run degrades through it).

## How to use this directory

1. Read `00-context-and-findings.md` first. It contains the evidence base:
   what exists in the code, what the W&B history shows, what has already been
   tried and rejected (**do not re-propose those**), and the gap analysis.
2. Execute phases in order below. Each numbered doc is a self-contained work
   package: motivation → exact changes (with file paths) → verification →
   launch instructions → success/failure signals.
3. `references/codebase-map.md` is the file:line index of every config,
   reward term, DR event, and script mentioned. Line numbers were correct at
   `adf2023`; re-grep if they've drifted.
4. Record outcomes of each launched batch in
   `docs/experiments/2026-07-nugus-gridsearch-summary.md` (the existing
   convention), and tick the checklist below.

## Execution order and dependencies

```
Phase 0  01-foundations-v16-base.md   REQUIRED FIRST — changes physics; every
         (E0.1 eval, E0.2 servo fixes,  later comparison depends on it
          A3 mass DR, C1 critic DR obs,
          no-op audit)
            |
            v
Phase 1  03-track-b-curriculum.md     B1 decoupling grid (v17) — the key
         (launch first)                diagnostic; then B2 hard-from-start (v18)
            |
Phase 1' 02-track-a-actuator-fidelity.md  A1 BAM params can land with Phase 0;
         (parallel, mostly robot-side)     A2/A4 need bench/robot time
            |
Phase 1'' 05-track-d-hardware-loop.md  D2 deployment plumbing + logging NOW,
         (parallel, no GPU cost)        in parallel with everything
            |
            v
Phase 2  04-track-c-policy.md         C2 mirror augmentation, C3 optimizer
                                       hygiene — after a stable v17/v18 base
```

If only one thing can be done: Phase 0, then launch the B1 grid.

## Checklist

- [x] E0.1 Fixed evaluation harness (`scripts/nugus_eval.py`)
- [x] E0.2a Servo velocity saturation (DcMotor-style torque-speed clamp)
- [x] E0.2b Nonzero joint frictionloss baselines (un-break `joint_friction` DR)
- [x] E0.2c Remove/implement dead `velocity_limit=30` TODOs
- [x] A3 Link-mass scale + torso payload DR
- [x] C1 Critic observes sampled DR parameters
- [x] No-op audit test (DR scale terms have nonzero baselines; active rewards fire)
- [ ] v16 smoke batch: confirm training still converges on new physics (manifests ready; **blocked on commit+push** — Phase 0 code is local-only)
- [x] ~~v16 base validated~~ **v16 COLLAPSED at p3 — see `06-v16-collapse-analysis.md`**: swing target infeasible on new actuators + clock_anneal→0 handoff; joule/DR largely acquitted
- [x] ~~v16b re-baseline~~ **v16b FAILED — never stood (ep_len ~25 for 1400 iters, both seeds); jobs killed. See `08-v16b-postmortem.md`**: un-specced effort-limit cut to rated torque + Phase-C penalties from step 0 with no alive bonus = learned fast-termination
- [x] v16c launched @ `547c67d` (2026-07-03) — **WALKS** (alive bonus fixed bootstrap) but shuffles/limps and tracking declines post-warm-up; jacc −3e-5 cell clearly better than −1e-4. Full analysis in `09-v16c-analysis-v16d.md`
- [x] v16d wave-0 **Completed** (2026-07-03) — D0 BASE1=`v16d-main`; keep `JOINT_ACC_W=−1e-5`, keep `MIRROR_AUG`; doc 09 gate FAIL (no eval yet)
- [x] Wave-1 R4–R5 **Completed**; R6–R7 **Running**; R8–R9 pending (doc 10)
- [x] peak-swing-height metric added to clock_persist telemetry (`Metrics/peak_height_mean` on `feet_swing_height_clock`, `94dae96`)
- [x] Velocity limits corrected: XH540 no-load is 39–46 rpm = 4.1–4.8 rad/s (Phase 0 used 3.2 from a wrong 30 rpm figure); set from measured bus voltage; check `saturation_effort` = stall torque (doc 06 §V5). Units are rad/s — confirmed, not rpm
- [ ] Classical-walk joint-velocity/torque log captured from the real robot (ground truth for limits + gait envelope)
- [ ] Stage 2 gait-period-as-command (v19) — variable step timing, doc 07
- [x] `Metrics/vel_sat_frac_*` saturation telemetry added
- [ ] B1 decoupling grid re-launched on **v16c** base (first v17 attempt **killed 2026-07-03** — ran on collapsed v16 base; **not analyzed** via fixed eval)
- [ ] B1 analyzed via fixed eval → destabilizer identified (**v17 attempt N/A — killed before eval**)
- [ ] B2 hard-from-start launched (v18)
- [ ] A1 BAM friction params adopted + DR re-centered
- [ ] **Deferred (post-D1):** A4 latency measurement, A2 bench replay, D2 hardware deployment — start after v17/v18 winner passes the D1 sim-to-sim gate
- [x] D1 Sim-to-sim ONNX gate script (`scripts/sim2sim_eval.py`; not yet run on a checkpoint)
- [x] C2 Mirror augmentation (`mirror_map.py`, `MIRROR_AUG=1`; `tests/test_mirror_map.py`)
- [x] C3 Entropy decay / LR cap / γ=0.97 cells (`NugusOnPolicyRunner`; `ENTROPY_DECAY`, `LR_CAP`, `GAMMA` env vars)
- [x] Overnight plan doc 10 + idea backlog doc 11 added; wave 1–4 manifests @ `ce4da03` (config-audit green; **push + launch pending approval**)
- [x] Overnight waves ran 2026-07-03/04 — wave 1 short-run results valid; **wave-4 R18/R19 collapsed late (entropy, doc 12); R20 rough OOM**. FINAL remains **R13 BASE3**. R18 `nugus_eval` on collapsed `model_3750`: **FAIL** (falls 68/min vs v13 1.1/min). Post-mortem: `12-overnight-postmortem.md`
- [x] v16e ran + killed mid-run 2026-07-04 — **entropy fix WORKED (std 0.097, no regrowth) but a second disease was exposed: objective-nonstationarity drift** (penalty warm-up moves the optimum → policy ratchets swing height 2.1→1.4 cm → trips → falls; value loss ×2 at warm-up end; suppressed exploration can't escape). Analysis + v16f spec: `14-v16e-analysis-v16f.md`
- [ ] **`0hlgni3s` `model_750`/`model_1000` evaluated (nugus_eval ≥256 envs/cmd + sim2sim) — probable FIRST GATE PASS** (err_x 0.147, air ~0.10, fell 0.375 at iter 891); best-checkpoint ≠ last-checkpoint
- [ ] v16f launched: const / const-half / entropy-floor cells (stationary objective test — doc 14 decision rules)
- [ ] Voided experiments re-run on entropy-fixed base: wide DR, self_paced, heel-toe long (check per-corner clearance ÷4 rescale first)
- [x] Competence-based curriculum implemented + **reviewed/fixed 2026-07-04** (doc 13 addendum): 2 launch-blocking bugs fixed (torch.where shape crash on standing-env resets; kwargs TypeError in staged_on_competence), pessimistic EMA init, **penalty gate now demotes on instability** (disease-#2 countermeasure), `const` stationary-objective cell added (v16f folded in). 20 tests green
- [x] **Feet min-separation fix landed** (`6163169`): `feet_min_sep` one-sided cost (min 0.18 m, sharpness 12, weight −1.0) — the v16e gate-pass checkpoint walked at 0.14–0.15 m lateral spacing and stood on its own foot outside Warp (symmetric feet_distance too gentle near zero, measures center-Y only). New gate metric: `min_foot_lateral_distance_mean ≥ 0.16`; `Metrics/feet_min_sep_violation_mean` should → 0
- [x] v20 pair 1 launched 2026-07-04 @ `6163169` — **v20-const KILLED at iter 553 (T2: ep_len 26, never stood)**: constant FULL-value penalties from iter 0 block bootstrap even with the alive bonus — doc 14's optimistic const branch is RESOLVED NEGATIVE (doc 08's suicide finding was penalty-level-dependent, not just alive-dependent). Competence gating (v20-full) is now the sole penalty mechanism under test; const-half only if full ratchets late. **v20-full healthy** (iter 330: ep 991, fell 0.04, levels progressing); `v20-cmd` launched in the freed slot. Script monitor active (triggers T1–T7, feet ≥0.16 / peak_height ≥0.016 gates)

## Standing recommendations (not tasks)

- **`ENTROPY_DECAY=1` in every future run** (`ENTROPY_START=0.01`,
  `ENTROPY_END=0.001`). The overnight collapse disease (doc 12): fixed
  entropy 0.01 regrows policy std after convergence (~iter 1000) and
  destabilizes every long run. Monitor `Policy/mean_std`; alarm if it rises
  >30% off its post-convergence minimum.
- **Baseline guard:** a new base must beat the previous batch's best at the
  SAME iteration count and comparable health — never promote a config whose
  fell_over is a multiple of the incumbent's.

- **Retire `clock_learned` for now** (B4). Every run shows the learned phase
  delta collapsing back to the nominal clock; the best "clock_learned" run
  (v13) is functionally a fixed-clock policy. Rationale and data:
  `00-context-and-findings.md` §F3. The variable-step-timing goal it served
  is NOT retired — the supported path is `07-gait-timing-strategy.md`
  (gait period as a command, push-window phase freedom).
- **Keep `CRITIC_HEIGHT_SCAN=true`** in all future batches (v10 evidence:
  fell_over → 0; v14/v15 dropped it and regressed).
- **Keep `JOULE_W=1e-5`**, not 3e-4 (v13 evidence; independently matches the
  systematic-sim2real study's recommendation, see references).
- Reward-curve values are NOT comparable across curriculum stages or between
  pre- and post-Phase-0 physics. Use the fixed eval (E0.1) for all comparisons.

## Directory contents

| File | Contents |
|---|---|
| `00-context-and-findings.md` | Current setup, W&B findings F1–F5, tried-and-rejected list, gap analysis |
| `01-foundations-v16-base.md` | Phase 0: eval harness, servo model fixes, mass DR, critic DR obs, audits |
| `02-track-a-actuator-fidelity.md` | BAM friction models, bench ID, latency measurement, torque-noise injection |
| `03-track-b-curriculum.md` | B1 decoupling grid, B2 hard-from-start, B3 plateau curriculum, B4 clock_learned retirement |
| `04-track-c-policy.md` | Mirror augmentation, optimizer hygiene, deferred CTS/history |
| `05-track-d-hardware-loop.md` | Sim-to-sim gate, first deployment protocol, logging spec |
| `06-v16-collapse-analysis.md` | v16 post-mortem (infeasible swing target, anneal handoff), v16b re-baseline spec, public config sources |
| `07-gait-timing-strategy.md` | Variable step timing done right: period-as-command (Stage 2), push-window phase freedom, unclocked endgame |
| `08-v16b-postmortem.md` | v16b never-stands failure (effort cut + step-0 penalties, no alive bonus), v16c spec, pre-launch smoke-test pattern. Supersedes doc 06 item 5 and §V5 effort wording |
| `09-v16c-analysis-v16d.md` | v16c walks-but-shuffles analysis (joint_acc kills tracking, no symmetry = limp, drag is incentive-limited), v16d change-set + launch checklist + success thresholds |
| `10-overnight-20run-plan.md` | Autonomous 10 h / 20-run adaptive plan: heel-toe reward fixes (per-corner clearance, center swing height, one-sided foot_flat), cadence/stride grid, economy ablations, seed variance, DR/push restoration, self_paced stretch |
| `11-idea-backlog.md` | Ranked backlog: adaptive command/push curricula (ADR-lite), single-support & stride rewards, IMU/incline DR, base-height & lean tweaks, AMP-from-classical-walk; ⭐ items are overnight-backfill eligible |
| `12-overnight-postmortem.md` | Overnight collapse root cause (entropy-driven std regrowth; ENTROPY_DECAY was off), corrected hypothesis verdicts, recalibrated gate, v16e spec, new process rules |
| `13-competence-curriculum.md` | Edge-of-competence curriculum: CompetenceController design (hysteresis/cooldown/levels), adaptive command/push/penalty axes, v20 A/B spec, replaces hard_continue |
| `14-v16e-analysis-v16f.md` | v16e: entropy fix confirmed, disease #2 isolated (nonstationary-objective height ratchet), probable first gate-pass checkpoint (`0hlgni3s` @750–1000), v16f stationarity cells + decision rules |
| `15-v20-era-findings.md` | v20-era chronicle: clock_owned promoted (10x resilience, p95 usage under stress), feet fix validated, controller bug chronicle + threshold-feasibility rule, attainment/wobble gating, std<0.15 destabilizer hypothesis (v22), multi-GPU program + batch-size economics |
| `16-auto-sysid-teacher-student.md` | automatic sysid plan: privileged teacher (dr_ratios actor) -> recurrent history student via rsl-rl distillation, adaptation-probe validation, route tradeoffs (action distillation vs explicit-z RMA) |
| `references/codebase-map.md` | file:line map of the NUgus task, rewards, DR, curriculum, k8s harness |
| `references/metrics-glossary.md` | every logged W&B key explained with units and measured healthy/alarm bands (AIMD, cohort diagnostics, hazards/frontier, watchdog, shadow-lambda, core rsl-rl gotchas) + 30-second run-reading checklist |
| `references/wandb-run-history.md` | v8–v15 leaderboard, per-run curves, re-pull script |
| `references/bam-actuator-models.md` | Rhoban BAM: extended friction for MX-64/MX-106 |
| `references/systematic-sim2real-pace.md` | Identification-over-randomization study; delay/friction/armature guidance |
| `references/cts-teacher-student.md` | CTS concurrent teacher-student (deferred track C4) |
| `references/fastsac-15min-recipe.md` | Minimal-reward off-policy recipe; γ and symmetry ablations |
| `references/small-humanoid-precedents.md` | OP3 soccer, FRASA, Bez thesis (failure case) |
| `references/reward-curriculum-symmetry.md` | Periodic reward composition, delay randomization, symmetry papers, torque injection |
