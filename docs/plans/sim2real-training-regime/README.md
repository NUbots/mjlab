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
- [ ] v16b re-baseline launched (clock_persist, GAIT_PERIOD 1.0, swing 0.05, flattened Phase-C, trimmed DR — spec in doc 06)
- [ ] Velocity limits corrected: XH540 no-load is 39–46 rpm = 4.1–4.8 rad/s (Phase 0 used 3.2 from a wrong 30 rpm figure); set from measured bus voltage; check `saturation_effort` = stall torque (doc 06 §V5). Units are rad/s — confirmed, not rpm
- [ ] Classical-walk joint-velocity/torque log captured from the real robot (ground truth for limits + gait envelope)
- [ ] Stage 2 gait-period-as-command (v19) — variable step timing, doc 07
- [ ] `Metrics/vel_sat_frac_*` saturation telemetry added
- [ ] B1 decoupling grid re-launched on v16b base (first v17 attempt **killed 2026-07-03** — ran on collapsed v16 base; **not analyzed** via fixed eval)
- [ ] B1 analyzed via fixed eval → destabilizer identified (**v17 attempt N/A — killed before eval**)
- [ ] B2 hard-from-start launched (v18)
- [ ] A1 BAM friction params adopted + DR re-centered
- [ ] **Deferred (post-D1):** A4 latency measurement, A2 bench replay, D2 hardware deployment — start after v17/v18 winner passes the D1 sim-to-sim gate
- [x] D1 Sim-to-sim ONNX gate script (`scripts/sim2sim_eval.py`; not yet run on a checkpoint)
- [x] C2 Mirror augmentation (`mirror_map.py`, `MIRROR_AUG=1`; `tests/test_mirror_map.py`)
- [x] C3 Entropy decay / LR cap / γ=0.97 cells (`NugusOnPolicyRunner`; `ENTROPY_DECAY`, `LR_CAP`, `GAMMA` env vars)
- [x] Stopped the flatlined v15 runs — deleted `mj-gs-v15-ca-base-hard-20k{,-s2}` vcjobs 2026-07-03 (~iter 16200/20000)

## Standing recommendations (not tasks)

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
| `references/codebase-map.md` | file:line map of the NUgus task, rewards, DR, curriculum, k8s harness |
| `references/wandb-run-history.md` | v8–v15 leaderboard, per-run curves, re-pull script |
| `references/bam-actuator-models.md` | Rhoban BAM: extended friction for MX-64/MX-106 |
| `references/systematic-sim2real-pace.md` | Identification-over-randomization study; delay/friction/armature guidance |
| `references/cts-teacher-student.md` | CTS concurrent teacher-student (deferred track C4) |
| `references/fastsac-15min-recipe.md` | Minimal-reward off-policy recipe; γ and symmetry ablations |
| `references/small-humanoid-precedents.md` | OP3 soccer, FRASA, Bez thesis (failure case) |
| `references/reward-curriculum-symmetry.md` | Periodic reward composition, delay randomization, symmetry papers, torque injection |
