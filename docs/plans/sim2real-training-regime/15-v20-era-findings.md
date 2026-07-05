# v20-era findings (2026-07-04, operator session)

Chronicle of the babysat batch day: v20 pairs 1–3, the controller bug
chronicle, the clock_owned promotion, and the multi-GPU program. Run IDs in
W&B carry tags `batch-v20/21/22`, `mn-smoke`, `mn-bench`.

## Headline results

### R1 — clock_owned promoted to base variant (F3 retirement overturned)

Pair 3 ran `full` (clock_persist, fixed clock) vs `owned` (policy-owned
phase, constant −0.2 quadratic tether) on otherwise identical configs.
Under the same late-run destabilizer:

| iter | full (fixed) | owned |
|---|---|---|
| 1437 | ep 950, fell 0.71 | ep 928, fell 0.58 |
| 1643 | ep 405, fell 8.25 | ep 755, fell 2.17 |
| 1979 | ep 46, fell 148 (destroyed) | ep 336, fell 15.8 (degraded, walking) |

~10× resilience gap, and `Metrics/phase_delta_dev_p95` (built for exactly
this question) shows why: p95 cadence deviation stayed 1.0–1.25 the whole
run — the constant tether preserves real phase freedom (the old STAGED
penalty collapsed usage to ~0, which is what doc 12's F3 verdict measured)
— and usage PEAKED during the stress window (p95 1.25, mean dev 0.55 at
iter 1437). The escape hatch fires when it matters; there is no
realignment debt to an external metronome (user's design argument,
confirmed). All future bases: `MJLAB_VARIANT=clock_owned`,
`PHASE_DELTA_W=-0.2` (weaken later to allow more adaptation; strengthen to
recover the fixed clock as the limiting case).

### R2 — Feet min-separation fix validated end-to-end

Every run since the fix: lateral spacing 0.18–0.28 m,
`feet_min_sep_violation_mean` ~1e-6, at every command level reached.
Calibration final: min 0.13 m center-to-center (~5 cm edge gap free),
sharpness 20 (steep wall through the 3→1 cm hazard band). Gate metric:
`min_foot_lateral_distance_mean ≥ 0.13`; monitor trigger < 0.115.

### R3 — The controller-gating bug chronicle (one species, four instances)

All four were **thresholds outside the feasible range of their metric**:

1. Demote bar 1.0 on a fall RATE bounded ≤ 1.0 → demotion dead code.
2. Cooldown 50 iters ≪ EMA refresh (~200 iters at α=0.1/episode) →
   promotion cascades on stale competence (cmd L1→L5 in 240 iters).
3. Attainment floor SQUARED: perfect tracking of a 0.1 m/s command scored
   0.25 vs a 0.75 promote bar → level-0 promotion unreachable (measured
   0.17–0.25 on runs passing every other gate).
4. Promote bar 0.75 above the feasible ceiling: with uniform command
   sampling, population-mean attainment for a GOOD policy tops out ~0.71
   (small-command steps dilute the mean). Healthy runs plateaued 0.55–0.63
   and never promoted. Fixed: promote 0.60 / demote 0.40 (generator env
   `COMPETENCE_PROMOTE_ATTAIN` / `COMPETENCE_DEMOTE_ATTAIN`).
5. The 0.60 bar is MARGINAL, not safe: the L0 attainment ceiling varies
   run-to-run with the realized gait (v22b/v23/v24 crossed 0.60 at iters
   1050–1300; v24b converged at 0.543 and NEVER promoted — then died of
   R6 saturation churn while healthy in every other respect). Corollary
   to the standing rule: bars must clear the WORST-CASE observed ceiling
   with margin, not the typical one. v24c: promote 0.50 / demote 0.35.

**Standing rule:** every gating threshold ships with a feasibility test —
a construction proving a plausibly-good policy crosses it (see
`test_attainment_true_fraction_at_small_commands` for the pattern).

### R4 — Gating metrics: falls are trailing, attainment+wobble lead

User-driven redesign (their observation: the policy prefers a stable stand
to risking falls, so fall-gated demotion never fires until terminal
collapse — pair 1 `full` ended ep 204/fell 28.9 after a quiet mid-run).
Current predicate, all axes unified: promote = attain > 0.60 AND wobble
< 0.10 AND fell < 0.3; demote = attain < 0.40 OR wobble > 0.25 OR fell
> 0.35 (bars corrected per R3 instance 4). Attainment = (v·c)/|c|² on steps with |c| ≥ 0.15 (sway-immune,
sandbag-visible). Penalties additionally refuse to ramp while attainment
< 0.6.

### R5 — Resolved: σ level is an amplifier; the v22 two-arm verdict

v22-floor (ENTROPY_END=0.004) completed all 2000 iters — first
corrected-physics run to finish without catastrophe. σ equilibrated ~0.22
throughout; end state ep 689 / fell 1.38 vs pair-3 owned (no floor) ep
336 / fell 15.8 and v20-full ep 46 / fell 148. clock_owned (~10×) and the
σ floor (~10×) stack to ~100× late-run failure reduction. v22b-stdmin
(hard `STD_MIN=0.13` via `std_range`, entropy schedule unchanged) also
completed with σ pinned 0.143–0.146: both mechanisms work; the hard clamp
is the tighter guarantee and is the standing default (`STD_MIN=0.13`).

BUT: v22-floor still *drifted* late (fell 0.29→1.38 over the final 500
iters, trkLin reward 2.2→1.6-1.9, value loss 0.07→0.17, stance widening
0.20→0.28) with σ flat — σ-collapse was the amplifier, not the driver.

### R6 — The driver: policy churn under a saturated objective (hypothesis,
strong support)

Every corrected-physics collapse happened in a run whose curriculum was
FROZEN at level 0 by an R3 gating bug. The frozen task saturates by ~iter
1000 (v22-floor: trkLin 2.07 by 796, reward plateau 85–94); PPO then keeps
taking 20 constant-KL updates/iter (surrogate stays ~0.004–0.005 — the
adaptive-KL schedule maintains step SIZE, not step VALUE) on advantages
that are now noise → random walk in policy space → drift out of the good
basin, with falls injecting −10 spikes that accelerate the exit. Onset
~1400–1600 in every config because saturation time was identical.
Explains: v13/old-physics stability (±0.5 commands never saturated), late
value-loss rise (critic chasing a wandering policy), σ's amplifier role.

Discriminating evidence: **v22b climbed cmd L0→L5 (full envelope) + push
L2 — first full ladder in project history** — with no flat-then-wander
phase; its sibling v22-floor, frozen at L0 by the old bars, drifted. The
edge-of-competence curriculum is therefore a STABILITY mechanism, not
just a speed feature: difficulty tracking competence keeps the objective
unsaturated and the gradient informative.

v24b (2026-07-05) is the pure-form demonstration: stuck at L0 by bar
instance 5, it was an EXCELLENT policy at iter 1200 (ep 1000, wobble
0.011, fell_ema 0.09) and then, with nothing left to learn, decayed on
schedule — attain slipping from 1232, fast fall rate 0.02→0.67, entropy
VALUE rising −1.15→+0.70 (the mid-decay entropy coefficient of the
4000-iter schedule actively pushed std 0.24→0.26 through the churn
window), ending in the same absorbing fall-spiral at ~2100. Saturation
alone kills, on a ~900-iteration fuse, with no hard task in sight.

**Unified law of the era: too-hard (brink burn, R7) and too-easy
(saturation churn, R6) both terminate in the same absorbing fall-spiral.
The only healthy corridor is a ladder that keeps moving — every bar
reliably below its level's worst-case ceiling, the top rung capped below
the burn zone, and the fast reflex trimming excursions.**

Open sub-question — ANSWERED by v23 (2026-07-05): v22b's late L5 sag was
the front edge of a structural failure, not mid-learning. v23 (8-GPU,
6144/GPU, 4000 iters, same stack) climbed L0→L5+push2 by iter 1815
(healthy plateau: ep 880, fell 1.83, full ±0.75 envelope — checkpoint
`model_1750` in log dir `v23-prod-20260704-222010`, best policy artifact
to date), then spiraled: ep 880→325 and fell 1.83→10.7 in ~280 iters.

### R7 — The brink spiral and the demote-latency hole (v23 postmortem)

Mechanism: the per-env competence EMAs (α=0.1 per EPISODE END) are nearly
static while episodes are long — at ep~900 each env reports once per ~9
iters, so a crash at the top of the ladder is invisible to the controller
for ~200 iters. The demote cascade DID fire (commands walked 5→0,
penalties retreated to zero — the machinery worked as designed), but by
then the −10 fall terminations had dominated the gradient for 200+ iters
and the policy was shattered: at L0, attain read −0.68 (moving AGAINST
commands), fell_ema 1.0, ep 23 steps, with NO recovery in 1200 further
iters. The fall-gradient spiral is absorbing once entered; σ floor did
not help (std pinned 0.13 throughout — this is not an exploration
failure). Run killed at 3328.

Fix shipped (commit a3433bc, v24): fast windowed fall-rate channel —
population falls/episode-ends refreshed every iteration with α=0.2 EMA;
demote fires at rate > 0.5 (healthy L5 measured ~0.26), reacting in ~5
iters and cascading L5→L0 in 5 more. Plus top-rung promote caution
(streak 5 for L≥4). Feasibility tests per the R3 standing rule: crash
demotes on the fast channel alone with stale-healthy slow EMAs; healthy
L5 band (0.30) does not trip the 0.5 bar.

v24 verdict (2026-07-05): the reflex fired EXACTLY at its 0.5 bar (iter
~2020, measured rate 0.51) and cascaded L5→L0 in ~40 iters — but the
spiral outran it: falls kept exploding at L0 (5.2→13.6 by iter 2137,
ep 253). Two lessons: (a) the point of no return sits BELOW 0.5 — during
the L5 burn (rate 0.3–0.45, under the bar) the −10 gradients were
already poisoning the batch, and once poisoned, easing the task does not
stop the bleeding; (b) all three L5 visits to date (v22b, v23, v24)
ignited the burn at ±0.75 — consistent with the XH540 velocity-margin
analysis that ±0.75 m/s sits at the actuator's physical edge. v24b:
fast bar 0.35 (healthy-L5 band measured 0.26) + ladder capped at L4
(±0.6) until a healthy full-length L4 run exists.

v24b/v24c verdicts (2026-07-05, overnight): v24b never promoted (bar
instance 5) and died of pure saturation churn at 2115. v24c — bar 0.50,
lmax 4 — proved the corridor: L0→L4 by iter 1157, **attainment record
0.701 at iter 1424**, ~800 healthy iterations at L4. Then the law, a
third time: saturation at the capped top rung lit the fuse at ~1424
(attain slide, fast rate 0.20→0.33), the fast demote fired CORRECTLY at
1959, cascaded to L0 — and the spiral completed straight through the
cascade (fell 21.6, ep 255 at 2133). Demotion is NOT an antidote to
churn damage: once the gait is broken, the −10s continue at any
difficulty. Five deaths at iter 2100±40 = saturation time (~1200-1500,
whatever rung) + the ~700-900-iter churn fuse; nothing is scheduled
there (PHASE 2000 vs 4000 made no difference).

**Era conclusion: the current ladder holds ~1400-1500 iterations of
genuine curriculum content. Running past the top rung's mastery point is
what kills — every time.** v24d (the landing run) stops at 1600, inside
the attainment peak, and its final checkpoint is the era's artifact.

Structural exits for the next code cycle (in order):
1. Landing anneal: at top level + reward plateau, decay LR/desired_kl to
   ~0 — turn the plateau into convergence instead of churn fuel.
2. More rungs: ADR-style continuous DR widening (friction, mass, IMU
   tilt, incline — doc 11 items 12/13), push escalation beyond L2, and
   terrain — so the ladder keeps moving for the full 4000 iterations
   and the policy banks robustness instead of churning.
3. Checkpoint rollback on crash (restore last healthy model; the only
   cure once a spiral starts — demotion demonstrably is not).
4. Per-|cmd| binned attainment (population mean dilutes high-command
   failure; would also make L5 gate honest before re-attempting ±0.75).

### R8 — AIMD continuous difficulty (the TCP turn, v26/v27)

User insight (2026-07-05): the level ladder's jumps are visible in every
trace, and TCP congestion control solves formally the same problem —
probe an unknown, drifting capacity from a binary distress signal. The
control law (commit 633d5f6, wart-fixed 1b4e71c): one difficulty scalar
d lerps command ranges across the full L0→L5 envelope and push magnitude
0.75×→2.0×; additive increase 0.002/iter (overshoot inside the ~11-iter
detection lag ≈ 0.01 m/s), multiplicative cut 0.7 at fast-fall 0.35
(proven recoverable twice, v25-push) and 0.5 at 0.55 (proven fatal,
v24), ssthresh high-water slow-probe, refractory doubling as backoff at
persistent walls. Chiu-Jain: AIMD converges to a stable sawtooth around
capacity; the sawtooth doubles as intrinsic anti-churn (the objective
never fully freezes). Structural win: increase gates modulate the RATE,
so the R3 bug species (infeasible bar silently blocks a jump) cannot
recur — an infeasible gate freezes d visibly. Wart found live on v26:
early-chaos "cuts" at d≈0 pinned ssthresh to its floor and the climb ran
at probe rate; fixed (ignore congestion at d<0.05) + regression test.

### R9 — Push-cohort stratification + frontier estimator (v27)

User design: fixed env-index cohorts — 30% pushed, 70% push-free — so
competence attribution is by membership, with no recovery-horizon guess.
Clean cohort = uncontaminated tracking/attainment signal (sandbagging
detector) AND deployment-matched distribution (matches are mostly
push-free); pushed cohort = push competence, with difference-in-rates
(fall_rate_pushed − fall_rate_clean) isolating the push effect including
delayed falls (both cohorts share the command sampler, so command
difficulty cancels). Plus: clean-cohort falls bucketed by commanded
speed (binned Bernoulli, ~1k samples/bucket per 50-iter window → ±3%)
give P(fail | speed) and a live frontier_speed estimate — the measured
version of what AIMD discovers by collision; and a push-to-fall timing
histogram answers "how long does recovery take" empirically. Phase 1 is
log-only on v27; the estimator earns control (fall-rate setpoint solved
against the measured curve) only if frontier_speed cross-validates
against the AIMD sawtooth's settling point.

### R10 — Penalties are Lagrangian, not AIMD (shadow pilot ready)

Taxonomy settled with the user: AIMD is for binary-catastrophic
feedback (falls) probing unknown capacity; penalties have a continuous
measurable cost and physically meaningful budgets, so they are
constrained-RL multipliers (RCPO / PID-Lagrangian). Inverted per the
user's framing — minimize energy SUBJECT TO competence: λ_joule rises
(additive, full ramp ≥1000 iters for timescale separation — the
disease-#2 lesson) only while swing peak height ≥ 85% of a trailing max
(foot-lift collapse gate: v16c, v25-slow), clean attainment ≥ 0.50
(sandbagging gate), clean fast-fall < 0.20 (shared governor); retreats
×0.8 when peak height breaks 70%. The two historical joule failure
modes ARE the gates, so the live controller cannot reproduce them
silently. Shadow (log-only) implementation shipped (5986850) with
feasibility replays: v25-slow's slide must trigger retreat ahead of the
T4 monitor bar; a healthy plateau must climb at η. Rides along on v28;
flips live only after one clean shadow run.

### Standing safety rails (all runs from v27 on)

- track_reward_watchdog: armed once Episode_Reward/track_linear_velocity
  EMA > 2.0; sustained < 1.0 for 60 iters → RuntimeError → job fails
  fast (user rule after v25-slow burned GPU-hours rotting). Calibrated:
  v25-push bounce transients must not fire it.
- Fast fall-rate reflex (0.35), σ floor STD_MIN=0.13, feet_min_sep,
  best-checkpoint harvesting.

## Multi-GPU program (user-driven, latency-first)

Reframe: while the lineage is still being debugged, single-run latency
beats A/B throughput. Built in-session:

- **torchrun env-rendezvous multi-node path** (no SSH, no image change):
  train.py passes through when RANK/WORLD_SIZE set; MJLAB_LOG_STAMP keeps
  pods on one log dir; Volcano svc/env plugins provide MASTER_ADDR and
  node rank; template parametrizes replicas (gang-scheduled). Network is a
  non-issue: ~2 MB grads × 20 all-reduces ≈ 80 ms/iter at 10G vs 2.5 s
  iterations; 400G unnecessary at this model size.
- **Pod shape rule:** one pod per node, all local GPUs per pod (2×4).
  Finer shards (4×2, 8×1) silently downgrade same-node NCCL to TCP
  (separate containers share no IPC namespace) — only justified for GPU
  scavenging on shared clusters.
- **Batch-size theory applied** (user: env count ≈ batch size — correct):
  786k samples/update for a 0.5M-param policy is plausibly far past
  critical batch; iteration-denominated machinery (competence cooldowns,
  entropy decay, warm-ups) makes wall-time ∝ sec/iter matter doubly.
  First measured point: 4×2048 runs 1.21 s/iter vs ~2.8 at 4×8192 —
  2.3× iteration rate at 58% fps.
- **v21 race verdict: sample-limited.** At equal wall time, 8192/GPU hit
  attain 0.625 at iter 748 vs 2048/GPU's 0.488 at iter 1932. Samples/sec
  is the binding resource; iteration rate is secondary.
- **The entrypoint ConfigMap split-brain.** First "8-GPU" wave was fake:
  pods run `entrypoint.sh` from the kustomize-generated `mjlab-entrypoint`
  ConfigMap, which still held the pre-multinode script (`kubectl apply -f
  configmap.yaml` does NOT update it — needs `apply -k`). No rendezvous →
  each pod of the gang silently trained an independent 4-GPU job with the
  same run name → 2× duplicate W&B runs per job, fps identical to 4-GPU
  baselines (the tell). Even post-fix, a pod starting <60 s after the
  apply still mounted the stale file (kubelet propagation). Third
  config-delivery split-brain of the era (git pin / generator env /
  ConfigMap): launch preflight should hash repo entrypoint vs cluster
  ConfigMap. Fake duplicate runs deleted from W&B.
- **mn-smoke + mn-bench (real, verified single-run, ranks 0–7 across both
  nodes):** scaling is 2.02× at 8192/GPU (563k vs 278k fps) — 10G
  Ethernet cost invisible, as predicted. Curve (envs/GPU → s/iter, total
  fps): 512→0.85/116k, 1024→1.03/191k, 2048→1.32/299k, 4096→1.70/463k,
  6144→2.20/536k, 8192→2.92/539k. **fps saturates at 6144**, which
  strictly dominates 8192 (same fps, 25% cheaper iterations). With the
  v21 sample-limited verdict, production config = **6144/GPU (49,152
  total), ~2.2 s/iter, ~536k fps**; fallback 4096 (the elbow) if a run
  ever looks iteration/cooldown-limited rather than sample-limited.
- **v23 (in flight):** first production 8-GPU run — clock_owned,
  STD_MIN 0.13, corrected 0.60/0.40 bars, feet fix, competence-gated
  penalties/commands/pushes, 6144/GPU, 4000 iters (~2.5 h). Primary
  question: does v22b's late L5 sag consolidate (mid-learning) or persist
  (structural — see R6 remedies).

## Corrections to earlier docs

- Doc 12 F3 verdict: correct for the STAGED-anneal clock_learned; does not
  transfer to the constant-tether clock_owned (R1). Doc 07's Stage 3 is
  effectively realized by clock_owned; period-as-command (Stage 2)
  composes with it (commanded period sets the nominal the delta modulates).
- Doc 11 idea 15c (multi-node): implemented; remove from backlog.
- The "16384 envs might help" suggestion (doc 09 era): wrong — the 1-GPU
  benchmark shows throughput FALLS above 8192 (139.6k → 112.7k sps), and
  the batch-size analysis argues for FEWER envs, not more.
