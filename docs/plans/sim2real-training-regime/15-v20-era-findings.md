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

### R11 — Command geometry: the box corners were hidden hard tasks

User insight (2026-07-05): (vx, vy, wz) are sampled independently, so
the box corners demand all axes at max simultaneously — at d=1.0 that is
0.87 m/s of combined translation while turning at 0.8 rad/s, beyond what
any single axis promises, sampled as often as anywhere else, plausibly a
hidden contributor to the L5 burn (every discrete jump also teleported
the corners outward). Fix: `command_geometry="ellipsoid"` constrains the
Mahalanobis radius rho = sqrt((vx/Rx)^2+(vy/Ry)^2+(wz/Rw)^2) <= 1 —
axis maxima stay reachable alone but trade off jointly (corner samples
re-placed along their ray with the uniform-in-ball radial profile; mild
diagonal oversampling, documented). The frontier estimator additionally
buckets hazards by rho, so under box sampling the corner cost is now
MEASURED (v28 will show it) before ellipsoid sampling removes it (v29).
Direction-resolved hazards (is x+w easier than y+w?) are the phase-3
upgrade: sample from the measured capability region + probe shell.

### R12 — The split governor (v28 postmortem) + per-axis attainment

v28 (arrest + ellipsoid + envelope 1.3 + cohort 0.3 + shadow-lambda) died
at iter 2143 BY WATCHDOG — the first fail-fast execution, exactly per the
user rule — and its autopsy is the cohort design paying off:

- The pushed cohort burned above the congestion bar for ~1000 iters
  (fast_fall_pushed 0.34 at iter 970 → 0.62 at 1920) while the single
  AIMD scalar consumed the BLENDED population rate (~0.15, diluted by
  the healthy 70% clean cohort). A third of every gradient batch was
  −10 poison, invisible to the one signal in charge; and the single
  scalar could not have eased pushes without easing commands anyway.
- Everything else validated: the clean cohort walked the extended
  ellipsoid envelope at fast-fall 0.03–0.07 (corners fix + roof-raise
  both proven — the vibes-based table was indeed conservative for
  combined-constrained commands); arrest mode was ferocious when it
  finally fired (d 1.0→0.04 in ~100 iters vs v27's 100+ iters to 0.28);
  the watchdog called the rot correctly.
- Recovery-time histogram: only 36% of push-attributed falls occur
  within 2 s of the push (62% @4 s, 92% @8 s). The damage tail is
  SECONDS long — cohort attribution was the only correct design; any
  horizon would have misattributed most of the tail.

Fix (661e46c): control now matches the stratified measurement. d_cmd is
governed by fast_fall_clean (bar 0.35); d_push by the EXCESS rate
(fast_fall_pushed − fast_fall_clean, bar 0.30 — healthy excess measured
0.15–0.18; v28's burn onset read 0.30 exactly when the cut was needed);
population ≥ 0.55 arrests both. Replay tests encode the v28
counterexample. Also added: per-axis (x/y) attainment logging — the
command table's axis ratios are vibes-based (user); these measure the
robot's real capability anatomy before any ellipsoid shape adaptation
(phase B: log-normalized per-axis radius multipliers with floors and an
all-axes gate, built only if the curves justify it).

v29 (running): split governor, same stack. Predictions on record —
pushes sawtooth around the excess bar (~1.4–1.6×), clean commands hold
the extended envelope, and if rot still appears with the poison stream
gone, pure cap-saturation churn is isolated (next lever: landing
anneal). Contingencies staged env-only at the same pin: v29b-push-soft
(bars 0.22/0.10), v29-landing (1800), v29-s2 (replicate).

### R29 — ROOT CAUSE: the effort_drift/restore mechanism mismatch

Every late-run collapse since v16b — the ~1500-1700 ignition invariant
across policies, curricula, schedules, learning rates, normalizers, and
controllers — was a single config bug:

- `effort_drift` (interval, 2-4 s, v16b-era leftover) multiplied
  `model.actuator_forcerange` by 0.995 per fire, unconditionally.
- `effort_limits` (per-reset restore) takes the IdealPdActuator branch
  for the NUgus DC actuator (DcMotorActuator subclasses IdealPdActuator)
  and writes `actuator.set_effort_limit()` — a DIFFERENT force-limit
  mechanism that never touches `actuator_forcerange`.
- The sim-level torque clamp therefore compounded unrestored for the
  whole run: x0.995^(t/3s) ≈ 0.67 by iter 500, ~0.30 by 1500-1700 —
  measured live by the R28 telemetry (0.802 → 0.787 → 0.668, rate
  x0.9956/iter = exactly the drift factor).

Everything the era observed now has one explanation: the "slide" was
the policy compensating for progressively melting servos (why it was
LR-independent: it tracked env drift, not optimizer dynamics); frozen
policies died faster because they could not compensate (v41b/v42a);
landings survived by stopping before the clamp crossed gait viability;
the attained frontier "ceiling" ~0.68 m/s was partly the clamp, not
the robot. The R16 freeze probe + R28 field telemetry closed it in two
runs once the question was finally forced env-side.

Chronicle of the hunt (methodological lessons, in order): task relief
(R7), optimizer steps (R14), obs normalization (R15), penalties (R17),
exposure/controllers (R19-R27) — every falsification narrowed the
space; the decisive moves were the bit-freeze discriminator and DIRECT
STATE MEASUREMENT instead of behavioral inference. Also: one false
smoking gun (zero-default DOFs polluting a mean — 0.4348 == 20/46)
caught within minutes by arithmetic sanity-checking.

Fix: effort_drift removed (959b7fb). If per-episode thermal derating
is wanted, reimplement symmetric with the restore path. v44 = the full
frontier architecture on honest physics, 4000 iters.

### Standing safety rails (all runs from v27 on)

- track_reward_watchdog: armed once Episode_Reward/track_linear_velocity
  EMA > 2.0; sustained < 1.0 for 60 iters → RuntimeError → job fails
  fast (user rule after v25-slow burned GPU-hours rotting). Calibrated:
  v25-push bounce transients must not fire it.
- Fast fall-rate reflex (0.35), σ floor STD_MIN=0.13, feet_min_sep,
  best-checkpoint harvesting.

## ERA CLOSED (2026-07-07): standing configuration and state of play

R29 ended the collapse era. v44 (honest physics, 4000 iters): attain
0.833 (x 0.863 / y 0.754), falls 0.022, frontier 1.18 m/s pinned
against the envelope cap, still improving at the end — every gait
metric a project record. v45 (open road: auto-extending envelope, R30)
runs 10,000 iters to find the true peak.

**Standing configuration** (the `gen_v45` cell is canonical):
clock_owned (PHASE_DELTA_W −0.2) · CURRICULUM_STYLE=aimd with the
frontier stack (survivor/censored conditional curves R20-R26, stress
valve R27, crash release, split cmd/push governors, adaptive t75
window R25, auto-envelope R30) · ellipsoid commands (R11) ·
push cohort 0.3 (R9) · STD_MIN 0.13 · obs-norm freeze 500 (harmless,
kept) · minimal penalties (joule weak + action_rate + soft_landing;
joint_acc and torque_rate deleted from the cocktail, R17/R35-36) ·
track watchdog (fail-fast) · 6144 envs/GPU · 8-GPU multinode.

**Champion artifacts**: v44 `model_3999` (log dir `v44-honest-*`) —
eval/sim2sim candidate; v45 final expected to supersede.

**Queue after v45**: nugus_eval + sim2sim on the champion · push-
exposure floor (v39 finding: speed can trade pushes to zero) ·
DR-widening rungs (sim2real robustness; prerequisite for the sysid
teacher, doc 16) · λ-live joule with a measured watts budget (R10;
shadow maxed its cap with gates green all of v44) · Webots gate
(assessment in session notes; route 1 = standalone ONNX controller,
1-2 days) · landing anneal retired from the critical path (was a
workaround for R29; keep as opt-in convergence polish).

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

## R34 (2026-07-08): bus-voltage base ratchet — R29 recurs, canary works

v48 (shared-bus voltage model, doc 17) flatlined from launch:
track_linear_velocity ~0.002 at iter 450,
simstate_actuator_forcerange_ratio 0.0000, clean_fell_ema 1.0 — zero
torque authority, every env falling on spawn. Same mechanism class as
R29, reintroduced by the fix's own author: `bus_voltage_step` refreshed
its per-episode forcerange base by re-reading the live field on reset,
assuming a reset-mode DR event restores it. Nothing does (R29's whole
point: NUgus effort DR goes through `actuator.set_effort_limit`), so
the "clean" base was our own previous scale — compounding ~0.85×/episode,
and early-training falls every ~2 s melt authority to zero within
minutes, self-reinforcing.

Lessons banked:

- **The unit test modeled the assumption, not reality**: its mock env
  restored forcerange on reset, green-lighting the exact ratchet. The
  rewritten test omits the restore (like the real event stack) and was
  verified RED against the old code before shipping the fix (fac1f4d).
- **The R28 telemetry caught it in one glance** — ratio 0.0000 is
  unambiguous. It is now also a monitor trigger (T8: ratio outside
  [0.5, 1.3] after iter 100; with BUS_VOLTAGE the legitimate steady
  state is ~0.8).
- **Generated manifests bake GIT_COMMIT**: applying a stale gen_ yaml
  relaunched the OLD code after the fix was pinned (fourth
  config-delivery split-brain of the era). After any re-pin,
  re-run gen-gridsearch.sh before kubectl apply.
- Any writer of a shared sim field must either own the restore path
  end-to-end or never re-read the field it writes. The bus model now
  caches its base once, from the clean first-call field, immutably.

Relaunched clean: ratio steady at 0.812 (designed sag), tracking 2.37
by iter 1750.

## R35 (2026-07-09): the cadence tether is mispriced — measured cadence law

A fixed-command cadence sweep of the v48 champion (model_9999; 24
envs/speed, 120-step settle, 180-step measure, CPU eval on the exact
training config) measured the policy's self-chosen phase rate against
the fixed 0.7 s tether (raw target 1.0, tail weight −0.2):

| cmd v (m/s) | raw | eff. period | \|raw−1\| | Froude |
|---|---|---|---|---|
| 0.2 | 0.372 | 1.880 s | 0.69 | 0.008 |
| 0.4 | 0.618 | 1.132 s | 0.46 | 0.033 |
| 0.6 | 0.649 | 1.078 s | 0.46 | 0.074 |
| 0.8 | 0.880 | 0.796 s | 0.36 | 0.132 |
| 1.0 | 1.170 | 0.598 s | 0.32 | 0.206 |
| 1.2 | 1.164 | 0.601 s | 0.28 | 0.297 |
| 1.4 | 0.942 | 0.743 s | 0.31 | 0.404 |

Findings:

- **The policy has a monotonic speed→cadence law** (least-squares
  `raw ≈ 0.22 + 0.84·v` through 1.2 m/s) and pays the tether penalty
  every step to express it. Heaviest tax at low speed: at 0.2 m/s it
  wants 1.9 s deliberate steps, |raw−1| ≈ 0.7.
- **In the 0.8–1.2 m/s band its chosen 0.6–0.8 s brackets the
  dynamic-similarity prior** (0.82 s full cycle for L = 0.495 m): the
  clock is holding back the *physically correct* cadence at off-nominal
  speeds, not restraining a degenerate one.
- The 1.4 m/s row is edge-of-envelope (frontier 1.37, Fr 0.40 near the
  walk→run boundary at Fr 0.5 ⇒ v* = 1.56 m/s) — least trustworthy.
- Running is a spring-mass (SLIP) regime, not pendulum: cadence there is
  a near-constant resonance, and we cannot verify these servos can
  produce a flight phase at all — so above the Froude boundary the right
  move is *no* target (release + measure later), not a guessed constant.

**Physical target, not a fit.** The obvious move — regress the sweep
(`raw ≈ 0.22 + 0.84·v`) and use that as the target — is circular: it
just tells the policy to keep doing what it already does, so the tether
teaches nothing. Instead use the *physical* law and let the network
override where it disagrees (it demonstrably will — it already deviates
by up to 0.5 in raw). Dynamic similarity (Alexander & Jayes 1983;
Alexander 1989): relative stride length `s/L = 2.3·Fr^0.3`, `Fr =
v²/(gL)`, so full-cycle period `= s/v = 2.3·√(L/g)·(v/√(gL))^(−0.4)` and
`raw_target = GAIT_PERIOD/period`. The only constants are `√(L/g)` (pure
physics from the measured 0.495 m hip height and known g) and Alexander's
cross-species coefficients 2.3 / 0.3 — none fit to our policy.

Physical vs measured cadence (raw):

| v | Froude period | net period | Froude raw | net raw |
|---|---|---|---|---|
| 0.2 | 1.35 s | 1.88 s | 0.52 | 0.37 |
| 0.4 | 1.02 s | 1.13 s | 0.69 | 0.62 |
| 0.6 | 0.87 s | 1.08 s | 0.81 | 0.65 |
| 0.8 | 0.78 s | 0.80 s | 0.90 | 0.88 |
| 1.0 | 0.71 s | 0.60 s | 0.99 | 1.17 |
| 1.2 | 0.66 s | 0.60 s | 1.06 | 1.16 |
| 1.4 | 0.62 s | 0.74 s | 1.13 | 0.94 |

The physical curve threads the middle of the network's behavior (dead-on
at 0.8 m/s) and crosses it near 0.9 m/s: the network walks *more
cautiously* than biomechanical-optimal at low speed (a stiff servo robot
without compliant tendons cannot cheaply coast a slow pendular gait) and
*more aggressively* at mid speed. Two independent physical checks land
where expected: the compound-pendulum leg resonance (uniform-rod model)
is a 1.15 s full cycle, squarely in the walking band, and the walk-run
boundary Fr=0.5 gives v* = 1.558 m/s — matching the taper endpoint
chosen independently.

v50 response: `target_mode="froude"` with L=0.495, tether tapered to zero
across 1.35→1.56 m/s (above which the walk law is invalid — running is
spring-mass, and we cannot assume these servos can even produce the
required flight phase), phase-delta action bounded [0, 2.5] as insurance
in the untethered band. Seed 1, otherwise the v48 stack: a pure ablation
of the cadence target. A linear ``target_mode`` (intercept+slope) remains
available for a fitted target if ever wanted.

## R36 (2026-07-10): v50 clock-death — the phase never meant footfalls

v50 (froude target, seed 1) ran 10k clean and walked at 2.08 m/s track /
0.007 fell / 0.83 forcerange — but with `phase_delta_raw_mean` ≈ 0.0001
from **iteration 250 onward**. The policy froze its clock in the first
250 iterations, never recovered, and learned a fully unclocked gait,
paying the tether (−0.13/step at the end) as a flat tax. Three causes,
in increasing order of importance:

1. **The froude floor priced clock-death at 0.02/step.** Early commands
   are slow, so the target sat at the 0.35 floor: a dead clock cost
   (0.35)²·0.2 ≈ 0.024/step vs 0.2/step under the fixed-1.0 tether that
   kept v48/v49's clocks alive. The policy bought clock-freedom cheap
   before it ever learned to use the clock, and by the time the target
   climbed to 0.89 the unclocked gait was fully sunk.
2. **`raw_min = 0.0` is a gradient-dead trap.** All negative action
   samples clamp to raw = 0 with identical outcome, so once the head's
   mean drifted negative there was no restoring signal. In v48 negative
   raw ran the clock *backwards* and paid (raw−1)² ≈ 0.58/step — a
   strong wall. The insurance clamp built the trapdoor.
3. **Nothing grounds the phase to the gait.** `policy_phase` feeds only
   the sin/cos observation and the tether; `feet_swing_height_clock`
   (0.75) nominally times the swing arc off the clock, but at NUgus's
   ~1.5 cm foot clearance a frozen clock's desired-height-0 still
   collects exp(−0.015²/0.05²) ≈ 0.96 of max. To a policy learning to
   walk from scratch, a rotating input it hasn't learned to use is
   self-generated observation noise — freezing it simplified its own
   inputs for pennies.

Consequences worth stating plainly: **the policy does not need the clock
to walk** (v50 is the ablation: 2.08 m/s unclocked, though −6% track and
2.3× fell vs v48 suggests the flat tether tax and/or lost metronome cost
something). And **all cadence-era numbers (R35 sweep included) describe
the clock, not the feet** — the sweep measured phase rate, which tracked
the froude law only because v48's policy voluntarily coupled its clock
to its gait. That coupling is optional and fragile.

v51 response (`gen_v51`): ground the clock. New
`gait_clock_contact_mismatch_cost` (`PHASE_CONTACT_W=0.3`) charges each
foot whose contact state contradicts its clock window (same windows as
`foot_swing_height`: swing = foot_phase < 0.45, offsets 0/0.5). A frozen
clock cannot satisfy it while stepping, and since the policy owns the
clock, the cheapest response is steering the phase to truthfully track
its own footfalls — which the froude tether then pins in physical units.
Plus `raw_min = 0.35` (= target floor): the clock physically cannot die,
and the boundary is no longer an attractor the grounding term can't pull
away from. Success criteria: `gait_clock_contact_match_mean` → ~1,
`phase_delta_raw_mean` tracking `phase_delta_target_mean`, capability
within noise of v48, and a *footfall* cadence (not just clock cadence)
following the froude curve in the post-hoc sweep.

## R37 (2026-07-11): v51 verdict — grounding is a transfer win the sim frontier hid; duck walk is velocity-limit-optimal and grounding amplifies it

v51 (contact-grounded froude clock) ran to 10k on both seeds. The final
in-training frontier looked like a regression vs v48 (track 2.22 → 1.63
s1 / 1.88 s2; clean_fell 0.003 → 0.033 s1 / 0.016 s2). But an eval sweep
at PRACTICAL commands (0.3-0.75 m/s fwd + lateral/yaw/diagonal/stand, 128
envs/cmd, 20 s) tells the opposite story — the frontier gap was almost
entirely 2 m/s behaviour the robot never uses on grass. At practical
speeds v51-s2 vs v48:

- forward-walk tracking is nearly matched (0.3: 0.094 vs 0.077, 0.5:
  0.116 vs 0.099, 0.75: 0.145 vs 0.134 m/s RMSE). v51's tracking penalty
  is concentrated in LATERAL (0.147 vs 0.087) and YAW (0.073 vs 0.028),
  not forward gait — a specific, targetable weakness.
- falls/min at practical speeds are near-zero for both (v48 0.000
  everywhere; v51-s2 tiny 0.02-0.05 only backward/lateral). The 5x
  frontier fall gap evaporates in the usable envelope.
- v51 WINS the two sim2real-critical metrics at every speed:
  swing_height_err 0.044 vs 0.070 (-37%, the anti-shuffle / turf
  clearance payoff) and slip_vel 0.036 vs 0.046 (-22%, real traction).

So v51-s2 is plausibly the better TRANSFER policy despite losing the sim
frontier. Grounding earned its place. Seed variance is real and s2 is the
keeper (beats s1 on essentially everything: 0.113 vs 0.142 track, 0.007
vs 0.028 fell) — s1 was the worse draw, confirming the user's instinct.

Gait-geometry probe (foot yaw relative to body heading, calibrated to the
standing pose; support-phase fractions), two open questions settled:

1. **"Is v50 running?" — almost, exactly.** v50 flight fraction (both feet
   airborne): 1.0 m/s 0%, 1.5 0.1%, 2.0 2.1%, 2.5 10.8%. At its ~2 m/s
   training frontier it is ~2% airborne (occasional flight, mostly
   fast-walk); by 2.5 it clearly bounds. The servos CAN produce flight,
   and the walk→run transition is real (not a sim artifact) — retroactive
   justification for releasing the cadence tether above Fr=0.5. v51-s2
   develops flight earlier (2.8% at 1.5 m/s) — the forced clearance lifts
   it off sooner.

2. **Duck walk is velocity-limit-optimal AND grounding amplifies it.**
   Duck angle (symmetric toe-out) grows monotonically and steeply with
   forward speed in BOTH v51 seeds (s2: 17°/33°/50° at 0.3/0.5/0.75;
   s1: 22°/30°/40°) — systematic, not seed lottery, confirming the
   velocity-limit motor-sharing theory (two orthogonal hip servos at
   w_max along the 45° diagonal → sqrt(2)*w_max foot speed). Lateral
   commands induce big duck even at 0.3 m/s (28-30°) — the ellipsoid
   second driver, confirmed. Pure yaw barely ducks (4-6°); backward
   walking ducks INWARD (-9°, geometry reverses). NEW: v48 barely ducks
   forward (small, sign-flipping ~noise; ducks only on the diagonal),
   while v51 ducks hard and consistently — the contact-grounding windows
   force a brisk single-support gait that recruits the duck geometry
   harder than v48's slower, more-double-support gait needed to. So
   grounding's clearance/slip win is paid for partly by leaning HARDER on
   the duck. If the duck is judged acceptable (honest physics), v51 is
   strictly good; if straight feet are ever required, v51 makes that
   harder and FOOT_HEADING_W (backlog 11c) becomes more necessary. And
   at high speed the duck DECREASES as flight sets in (v51-s2:
   43°→27°→16° at 1.0/2.0/2.5) — once bounding, the aerial phase removes
   the need for the yaw trick.

Also verified from the v48 checkpoint directly (no rollout): the actor
per-dim action log-std is UNIFORM ~0.13 across all 21 joints, falsifying
the "entropy parks extra noise on the head" hypothesis (backlog 11d
corrected). Head flail is uniform exploration noise on a light, unloaded,
reward-flat joint — same noise everywhere, only the head has no restoring
force.

## R38 (2026-07-12): v53 RMA verdict — specialization beats hedging; the student's only tax is cold-start falls. Plus: the mirror had been scrambling per-actuator obs since v48

v53 = v52 champion + the RMA concurrent adaptation module (encoder maps
the true DR realization dr_ratios+dr_extras (169) to z (16) conditioning
the policy; TCN estimator regresses z_hat from a 25-step obs history
concurrently with PPO, stop-grad both ways; run uod84pdx, pin 16318c9).

Training: the strongest run of the lineage on every axis — final track
2.01 (v52 1.53, +31%), clean_fell 0.0079 (v52 0.041, 5x fewer),
pushed_fell 0.088 (-33%), contact_match 0.947 held, arm 1.21 held.
Estimation loss fell 0.119 -> 0.0046 and never plateaued; capability
ROSE through the late run (track 1.84 -> 2.02 from iter 4500) — the
signature of stop-hedging-start-specializing. Duck: 8.7 deg mid-run,
~20 deg final (v52: 28) — partially returns with speed, consistent with
velocity-limit-optimality at the frontier (R37).

Eval (fixed grid, 128 envs/cmd, 20 s):

| metric          | v48   | v51s2 | v53 teacher | v53 student |
|-----------------|-------|-------|-------------|-------------|
| lin_vel_rmse    | 0.086 | 0.113 | 0.099       | 0.105       |
| ang_vel_rmse    | 0.300 | 0.342 | **0.199**   | 0.216       |
| falls/min       | 0.000 | 0.007 | 0.005       | **0.028**   |
| slip_vel        | 0.046 | 0.037 | **0.030**   | 0.031       |
| swing_height_err| 0.070 | 0.044 | 0.045       | 0.046       |

- Teacher is the best policy produced so far: ang tracking -40% vs all
  prior, best slip, near-zero falls. The information was worth exactly
  what the hedging had been costing.
- Student (the deployable path: z_hat from history only) holds tracking
  within 6-8% and slip/swing at par, but falls 6x the teacher
  (0.028/min ~ one per 35 min). Falls skew early-episode (steps 87-553)
  -> cold-start z_hat on backfilled windows, and the policy never rolled
  out under its own estimator (zhat_mix stayed 0 all run). v54 = v53 +
  the anneal tail (mix 0->1 over iters 7000-9000) is the designed fix.
- z-collapse falsified: per-dim z std 0.06-0.15 (mean 0.096) vs
  estimation residual 0.068 — the estimator recovers ~50% of the latent
  variance from 0.5 s of history. The unrecovered half is presumably the
  slow/weakly-excited params (bus sag, frictions at low load).

CONFOUND, and a finding in its own right: v53 is also the first run with
the actuator-order mirror fix. Entity actuator columns are in SPEC order,
not motor-joint order; the symmetry augmentation had been applying the
joint-order permutation to them, so every MIRROR_AUG run since v48
trained on mirrored samples whose actuator_current/servo_voltage channels
were cross-wired between servos (shoulder current read from a hip) and
whose critic dr_ratios kp/kd/effort segments were scrambled. Policies
plausibly learned to distrust current/voltage obs. v53's gains are
RMA+fix jointly; if attribution ever matters, the ablation is
mirror-fix-only (no RMA) — not queued, since both changes are
independently justified and permanent.

Ops scar tissue: the v53 first launch trained 4 iterations of plain
MLPModel because the manifest baked the OLD GIT_COMMIT — manifests
resolve the pin at GENERATION time (env > configmap.yaml > HEAD), so the
invariant is pin FIRST, generate SECOND, apply THIRD, and the boot check
must verify both the `HEAD is now at` line and `Actor Model: RmaActor`.

## R39 (2026-07-12): v54 verdict — the z-hat anneal HURT; the fall mechanism is estimator cold-start, not distribution shift

v54 = v53 + zhat_mix anneal 7000->9000 (run 1u8f1mlf). Two predictions
falsified in one run:
- Estimation loss did not drop when the encoder froze at mix=1 — it
  exploded 50x (0.0044 -> 0.225, ABOVE the predict-the-mean level of
  ~0.009) and plateaued there: once the policy consumes z_hat, the
  histories the estimator learns from are generated by a policy
  conditioned on the estimator's own output; PPO has no reason to keep
  the latent honest, the z correspondence unravels (self-referential
  co-evolution), and the late-run LR floor freezes the wreckage.
- Training under z_hat did not cut deployment falls — it raised them
  (student eval 0.045/min vs v53's 0.028; teacher-path eval of the SAME
  checkpoint: 0.024, better than its own deployment mode). If
  never-rolling-out-on-z_hat had caused v53's student falls, v54 would
  fall less. It falls more.
Conclusion: the student fall mechanism is the estimator's COLD START
(backfilled windows -> prior garbage; falls cluster early-episode), and
training under it just co-adapts the pair at a robustness cost. The
anneal is valid only where there is no information (deployment == prior
== training, identically) — which is the learned-safe-prior + gated
evidence design, backlog 15d. Both v53 and v54 students pass the
sim2sim gate (0 falls; the gate itself was resurrected this era —
servo-PD emulation, see changelog).

## R40 (2026-07-13): v55 verdict — end-to-end history wins in-envelope; the anchor was suppressing odometry; params still earn angular control

v55 = v53's exact architecture and capacity with the teacher removed
(RMA_E2E: PPO backprops through the TCN, no encoder, no regression; run
mqdbtfbv). Trent's "forget the student/teacher thing, just let it
learn" question, answered:

| candidate     | lin RMSE | ang RMSE | falls/min | sim2sim        |
|---------------|----------|----------|-----------|----------------|
| v53 teacher   | 0.099    | 0.199    | 0.005     | (undeployable) |
| v53 student   | 0.105    | 0.216    | 0.028     | 0 falls, 0.140 |
| v54 student   | 0.104    | 0.207    | 0.045     | 0 falls, 0.124 |
| v55 e2e       | 0.080    | 0.286    | 0.0023    | 0 falls, 0.110 |

- v55 is the LINEAGE CHAMPION: best lin tracking ever recorded (beats
  pre-wide-DR v48's 0.086), falls 0.0023/min (~1 per 7 h walking, 12x
  better than v53's deployable mode), clean cross-engine. Structurally
  it has no teacher/student gap to manage — train mode IS deploy mode.
- Odometry probe (ridge regression, trunk 16-dim features -> true
  base_lin_vel on shared rollouts): v55 vx R^2 0.947 vs v53 0.515.
  Trent's hypothesis confirmed: the actor has no velocity sensor, 0.5 s
  of history is a leg odometer, and that is the window's most valuable
  cargo — which the v53 anchor actively suppressed (its regression only
  rewarded DR-param content in the 16 dims).
- The anchor's residual value is ANGULAR control: v53 (0.199/0.216)
  beats v55 (0.286) there — the per-servo params in z evidently help
  rotational control. Neither stream subsumes the other.
- Synthesis (v57, backlog 15d): e2e policy path + AUXILIARY supervised
  heads — v_hat (velocity, exported as walk-coupled odometry) and a
  side z-head (params, to recover the angular edge) — supervision as
  side information, not as the channel's jailer.
- v55 also produced the toe-IN duck sighting for some command regime
  (episode-mean toeout is +39.6 OUT — the mean hides bimodality; the
  new inward-only foot_toein_deg metric and the FOOT_TOEIN_W guard came
  out of it, along with the discovery that the leg collision boxes are
  contype/conaffinity-0 ghosts: knees cannot collide in sim).

## R41 (2026-07-13): v56 verdict — the toe-in guard is free; 10x joule overshot (6x fall tax for the heat price)

v56 = v55 champion recipe + FOOT_TOEIN_W=-2.0 + JOULE_W=-3e-3 (10x; run
ycburzcy). Split verdict:
- Toe-in guard: SUCCESS and effectively free. foot_toein_deg pinned at
  1.3-1.7 deg from early training (82 deg at boot) while the outward
  duck ran untouched at ~40 deg — the one-sided design worked exactly as
  intended, and the residual charge decayed to -0.016. Keep permanently.
- 10x joule: BINDS (reward share -0.63 -> -0.48, arm speed driven
  2.3 -> 1.55 through the ramp) and produced the best cross-engine
  tracking yet (sim2sim lin 0.097, 0 falls) — the economical gait
  transfers well. But in-envelope it cost 6x the falls (eval 0.0141/min
  vs v55's 0.0023; clean_fell 0.0078 vs 0.0034; contact_match slid to
  0.889, first sub-0.93 of the lineage) at slightly worse lin tracking
  (0.089 vs 0.080). Too much stability traded for heat.
- v55 RETAINS the hardware crown. v56b (launched) bisects the dose:
  guard kept, JOULE_W=-1e-3 (~3x). Target: falls < 0.005 with the term
  still visibly binding.
- Field observation (Trent, from v56 rollout video): the 10x joule
  visibly fixed arm QUALITY, not just speed — much less wild movement,
  and the arms settle into a relaxed at-the-sides posture instead of
  the outstretched-forward hold every prior run converged to. Mechanism:
  arms held out front carry a constant gravity moment = constant servo
  current = constant I^2 R bleed; at 1x that static cost was priced at
  noise level, so the policy never cared. This is the strongest argument
  that the right joule dose is BINDING-but-gentler, not a revert — the
  bisect is looking for the weight that keeps the relaxed arms without
  v56's 6x fall tax. (Also relevant for transfer: outstretched arms are
  a collision/handling liability on a real pitch, and MX-64s held at
  stall torque cook.)

## R42 (2026-07-14): v56c verdict — the guard is free (better than free); the fall tax was the joule at ANY binding dose. NEW CHAMPION.

v56c (guard-only, joule 1x): eval lin 0.0795 (best ever), ang 0.313,
falls 0.0000 across 1280 episodes. Attribution closed: v56 (10x) and
v56b (3x) both eval at 0.0141 falls despite v56b's v55-class training
falls — binding joule trades eval-push stability at any dose tried; the
toe-in guard costs nothing (possibly prunes unstable footwork). Trade
menu now explicit: v56c (max stability, functional flail) vs v56b
(quiet arms, ang 0.236, 0.014 falls); backlog 11e (arm envelope,
insurance-premium pricing) is the instrument for quiet arms without the
energy tax. Also: spawn drop-in identified (Trent) as the flight-metric
floor — with the settle-gate fix, flight under the grounding tax is
true zero; v57 (flight exemption, single variable) measures whether the
tax was the walk->run barrier.

## R43 (2026-07-15): v57 verdict — the flight exemption is free and strictly dominant; running needs DEMAND, not permission. CERTIFIED CHAMPION.

v57 = v56c recipe + PHASE_CONTACT_FLIGHT_EXEMPT=1 (Trent's state-gated
design: the stance-window-airborne charge is waived only when ALL feet
are airborne, so single-foot skipping still pays and pronking pays on
landing; batch-v57). Full 10k clean.

| candidate | lin RMSE | ang RMSE | falls/min | sim2sim         |
|-----------|----------|----------|-----------|-----------------|
| v56c      | 0.0795   | 0.313    | 0.0000    | 0 falls, 0.1035 |
| v57       | 0.0761   | 0.276    | 0.0000    | 0 falls, 0.0923 |

- v57 strictly dominates v56c on every eval axis (lin, ang, slip 0.028,
  swing_height_err 0.065) at identical zero falls, and improves
  cross-engine tracking too. The exemption is not merely free — removing
  a mispriced charge (taxing correct flight as a contact error) bought
  tracking. NEW CHAMPION, and CERTIFIED: sim2sim gate passed with 0
  falls, full 30 s episodes, lin 0.0923. Deploy artifact: the v57 ONNX
  (batch-v57 run, model_9999).
- The flight question is ANSWERED: with the tax removed and the
  spawn-settle gate correcting the metric, flight_frac reads 0.0011 and
  flight_frac_fast 0.0000 — the policy does not fly even when flight is
  legal. The grounding tax was never the walk->run barrier; the command
  envelope is. Running needs DEMAND (commands past the ~1.56 m/s Froude
  boundary with the cadence tether opened), not permission. Filed as
  its own future experiment; do not spend more reward-shaping effort on
  flight.
- Attribution note: single-variable vs v56c, so the whole delta is the
  exemption's. Mechanism is likely second-order — the charge fired
  rarely (true flight was already ~0), but its GRADIENT pressed on every
  near-flight stance transition; removing it un-pinched the fast-gait
  optimum.
- Era status: v57 closes the clock_owned/RMA/e2e line opened at R31.
  Next line (v58): aux heads on the e2e trunk (15d — v_hat walk-coupled
  odometry as a second ONNX output; the side z-head angular-recovery
  question) + the 11e arm envelope (geometry priced as insurance, so
  v56c's functional flail stays legal but held reach does not).

## R44 (2026-07-16): v58 verdict — the gate closes the cutover seam completely; the angular edge was never in the params; the odometry head ships

v58 = full backlog-15d gated dual-channel design (run mxkpnmc7): e2e
z_fast + gated slow channel (1-g)*z0 + g*z_signal, learned safe prior
on the policy path, three aux losses, 17-float deployment hold, v_hat
odometry from the policy trunk, arm envelope -2.5. Full 10k clean.

| path         | lin RMSE | ang RMSE | falls/min | v_hat all/push/steady |
|--------------|----------|----------|-----------|-----------------------|
| v58 teacher  | 0.0785   | 0.282    | 0.0000    | 0.057 / 0.065 / 0.055 |
| v58 student  | 0.0864   | 0.286    | 0.0000    | 0.062 / 0.069 / 0.061 |
| (v57 champ)  | 0.0761   | 0.276    | 0.0000    | —                     |

- CUTOVER SOLVED: zero falls on the student path across 2560 episodes
  (v53: 0.028, v54: 0.045). The gate+prior design did exactly what 15d
  intended — at zero evidence the deployment recursion equals the
  training form, so there is no first-tick distribution the policy
  never saw. Teacher-student gap: lin +10%, ang ~nil. Trent's cutover
  objection is fully answered by the gate.
- THE ANGULAR EDGE WAS NEVER IN THE PARAMS: even the TEACHER path (true
  params flowing) reads ang 0.282 ~= v57's 0.276, nowhere near v53's
  0.199 — and v53 paid lin 0.099 for it. Operating-point trade, not an
  information deficit. Chasing angular via supervision anchors is a
  dead end in this architecture; remaining lever = memory (v59).
- ODOMETRY SHIPS: v_hat RMSE 0.062 m/s overall, pushes cost only +14%
  (0.061 -> 0.069) — the 15d acceptance spec (push-conditioned windows)
  passed. Walk-coupled learned odometry for the localization stack, as
  a second ONNX output, at zero cost to the walk (detached probe).
- Sim2sim: 0 falls, lin 0.108, full episodes, through the live
  17-float (z_state, evidence) loop — the stateful contract works
  cross-engine.
- Verdict: v58 is a certified deployable with the best odometry
  artifact and hand-designed memory; v57 keeps the tracking crown. The
  teacher/student machinery is retired going forward (Trent's call):
  v59 tests reward-driven memory (GRU) as the clean successor.

## R45 (2026-07-17): v59 verdict — reward-driven memory wins angular; the defined latent mirror works; two launch postmortems worth keeping

v59 = Trent's design: GRU hidden state replaces the 0.5 s window, no
teacher, no params side-loss — what to remember and how long to hold it
learned purely from reward via truncated BPTT (run qplxfzo0, pin
2e4427e, full 10k clean).

| candidate | lin RMSE | ang RMSE | falls | sim2sim (lin, falls) |
|-----------|----------|----------|-------|----------------------|
| v57       | 0.0761   | 0.276    | 0.0000| 0.092, 0             |
| v58 stud. | 0.0864   | 0.286    | 0.0000| 0.108, 0             |
| v59 GRU   | 0.0835   | 0.2502   | 0.0000| 0.174, 0             |

- MEMORY WINS ANGULAR: 0.250 is the best DEPLOYABLE angular ever
  recorded (v53's 0.199 was the undeployable teacher; v56b's 0.236
  cost 0.014 falls). -9.4% vs v57 at zero falls everywhere, confirming
  R44's conclusion: observation time, not supervision, was the angular
  lever. Odometry rides along (v_hat 0.065, push +14%). Deployment is
  the simplest yet: obs + h in, actions + velocity + h_out back, no
  ring buffer.
- THE DEFINED LATENT MIRROR WORKS (Trent's insight: a learned latent
  has no canonical mirror, so impose one — swap hidden halves — and
  training conforms): as a supervised equivariance LOSS (coeff 0.5) it
  converged to 0.0105 with no gait asymmetry (toe-in 0.83 deg). The
  twin-RNN hard variant (backlog 15e) stays in reserve.
- Two launch postmortems, both generalizable:
  1. (launch 1) Recurrent updates peak far above feedforward at equal
     envs: BPTT holds every timestep's activations, augmentation
     doubles it, and a detached aux pass must run its trunk under
     no_grad or it builds a throwaway BPTT graph. Warp shares the GPU:
     watch the SIM's headroom.
  2. (launch 2) Mirror DATA AUGMENTATION is quietly incompatible with
     recurrent PPO: the KL check's batch-dim slice hits the time dim on
     trajectory batches, so mirrored samples (off-policy until
     equivariance is learned) enter the KL and pin the adaptive LR at
     its floor — self-locking. Trent diagnosed it from the videos
     ("at worst it should ignore the RNN"); the fix pair is the obs
     skip connection (memory additive by construction) + mirror as a
     loss. Batch-mode BPTT recomputation was verified bit-identical to
     rollout replay before relaunch.
- The cross-engine lin question, ANSWERED by the frozen-h ablation
  (15c's design, run same-day): zeroing the hidden state every tick in
  vanilla MuJoCo collapses the policy — 2.52 falls/min, 14 s mean
  episodes, lin 0.213, ang 0.615 — versus ZERO falls, full episodes,
  0.174/0.442 with the memory running. The memory is load-bearing
  cross-engine: it identifies the foreign dynamics and compensates
  (adaptation transferring, not engine overfit). The 0.174 lin is the
  gait the memory chooses under a physics it was never trained on,
  while keeping the robot up; the strongest adaptation-transfer
  evidence in the program. Remaining hardware call is judgment: v57
  transfers a better raw gait (0.092) with no adaptation machinery;
  v59 transfers a live adapter with the angular record. Field both:
  v57 as the safe baseline, v59 as the instrumented candidate (its
  velocity output doubles as the odometry probe).

## Corrections to earlier docs

- Doc 12 F3 verdict: correct for the STAGED-anneal clock_learned; does not
  transfer to the constant-tether clock_owned (R1). Doc 07's Stage 3 is
  effectively realized by clock_owned; period-as-command (Stage 2)
  composes with it (commanded period sets the nominal the delta modulates).
- Doc 11 idea 15c (multi-node): implemented; remove from backlog.
- The "16384 envs might help" suggestion (doc 09 era): wrong — the 1-GPU
  benchmark shows throughput FALLS above 8192 (139.6k → 112.7k sps), and
  the batch-size analysis argues for FEWER envs, not more.
