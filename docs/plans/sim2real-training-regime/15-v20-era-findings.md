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

Remedies still in reserve if v24 spirals anyway: lower fast bar (0.4),
soften the fall termination penalty during demote windows, landing
LR/desired_kl anneal at top level, per-|cmd| binned attainment (the
population mean dilutes high-command failure), best-checkpoint selection
(standing).

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
