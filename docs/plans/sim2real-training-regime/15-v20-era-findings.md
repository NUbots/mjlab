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

### R3 — The controller-gating bug chronicle (one species, three instances)

All three were **thresholds outside the feasible range of their metric**:

1. Demote bar 1.0 on a fall RATE bounded ≤ 1.0 → demotion dead code.
2. Cooldown 50 iters ≪ EMA refresh (~200 iters at α=0.1/episode) →
   promotion cascades on stale competence (cmd L1→L5 in 240 iters).
3. Attainment floor SQUARED: perfect tracking of a 0.1 m/s command scored
   0.25 vs a 0.75 promote bar → level-0 promotion unreachable (measured
   0.17–0.25 on runs passing every other gate).

**Standing rule:** every gating threshold ships with a feasibility test —
a construction proving a plausibly-good policy crosses it (see
`test_attainment_true_fraction_at_small_commands` for the pattern).

### R4 — Gating metrics: falls are trailing, attainment+wobble lead

User-driven redesign (their observation: the policy prefers a stable stand
to risking falls, so fall-gated demotion never fires until terminal
collapse — pair 1 `full` ended ep 204/fell 28.9 after a quiet mid-run).
Current predicate, all axes unified: promote = attain > 0.75 AND wobble
< 0.10 AND fell < 0.3; demote = attain < 0.5 OR wobble > 0.25 OR fell
> 0.35. Attainment = (v·c)/|c|² on steps with |c| ≥ 0.15 (sway-immune,
sandbag-visible). Penalties additionally refuse to ramp while attainment
< 0.6.

### R5 — Open: the late-run destabilizer correlates with std < ~0.15

Pair 3 degraded late with penalties OFF (attainment bug kept the stability
gate closed), commands at L0, feet healthy — none of the previous suspects.
Every degrading run on corrected physics crossed into trouble as std sank:
full 0.138→0.106 (catastrophe), owned 0.171→0.13 (mild), v16e 0.103, the
4k run 0.08 (dead). Old physics tolerated 0.046 (v13) — the corrected
physics plausibly needs dither for robust recovery. Hypothesis under test:
`ENTROPY_END=0.001` over-sharpens; **v22-floor holds 0.004**. If v22's
late-run fell stays < 1 while std holds ~0.15+, the floor becomes a
standing default.

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
- **In flight:** v21 race (8k vs 32k total envs, milestone-crossing
  verdict), mn-smoke (2×4 validation; pass = ONE W&B run + sane Perf),
  mn-bench ×6 (512–8192/GPU: sim-overhead knee + straggler cost under the
  gang). Launch config for the 8-GPU era =
  argmin(sec/iter × iters-to-milestone), both factors measured.
  The sim knee also bounds how many GPUs one run can productively use at a
  given total batch.

## Corrections to earlier docs

- Doc 12 F3 verdict: correct for the STAGED-anneal clock_learned; does not
  transfer to the constant-tether clock_owned (R1). Doc 07's Stage 3 is
  effectively realized by clock_owned; period-as-command (Stage 2)
  composes with it (commanded period sets the nominal the delta modulates).
- Doc 11 idea 15c (multi-node): implemented; remove from backlog.
- The "16384 envs might help" suggestion (doc 09 era): wrong — the 1-GPU
  benchmark shows throughput FALLS above 8192 (139.6k → 112.7k sps), and
  the batch-size analysis argues for FEWER envs, not more.
