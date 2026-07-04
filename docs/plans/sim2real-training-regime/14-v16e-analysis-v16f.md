# v16e analysis (2026-07-04): entropy fixed, a second disease exposed — v16f spec

v16e (R13 stack + ENTROPY_DECAY) killed at iter ~1685/1630 (2k/4k) with
degradation underway. **The entropy fix worked** (std 0.097 and falling in
the 2k run — no regrowth) **and the run degraded anyway**: two diseases were
superposed all along; decay removed one and exposed the other.

## Evidence (2k run `0hlgni3s`)

| iter | fell/ep | ep_len | std | value loss | trkLin | peak_height (m) | err_x |
|---|---|---|---|---|---|---|---|
| 645 | 0.17 | 994 | 0.198 | 0.051 | 2.12 | 0.0209 | 0.166 |
| 891 | 0.38 | 987 | 0.136 | 0.068 | 2.20 | 0.0197 | 0.147 |
| 1134 | 0.71 | 947 | 0.113 | 0.114 | 2.10 | 0.0169* | — |
| 1408 | 1.75 | 893 | 0.103 | 0.113 | 1.94 | 0.0156 | 0.155 |
| 1702 | 2.46 | 849 | 0.096 | 0.122 | 1.67 | 0.0143 | 0.186 |

(*nearest sample.) Ordering of first movers:

1. **`peak_height_mean` erodes monotonically from ~iter 630** (2.1→1.4 cm,
   −31%) — before falls, before tracking loss, while std is still falling.
2. **Value loss doubles exactly at penalty warm-up completion** (~iter
   1000): the objective changed under the critic.
3. Falls follow the height loss with ~400-iter lag (clearance drops below
   trip margin), tracking follows the falls.
4. The 4k sibling (`cxrpije2`, slower entropy schedule, std still 0.158)
   degrades SLOWER (fell 1.62 vs 3.08 at the same wall) — residual
   exploration noise resists the slide. Low noise cannot escape it.

## Diagnosis

**Objective nonstationarity + suppressed exploration = one-way ratchet.**
The warm-up (iters 500–1000) moves the optimum after the gait has formed;
the policy then descends the penalty gradient by shaving swing height
(the cheapest way to cut joint_acc/torque_rate/joule spend), the stale
critic (value loss ×2) misprices the trade, and with std ~0.1 there is no
exploration to rediscover the higher-clearance basin. v16c's
tracking-shedding was the same disease in a different currency. The
overnight (doc 12) attributed everything to entropy; correct scope: entropy
noise was disease #1 (real, fixed), drift-under-nonstationary-objective is
disease #2 (now isolated).

**The run also contains the project's first probable gate pass:** at iters
750–1000, err_x 0.147 (≤0.15 ✓), air_time ~0.10 (≥0.11 recalibrated ≈✓),
fell 0.375 (≤0.5 ✓), ep_len 987. Checkpoints exist (save_interval 250:
`model_750`, `model_1000`).

## Immediate actions

1. **Evaluate `model_750` and `model_1000` of `0hlgni3s`** with
   `nugus_eval` (≥256 envs/command, GPU) and `sim2sim_eval`. If the gate
   passes, this checkpoint is the deploy/eval reference REGARDLESS of the
   v16f outcome — best-checkpoint ≠ last-checkpoint.
2. Both v16e jobs killed 2026-07-04 (~iter 1685/1630); s2 pending cell
   deleted before start.

## v16f spec — remove the nonstationarity

Principle: with the alive bonus in place (which is what actually fixed
v16b's bootstrap failure — NOT the warm-up), the warm-up may be pure
liability. Make the objective stationary end-to-end and let the alive
bonus carry early exploration through the penalty landscape.

| Cell | Config (all: R13 stack + ENTROPY_DECAY=1 + alive 0.5, 2000 it) |
|---|---|
| v16f-const | Penalties CONSTANT from iter 0 at final values (joule 1e-5, jacc 1e-5, torque_rate 1e-3, soft_landing 0.01). `PHASE_C_WARMUP=0`, `FLATTEN_PHASE_C=1`. The stationarity test. |
| v16f-const-half | Same but all four at HALF values, also constant. Insurance for bootstrap difficulty AND tests whether the penalty LEVEL (not just its schedule) drives the height ratchet. |
| v16f-floor | v16e as-is but `ENTROPY_END=0.003` (hold a noise floor). Tests the "exploration resists the slide" observation from the 4k run without touching the objective. |

**Decision rules:**
- v16f-const bootstraps (ep_len >600 by iter 500, as with alive it should)
  AND `peak_height_mean` holds ≥0.018 through iter 2000 → stationary
  objective wins; warm-up machinery retired; doc 08's warm-up guidance
  superseded (its real fix was the alive bonus).
- v16f-const fails to bootstrap → alive alone insufficient at full
  penalties → v16f-const-half becomes the base and penalties re-ramp via
  the COMPETENCE gate (doc 13 axis 3 — gated on measured stability, which
  never fires while unstable, unlike the timer).
- v16f-const bootstraps but STILL ratchets height late → the penalty level
  itself prices lift too high → compare v16f-const-half's ratchet slope;
  next levers: halve torque_rate specifically, tighten swing-reward std
  0.05→0.03 (steeper restoring gradient at the target), raise swing target.
- v16f-floor healthier than v16e at 2000 → keep an entropy floor ~0.003 in
  all configs (compose with whichever objective wins).

**Watch:** `Metrics/peak_height_mean` is the new leading indicator (~800
iters of warning before falls) — add it to the health-alarm set beside
`Policy/mean_std`.

## Standing-guidance updates

- Doc 08's "warm up movement penalties" is superseded IF v16f-const
  bootstraps: the alive bonus was the fix; the warm-up introduced disease
  #2. (Doc 08's core finding — full-penalties-from-0 WITHOUT alive causes
  suicide — remains true.)
- Doc 12's disease model is refined, not overturned: entropy decay stays
  mandatory; it is necessary but not sufficient.
- Best-checkpoint selection: batches should record which checkpoint (not
  just final) passes eval; the eval harness should sweep the last ~4
  checkpoints by default.
