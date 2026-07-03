# W&B run history — v8–v15 gridsearch evidence

Pulled 2026-07-03 from `vincenttumm-the-university-of-newcastle/mjlab`
(120 runs total; v15 still running at iter ~15k when sampled). The committed
narrative summary is `docs/experiments/2026-07-nugus-gridsearch-summary.md`;
this file adds the sampled history curves that ground findings F1–F5 in
`../00-context-and-findings.md`.

## Chronology (one line per batch)

| Batch | Change | Outcome |
|---|---|---|
| v8 | clock_learned, CURRENT_OBS A/B, 2k iters | Rewards 94/100 but phase-delta ratio ≈ 0 (degenerate); CURRENT_OBS no effect |
| v9 | strong phase penalty 1000 iters; upright 1.0→0.5; backslide (silently inactive) | clock_learned 63.5/41.5 with real phase use; clock_anneal baseline collapsed (peak 59→14.5) |
| v10 | (A) resume→hard: crashed, phase collapse. (B) fresh + critic height_scan | **hs-critic: fell_over 0** — key stability result |
| v11 | single-run base→hard, 4 seeds, 4k iters | All peak ~68–76 @ iter ~960 then degrade through ramp; high seed variance |
| v12 | phase-delta tail hold −0.2 vs −0.1 | −0.2: 67.4 (recovered); −0.1: 28.9, fell 2.75/ep (worst run) |
| v13 | tail −0.1 + **joule 3e-4→1e-5**; clock_anneal baseline | joule fix rescued −0.1: **69.0, fell 0.25 — best base→hard**; anneal baseline 55.0 stable |
| v14 | clock_anneal base→hard, **hs=false** | peak 71.8 → 33.4, fell 1.08 — worse without hs-critic |
| v15 | v14 × 20k iters, seeds 1–2 | Plateau 38–40 from ~4k to ~15k; extra compute buys nothing |

## Per-run curve samples (iter: reward / fell_over per ep / track_lin / phase-delta ratio)

trkLin ceiling ≈ 1.5–1.6 observed at best; fell_over is terminations per
episode. Sampled at ~10/25/50/75/100% of each run.

### clock_anneal

```
v9  baseline (2k):   436: 48.0/0.17/1.25   960: 58.1/0.21/1.48   1426: 3.0/0.46/1.37   1982: 14.4/0.12/1.42   ← collapse after peak
v13 baseline (2k):   436: 52.0/0.54/1.24   960: 68.6/0.04/1.57   1426: 50.2/0.00/1.56   1982: 55.0/0.08/1.52   ← stable
v14 base→hard (4k):  860: 70.2/0.04/1.49   1893: 53.4/0.04/1.46   2955: 44.5/0.46/1.35   3982: 34.6/1.00/1.25
v15 s1 (20k):        1103: 17.0/0.00/1.50  3916: 35.9/1.08/1.24   7778: 39.3/0.50/1.29   10726: 37.6/0.67/1.28   15105: 38.9/0.46/1.28
v15 s2 (20k):        1103: 27.6/0.12/1.50  3916: 35.2/0.88/1.19   7778: 38.3/0.88/1.18   10726: 38.9/0.92/1.25   15105: 39.7/1.12/1.25
```

Read: v15's tracking holds (~1.25) while falls stay ~0.5–1.1 → the hard-stage
problem for clock_anneal is *stability*, not tracking; and 11k iters of
plateau → no recovery is coming (F2).

### clock_learned (last field = phase-delta nominal ratio; 1.0≈uses freedom, 0≈locked to nominal clock)

```
v9  cur0 (2k):        960: 66.7/0.08/1.42/1.03   1982: 63.3/0.21/1.34/0.67
v10 hs-critic (2k):   960: 74.0/0.08/1.46/1.04   1426: 65.9/0.00/1.34/0.71   1982: 72.5/0.00/0.87/0.17   ← fell_over 0
v11 s1 (4k):          860: 69.5/0.04/1.41/1.03   1893: 62.4/0.12/1.19/0.45   2955: 54.0/0.25/0.94/0.23   3982: 48.6/0.50/0.68/0.04
v11 s4 (4k, best):   1893: 70.3/0.04/0.79/0.08   2955: 67.6/0.12/0.73/0.04   3982: 55.8/0.71/0.67/0.06
v12 tail−0.2 (4k):    860: 78.9/0.33/1.40/1.04   1893: 77.9/0.04/1.27/0.75   2955: 71.4/0.25/1.12/0.58   3982: 66.7/0.12/0.83/0.22
v12 tail−0.1 (4k):    860: 64.5/0.42/1.38/1.03   1893: 47.4/0.08/1.27/0.63   2955: 43.7/0.75/1.17/0.58   3982: 28.5/2.04/0.86/0.30
v13 tail−0.1+joule1e-5: 860: 81.8/0.17/1.43/1.05  1893: 78.7/0.12/1.25/0.71  2955: 69.8/0.25/0.94/0.31   3982: 70.2/0.08/0.76/0.05
```

Read: in EVERY base→hard run the ratio decays toward 0 (F3) — including the
best run (v13, ends 0.05 = fixed-clock in effect) — and for clock_learned
the late-run tracking (0.67–0.86) degrades in lockstep with it.

## Cross-batch leaderboard (summary-last `Train/mean_reward` — NOT comparable across curriculum stages; see F1)

| Run | Reward | fell/ep | Note |
|---|---:|---:|---|
| v8 clock_learned cur1 | 100.5 | — | phase-delta degenerate (ratio≈0), easy task |
| v10 clock_learned hs-critic | 72.1 | 0.00 | best stability, base task |
| v13 clock_learned tail−0.1 joule1e-5 | 69.0 | 0.08 | **best base→hard; current deploy candidate** |
| v12 clock_learned tail−0.2 | 67.4 | 0.12 | |
| v13 clock_anneal baseline | 55.0 | 0.08 | stable fixed-clock reference |
| v11 s4 | 54.8 | 0.71 | best v11 seed |
| v15 s1/s2 @15k | 37.6/40.0 | 0.5–1.1 | flatlined |
| v14 clock_anneal hard hs=false | 33.4 | 1.08 | dropped hs-critic, regressed |
| v12 tail−0.1 (joule 3e-4) | 28.9 | 2.75 | worst; joule interaction |
| v9 clock_anneal baseline | 14.5 | 0.12 | collapsed after peak 59 |

## Re-pulling the data

API key: k8s secret `mjlab-wandb` (namespace `mjlab`) — never commit it.

```python
# WANDB_API_KEY in env. Sampled-history pull pattern:
import wandb
api = wandb.Api(timeout=120)
runs = api.runs("vincenttumm-the-university-of-newcastle/mjlab",
                order="-created_at", per_page=50)
KEYS = ["Train/mean_reward", "Train/mean_episode_length",
        "Episode_Termination/fell_over",
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity"]
# NOTE: run.history(keys=...) drops rows missing ANY key — query
# "Metrics/phase_delta_nominal_ratio_mean" only for clock_learned runs,
# else you get 0 rows for clock_anneal.
for r in runs:
    h = r.history(samples=200, keys=KEYS, pandas=False)
```

Gotchas hit during this pull: `api.runs()` full pagination of 120 runs can
exceed 2 min — iterate with a cap instead of `len(runs)`; don't mix
variant-specific metric keys (above); `_step` is the training iteration.
