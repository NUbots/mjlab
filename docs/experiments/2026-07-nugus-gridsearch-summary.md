# Nugus Grid Search Summary (v8–v15)

Generated 2026-07-03 from Weights & Biases runs in [`vincenttumm-the-university-of-newcastle/mjlab`](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab).

Window: last 7 days (since 2026-06-26 UTC). Task: `Mjlab-Velocity-Flat-Nubots-Nugus`, 8192 envs/GPU, experiment group `nugus_gridsearch_v8` … `nugus_gridsearch_v15`.

## Overview

- **22 runs** across batches v8–v15 (17 finished, 2 running, 3 crashed)
- Metrics below are final summary values from each W&B run (last logged step).
- Lower velocity tracking error and swing clock error are better; higher mean reward and episode length indicate more stable training.

## Cross-batch comparison

Best finished run per batch by mean reward:

| Batch | Best run | State | Reward | Ep len | Vel XY err | Vel yaw err | Swing clk err |
|-------|----------|-------|--------|--------|------------|-------------|---------------|
| v8 | `clock_learned__stand-0.15__pc-0.5__cur1__s1__v8` | ✓ | 100.48 | 997 | 0.529 | 0.365 | 0.0023 |
| v9 | `clock_learned__stand-0.15__pc-0.5__cur0__s1__v9` | ✓ | 63.50 | 982 | 0.318 | 0.440 | 0.0190 |
| v10 | `clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10` | ✓ | 73.73 | 1000 | 0.559 | 0.551 | 0.0111 |
| v11 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s4__v11` | ✓ | 54.76 | 943 | 0.643 | 0.622 | 0.0058 |
| v12 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.2__s1__v12` | ✓ | 67.40 | 992 | 0.598 | 0.559 | 0.0132 |
| v13 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__joule-1e-5__s1__v13` | ✓ | 68.98 | 987 | 0.632 | 0.613 | 0.0058 |
| v14 | `clock_anneal__stand-0.15__pc-0.5__base-hard__s1__v14` | ✓ | 33.43 | 913 | 0.318 | 0.521 | 0.0457 |
| v15 | `clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s2__v15` | ⏳ | 38.99 | 955 | 0.346 | 0.491 | 0.0459 |

## Key findings

1. **v8 (clock_learned, cur0 vs cur1)** — Both seeds finished strongly. `cur1` reached the highest flat-reward in this window (100.5) with the best yaw tracking (0.365). Phase delta nominal ratio stayed near zero, indicating tight cadence lock during learned-clock training.
2. **v9 (strong phase_delta + upright)** — `clock_learned cur0` (63.5 reward, vel_xy 0.318) outperformed `cur1` (41.5) and the `clock_anneal` baseline (14.5). Anneal runs show higher swing clock error (~0.046) and air-time metrics absent from learned-clock runs.
3. **v10 (hard cont vs hs-critic retrain)** — Height-scan critic retrain runs finished well (72–74 reward, vel_xy ~0.56). Hard-continuation runs were unstable: 3/6 crashed; finished hard-cont runs had poor velocity tracking (vel_xy up to 0.94).
4. **v11 (base→hard overnight, 4 seeds)** — All finished. Seed 4 best reward (54.8); seed 3 weakest (33.0, shortest episodes at 890). Velocity tracking moderate (vel_xy 0.45–0.66).
5. **v12 (phase_delta tail -0.2 vs -0.1)** — Tail -0.2 clearly better on reward (67.4 vs 28.9) and episode length (992 vs 867). Velocity tracking slightly favors -0.1 (vel_xy 0.464 vs 0.598), but overall training stability favors the stronger tail.
6. **v13 (joule 1e-5 learned vs anneal baseline)** — Low-joule learned run best overall in v11–v13 range: 69.0 reward, vel_xy 0.632. Anneal baseline again lower reward (55.0) but excellent velocity tracking (vel_xy 0.250).
7. **v14 (clock_anneal base→hard, legacy critic)** — Finished at 33.4 reward; velocity tracking good (vel_xy 0.318) but episodes shorter (913) than learned-clock hard runs.
8. **v15 (20k clock_anneal base→hard)** — Both seeds still running. Current snapshots: s1 reward 35.7, s2 reward 39.0; vel_xy ~0.32–0.35. Long-horizon training in progress.

## Batch v8

clock_learned CURRENT_OBS 0 vs 1 at pinned commit (strong early phase_delta_nominal). 2000 iters.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_learned__stand-0.15__pc-0.5__cur0__s1__v8](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/57gr0muv) | ✓ | 94.22 | 992 | 0.438 | 0.244 | 0.554 | 0.474 | 0.0021 | 7.36e-05 | — |
| [clock_learned__stand-0.15__pc-0.5__cur1__s1__v8](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/v21878z8) | ✓ | 100.48 | 997 | 0.418 | 0.236 | 0.529 | 0.365 | 0.0023 | -1.24e-04 | — |

## Batch v9

Extended strong phase_delta stage (1000 iters), upright_w=0.5. clock_learned cur0/cur1 vs clock_anneal baseline. 2000 iters.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_anneal__stand-0.15__pc-0.5__s1__v9](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/xm5t9ilu) | ✓ | 14.50 | 999 | 0.167 | 0.197 | 0.285 | 0.443 | 0.0466 | — | 0.097 |
| [clock_learned__stand-0.15__pc-0.5__cur0__s1__v9](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ift9sd2w) | ✓ | 63.50 | 982 | 0.191 | 0.216 | 0.318 | 0.440 | 0.0190 | 0.6644 | — |
| [clock_learned__stand-0.15__pc-0.5__cur1__s1__v9](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ffcdmlkz) | ✓ | 41.51 | 985 | 0.212 | 0.220 | 0.336 | 0.432 | 0.0200 | 0.6086 | — |

## Batch v10

Hard continuation from v9 cur0 (legacy critic) vs flat retrain with critic height_scan. 2000 iters. (3 crashed reruns also logged.)

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/cxl0l9d8) | ✓ | 49.55 | 965 | 0.767 | 0.401 | 0.938 | 0.790 | 0.0054 | 0.0420 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/1tdz3rjv) | ✓ | 73.73 | 1000 | 0.449 | 0.246 | 0.559 | 0.551 | 0.0111 | 0.2031 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ukz6aprc) | ✗ | 58.73 | 989 | 0.330 | 0.256 | 0.457 | 0.598 | 0.0137 | 0.3758 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/h8hszyrf) | ✗ | 25.21 | 909 | 0.311 | 0.294 | 0.475 | 2.149 | 0.0160 | 0.9941 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/lhlbam8a) | ✗ | 52.98 | 1000 | 0.768 | 0.395 | 0.937 | 0.844 | 0.0034 | 0.0013 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/yz5baxda) | ✓ | 72.07 | 1000 | 0.445 | 0.263 | 0.564 | 0.582 | 0.0095 | 0.1667 | — |

## Batch v11

Overnight base→hard single-run, 4000 iters, critic height_scan, seeds 1–4.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s1__v11](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/16cbg6lm) | ✓ | 47.31 | 989 | 0.551 | 0.266 | 0.661 | 0.696 | 0.0059 | 0.0446 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s2__v11](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/u5mbohzy) | ✓ | 38.29 | 939 | 0.335 | 0.237 | 0.450 | 0.552 | 0.0120 | 0.3136 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s3__v11](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/4jx3q9es) | ✓ | 33.02 | 890 | 0.373 | 0.246 | 0.494 | 0.538 | 0.0107 | 0.2466 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s4__v11](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/px06ulu0) | ✓ | 54.76 | 943 | 0.539 | 0.251 | 0.643 | 0.622 | 0.0058 | 0.0553 | — |

## Batch v12

v11-style base→hard with phase_delta tail weight -0.2 vs -0.1. 4000 iters, seed 1.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.2__s1__v12](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/lyhwmnll) | ✓ | 67.40 | 992 | 0.482 | 0.270 | 0.598 | 0.559 | 0.0132 | 0.2179 | — |
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__s1__v12](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/260z9ekp) | ✓ | 28.92 | 867 | 0.360 | 0.227 | 0.464 | 0.537 | 0.0140 | 0.2976 | — |

## Batch v13

Lower joule (1e-5) clock_learned base→hard (pd-tail -0.1) vs clock_anneal flat baseline. 4000/2000 iters.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__joule-1e-5__s1__v13](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/l9wok1ss) | ✓ | 68.98 | 987 | 0.525 | 0.264 | 0.632 | 0.613 | 0.0058 | 0.0542 | — |
| [clock_anneal__stand-0.15__pc-0.5__s1__v13](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ojozkbfs) | ✓ | 54.99 | 985 | 0.140 | 0.178 | 0.250 | 0.401 | 0.0455 | — | 0.091 |

## Batch v14

clock_anneal base→hard single-run, legacy critic, 4000 iters, seed 1.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_anneal__stand-0.15__pc-0.5__base-hard__s1__v14](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/jyksw3mg) | ✓ | 33.43 | 913 | 0.183 | 0.222 | 0.318 | 0.521 | 0.0457 | — | 0.099 |

## Batch v15

v14 extended to 20k iters, seeds 1–2. Still running at time of report.

| Run | State | Reward | Ep len | Vel X | Vel Y | Vel XY | Vel yaw | Swing clk | Phase δ ratio | Air time |
|-----|-------|--------|--------|-------|-------|--------|---------|-----------|---------------|----------|
| [clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s1__v15](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ynquy630) | ⏳ | 35.66 | 924 | 0.185 | 0.218 | 0.319 | 0.484 | 0.0455 | — | 0.135 |
| [clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s2__v15](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/rntq7onj) | ⏳ | 38.99 | 955 | 0.219 | 0.222 | 0.346 | 0.491 | 0.0459 | — | 0.132 |

## Secondary metrics (best finished run per batch)

| Batch | Base height | Foot tilt | Gait air CV | Landing force | Slip vel | Angular mom |
|-------|-------------|-----------|-------------|---------------|----------|-------------|
| v8 | 0.454 | 0.019 | 0.360 | 21.2 | 0.028 | 0.099 |
| v9 | 0.461 | 0.019 | 0.508 | 16.0 | 0.032 | 0.102 |
| v10 | 0.457 | 0.024 | 0.460 | 22.7 | 0.014 | 0.051 |
| v11 | 0.458 | 0.070 | 0.304 | 43.7 | 0.011 | 0.046 |
| v12 | 0.458 | 0.029 | 0.374 | 24.2 | 0.014 | 0.062 |
| v13 | 0.458 | 0.064 | 0.337 | 33.8 | 0.010 | 0.048 |
| v14 | 0.467 | 0.023 | 0.462 | 20.1 | 0.037 | 0.108 |
| v15 | 0.466 | 0.046 | 0.475 | 21.6 | 0.031 | 0.108 |

## Notes

- v10 includes duplicate crashed reruns of the same configuration (hard-cont and hs-critic); only the latest finished run per config should be used for comparison.
- v15 metrics are in-progress snapshots; re-fetch after runs complete for final numbers.
- Experiment configs are defined in `scripts/k8s/gen-gridsearch.sh` (`gen_v8_grid` … `gen_v15_clock_anneal_hard_long`).
