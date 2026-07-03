# Nugus grid-search experiment summary (v8–v15)

W&B project [vincenttumm-the-university-of-newcastle/mjlab](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab). Metrics fetched 2026-07-03T00:39:59Z from run summaries and history.

Batch definitions live in `scripts/k8s/gen-gridsearch.sh`. When multiple attempts exist for the same `run_name`, this report keeps the latest W&B run by `created_at`.

## Overview

| Batch | Runs | State | Best `Train/mean_reward` | `Train/mean_episode_length` (last) | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| v8 | 2 | finished | 100.48 | 997.29 | clock_learned CURRENT_OBS 0 vs 1 at pinned 2229f92 (strong early phase_delta_nom… |
| v9 | 3 | finished | 63.50 | 981.91 | Extend strong phase_delta_nominal to 1000 iters (w=-5), halve upright to 0.5, pr… |
| v10 | 2 | crashed, finished | 72.07 | 1000 | Two parallel jobs: (A) flat hard continuation from v9 cur0 with legacy critic (r… |
| v11 | 4 | finished | 54.76 | 943.24 | Overnight base→hard single-run (no resume): 4000 iters, v9-equivalent base for 2… |
| v12 | 2 | finished | 67.40 | 992.09 | v11-like base→hard with non-zero phase_delta_nominal tail: compare PHASE_DELTA_T… |
| v13 | 2 | finished | 68.98 | 987.42 | Lower joule (1e-5) on v12-like clock_learned base→hard (pd-tail -0.1) plus v9-eq… |
| v14 | 1 | finished | 33.43 | 913.16 | clock_anneal base→hard single-run (no resume), v11-style 4000 iters with legacy … |
| v15 | 2 | running | 39.97 | 955.63 | v14 clock_anneal base→hard extended to 20k iters; hard ramp in first ~3k iters t… |

## Batch intentions

### v8

clock_learned CURRENT_OBS 0 vs 1 at pinned 2229f92 (strong early phase_delta_nominal). pc-0.5, STAND_W 0.15, seed 1, 2000 iters, PHASE_ITERATIONS=2000.

### v9

Extend strong phase_delta_nominal to 1000 iters (w=-5), halve upright to 0.5, progress_backslide -0.5. Three runs: clock_learned cur0/cur1 and clock_anneal baseline.

### v10

Two parallel jobs: (A) flat hard continuation from v9 cur0 with legacy critic (resume), (B) flat v9-equivalent retrain with critic height_scan.

### v11

Overnight base→hard single-run (no resume): 4000 iters, v9-equivalent base for 2000 iters then hard_continue from step 48000. clock_learned, hs-critic, seeds 1–4.

### v12

v11-like base→hard with non-zero phase_delta_nominal tail: compare PHASE_DELTA_TAIL_W -0.2 vs -0.1 (seed 1).

### v13

Lower joule (1e-5) on v12-like clock_learned base→hard (pd-tail -0.1) plus v9-equivalent clock_anneal flat baseline at JOULE_W=3e-4.

### v14

clock_anneal base→hard single-run (no resume), v11-style 4000 iters with legacy critic (no height_scan), seed 1.

### v15

v14 clock_anneal base→hard extended to 20k iters; hard ramp in first ~3k iters then hold. seeds 1–2.

## Per-run metrics

### Batch v8

#### `clock_learned__stand-0.15__pc-0.5__cur0__s1__v8`

- **W&B:** [57gr0muv](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/57gr0muv)
- **State:** finished
- **Created:** 2026-07-01T05:14:37
- **Tags:** batch-v8, clock_learned, current-0, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 94.22 |
| `Train/mean_episode_length` | 991.77 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.000 |
| `Metrics/phase_delta_mean` | 0.000 |
| `Metrics/phase_delta_nominal_error_mean` | 1.065 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.997 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.017 |
| `Episode_Reward/upright` | 0.954 |
| `Episode_Reward/stand_still_pose` | -0.007 |
| `Episode_Termination/fell_over` | 0.083 |
| `Episode_Termination/time_out` | 8.125 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=92.74 (step 1995)
- mean_reward max=95.40 (step 1954)
- mean_episode_length last=979.38 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.034 |
| `Episode_Reward/action_rate_l2` | -0.016 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.240 |
| `Episode_Reward/body_ang_vel` | -0.003 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.043 |
| `Episode_Reward/feet_distance` | -0.018 |
| `Episode_Reward/foot_clearance` | -0.009 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.008 |
| `Episode_Reward/foot_swing_height` | 1.200 |
| `Episode_Reward/gait_phase_regularity` | -0.059 |
| `Episode_Reward/joint_acc_l2` | -0.836 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.771 |
| `Episode_Reward/soft_landing` | -0.005 |
| `Episode_Reward/stand_still_motion` | -0.000 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.011 |
| `Episode_Reward/track_angular_velocity` | 1.633 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur1__s1__v8`

- **W&B:** [v21878z8](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/v21878z8)
- **State:** finished
- **Created:** 2026-07-01T05:14:46
- **Tags:** batch-v8, clock_learned, current-1, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 100.48 |
| `Train/mean_episode_length` | 997.29 |
| `Metrics/phase_delta_nominal_ratio_mean` | -0.000 |
| `Metrics/phase_delta_mean` | -0.000 |
| `Metrics/phase_delta_nominal_error_mean` | 1.056 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.098 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.017 |
| `Episode_Reward/upright` | 0.972 |
| `Episode_Reward/stand_still_pose` | -0.008 |
| `Episode_Termination/fell_over` | 0.042 |
| `Episode_Termination/time_out` | 5.667 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=100.93 (step 1995)
- mean_reward max=101.41 (step 1971)
- mean_episode_length last=995.40 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.032 |
| `Episode_Reward/action_rate_l2` | -0.015 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.240 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.019 |
| `Episode_Reward/feet_distance` | -0.015 |
| `Episode_Reward/foot_clearance` | -0.014 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.013 |
| `Episode_Reward/foot_swing_height` | 1.204 |
| `Episode_Reward/gait_phase_regularity` | -0.064 |
| `Episode_Reward/joint_acc_l2` | -0.731 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.722 |
| `Episode_Reward/soft_landing` | -0.008 |
| `Episode_Reward/stand_still_motion` | -0.000 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.011 |
| `Episode_Reward/track_angular_velocity` | 1.746 |

</details>

### Batch v9

#### `clock_anneal__stand-0.15__pc-0.5__s1__v9`

- **W&B:** [xm5t9ilu](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/xm5t9ilu)
- **State:** finished
- **Created:** 2026-07-01T07:16:11
- **Tags:** batch-v9, clock_anneal, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 14.50 |
| `Train/mean_episode_length` | 998.56 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.420 |
| `Episode_Reward/foot_swing_height_landing` | -0.052 |
| `Episode_Reward/joule_heating` | -0.072 |
| `Episode_Reward/upright` | 0.481 |
| `Episode_Reward/stand_still_pose` | -0.142 |
| `Episode_Termination/fell_over` | 0.208 |
| `Episode_Termination/time_out` | 7.792 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=14.93 (step 1995)
- mean_reward max=59.43 (step 941)
- mean_episode_length last=982.42 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.077 |
| `Episode_Reward/action_rate_l2` | -0.039 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0.026 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.236 |
| `Episode_Reward/body_ang_vel` | -0.004 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.221 |
| `Episode_Reward/feet_distance` | -0.111 |
| `Episode_Reward/foot_clearance` | -0.030 |
| `Episode_Reward/foot_flat` | -0.003 |
| `Episode_Reward/foot_slip` | -0.016 |
| `Episode_Reward/foot_swing_height` | 0 |
| `Episode_Reward/gait_phase_regularity` | -0.086 |
| `Episode_Reward/joint_acc_l2` | -2.131 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.003 |
| `Episode_Reward/soft_landing` | -0.012 |
| `Episode_Reward/stand_still_motion` | -0.002 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.022 |
| `Episode_Reward/track_angular_velocity` | 1.586 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__s1__v9`

- **W&B:** [ift9sd2w](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ift9sd2w)
- **State:** finished
- **Created:** 2026-07-01T07:26:56
- **Tags:** batch-v9, clock_learned, current-0, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, strong-1000, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 63.50 |
| `Train/mean_episode_length` | 981.91 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.664 |
| `Metrics/phase_delta_mean` | 0.019 |
| `Metrics/phase_delta_nominal_error_mean` | 0.482 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.344 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.050 |
| `Episode_Reward/upright` | 0.477 |
| `Episode_Reward/stand_still_pose` | -0.041 |
| `Episode_Termination/fell_over` | 0.250 |
| `Episode_Termination/time_out` | 8.417 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=65.03 (step 1995)
- mean_reward max=68.09 (step 995)
- mean_episode_length last=994.04 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.062 |
| `Episode_Reward/action_rate_l2` | -0.033 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.243 |
| `Episode_Reward/body_ang_vel` | -0.003 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.095 |
| `Episode_Reward/feet_distance` | -0.137 |
| `Episode_Reward/foot_clearance` | -0.026 |
| `Episode_Reward/foot_flat` | -0.004 |
| `Episode_Reward/foot_slip` | -0.013 |
| `Episode_Reward/foot_swing_height` | 1.109 |
| `Episode_Reward/gait_phase_regularity` | -0.074 |
| `Episode_Reward/joint_acc_l2` | -1.163 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.146 |
| `Episode_Reward/soft_landing` | -0.009 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.015 |
| `Episode_Reward/track_angular_velocity` | 1.597 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur1__s1__v9`

- **W&B:** [ffcdmlkz](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ffcdmlkz)
- **State:** finished
- **Created:** 2026-07-01T08:36:40
- **Tags:** batch-v9, clock_learned, current-1, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, strong-1000, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 41.51 |
| `Train/mean_episode_length` | 985.12 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.609 |
| `Metrics/phase_delta_mean` | 0.017 |
| `Metrics/phase_delta_nominal_error_mean` | 0.276 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.301 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.062 |
| `Episode_Reward/upright` | 0.471 |
| `Episode_Reward/stand_still_pose` | -0.094 |
| `Episode_Termination/fell_over` | 0.250 |
| `Episode_Termination/time_out` | 9.917 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=41.50 (step 1995)
- mean_reward max=68.95 (step 858)
- mean_episode_length last=980.29 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.089 |
| `Episode_Reward/action_rate_l2` | -0.043 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.236 |
| `Episode_Reward/body_ang_vel` | -0.003 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.238 |
| `Episode_Reward/feet_distance` | -0.124 |
| `Episode_Reward/foot_clearance` | -0.022 |
| `Episode_Reward/foot_flat` | -0.008 |
| `Episode_Reward/foot_slip` | -0.010 |
| `Episode_Reward/foot_swing_height` | 1.069 |
| `Episode_Reward/gait_phase_regularity` | -0.078 |
| `Episode_Reward/joint_acc_l2` | -1.831 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.004 |
| `Episode_Reward/soft_landing` | -0.008 |
| `Episode_Reward/stand_still_motion` | -0.002 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.024 |
| `Episode_Reward/track_angular_velocity` | 1.617 |

</details>

### Batch v10

#### `clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10`

- **W&B:** [lhlbam8a](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/lhlbam8a)
- **State:** crashed
- **Created:** 2026-07-01T11:35:35
- **Tags:** batch-v10, clock_learned, continuation, gridsearch, hard-continue, joule-3e-4, pc-0.5, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 52.98 |
| `Train/mean_episode_length` | 1000 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.001 |
| `Metrics/phase_delta_mean` | 0.000 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.522 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.046 |
| `Episode_Reward/upright` | 0.227 |
| `Episode_Reward/stand_still_pose` | -0.017 |
| `Episode_Termination/fell_over` | 0.208 |
| `Episode_Termination/time_out` | 8.250 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=52.33 (step 3819)
- mean_reward max=58.31 (step 2994)
- mean_episode_length last=970.23 (step 3819)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.088 |
| `Episode_Reward/action_rate_l2` | -0.041 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.242 |
| `Episode_Reward/body_ang_vel` | -0.001 |
| `Episode_Reward/command_progress_backslide` | -0.001 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.079 |
| `Episode_Reward/feet_distance` | -0.072 |
| `Episode_Reward/foot_clearance` | -0.003 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.003 |
| `Episode_Reward/foot_swing_height` | 1.211 |
| `Episode_Reward/gait_phase_regularity` | -0.069 |
| `Episode_Reward/joint_acc_l2` | -1.043 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.545 |
| `Episode_Reward/soft_landing` | -0.002 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.015 |
| `Episode_Reward/track_angular_velocity` | 1.358 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10`

- **W&B:** [yz5baxda](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/yz5baxda)
- **State:** finished
- **Created:** 2026-07-01T11:36:34
- **Tags:** batch-v10, clock_learned, critic-height-scan, flat-retrain, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 72.07 |
| `Train/mean_episode_length` | 1000 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.167 |
| `Metrics/phase_delta_mean` | 0.005 |
| `Metrics/phase_delta_nominal_error_mean` | 0.742 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.893 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.053 |
| `Episode_Reward/upright` | 0.473 |
| `Episode_Reward/stand_still_pose` | -0.017 |
| `Episode_Termination/fell_over` | 0 |
| `Episode_Termination/time_out` | 7.875 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=70.83 (step 1995)
- mean_reward max=75.63 (step 997)
- mean_episode_length last=982.09 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.057 |
| `Episode_Reward/action_rate_l2` | -0.025 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.243 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.011 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.126 |
| `Episode_Reward/feet_distance` | -0.100 |
| `Episode_Reward/foot_clearance` | -0.007 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.005 |
| `Episode_Reward/foot_swing_height` | 1.188 |
| `Episode_Reward/gait_phase_regularity` | -0.091 |
| `Episode_Reward/joint_acc_l2` | -0.819 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.548 |
| `Episode_Reward/soft_landing` | -0.004 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | 0 |
| `Episode_Reward/torque_rate` | -0.014 |
| `Episode_Reward/track_angular_velocity` | 1.611 |

</details>

### Batch v11

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s1__v11`

- **W&B:** [16cbg6lm](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/16cbg6lm)
- **State:** finished
- **Created:** 2026-07-01T12:50:26
- **Tags:** base-hard, batch-v11, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 47.31 |
| `Train/mean_episode_length` | 989.22 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.045 |
| `Metrics/phase_delta_mean` | 0.001 |
| `Metrics/phase_delta_nominal_error_mean` | 0.661 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.698 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.057 |
| `Episode_Reward/upright` | 0.225 |
| `Episode_Reward/stand_still_pose` | -0.075 |
| `Episode_Termination/fell_over` | 0.667 |
| `Episode_Termination/time_out` | 7.083 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=48.58 (step 3997)
- mean_reward max=71.74 (step 997)
- mean_episode_length last=969.13 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.106 |
| `Episode_Reward/action_rate_l2` | -0.052 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.240 |
| `Episode_Reward/body_ang_vel` | -0.001 |
| `Episode_Reward/command_progress_backslide` | -0.001 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.054 |
| `Episode_Reward/feet_distance` | -0.112 |
| `Episode_Reward/foot_clearance` | -0.003 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.003 |
| `Episode_Reward/foot_swing_height` | 1.189 |
| `Episode_Reward/gait_phase_regularity` | -0.058 |
| `Episode_Reward/joint_acc_l2` | -1.229 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.351 |
| `Episode_Reward/soft_landing` | -0.002 |
| `Episode_Reward/stand_still_motion` | -0.002 |
| `Episode_Reward/termination_penalty` | -0.001 |
| `Episode_Reward/torque_rate` | -0.018 |
| `Episode_Reward/track_angular_velocity` | 1.455 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s2__v11`

- **W&B:** [u5mbohzy](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/u5mbohzy)
- **State:** finished
- **Created:** 2026-07-01T12:59:02
- **Tags:** base-hard, batch-v11, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, seed-2, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 38.29 |
| `Train/mean_episode_length` | 938.95 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.314 |
| `Metrics/phase_delta_mean` | 0.009 |
| `Metrics/phase_delta_nominal_error_mean` | 0.569 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.937 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.062 |
| `Episode_Reward/upright` | 0.212 |
| `Episode_Reward/stand_still_pose` | -0.064 |
| `Episode_Termination/fell_over` | 1.458 |
| `Episode_Termination/time_out` | 6.833 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=33.81 (step 3997)
- mean_reward max=68.64 (step 985)
- mean_episode_length last=854.25 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.114 |
| `Episode_Reward/action_rate_l2` | -0.057 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.221 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.018 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.117 |
| `Episode_Reward/feet_distance` | -0.124 |
| `Episode_Reward/foot_clearance` | -0.014 |
| `Episode_Reward/foot_flat` | -0.004 |
| `Episode_Reward/foot_slip` | -0.011 |
| `Episode_Reward/foot_swing_height` | 1.065 |
| `Episode_Reward/gait_phase_regularity` | -0.071 |
| `Episode_Reward/joint_acc_l2` | -1.384 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.115 |
| `Episode_Reward/soft_landing` | -0.007 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.002 |
| `Episode_Reward/torque_rate` | -0.020 |
| `Episode_Reward/track_angular_velocity` | 1.366 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s3__v11`

- **W&B:** [4jx3q9es](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/4jx3q9es)
- **State:** finished
- **Created:** 2026-07-01T15:37:58
- **Tags:** base-hard, batch-v11, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, seed-3, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 33.02 |
| `Train/mean_episode_length` | 890.23 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.247 |
| `Metrics/phase_delta_mean` | 0.007 |
| `Metrics/phase_delta_nominal_error_mean` | 0.369 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.842 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.084 |
| `Episode_Reward/upright` | 0.215 |
| `Episode_Reward/stand_still_pose` | -0.078 |
| `Episode_Termination/fell_over` | 2.083 |
| `Episode_Termination/time_out` | 6.833 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=35.20 (step 3997)
- mean_reward max=72.06 (step 997)
- mean_episode_length last=922.34 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.113 |
| `Episode_Reward/action_rate_l2` | -0.054 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.214 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.014 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.204 |
| `Episode_Reward/feet_distance` | -0.107 |
| `Episode_Reward/foot_clearance` | -0.010 |
| `Episode_Reward/foot_flat` | -0.002 |
| `Episode_Reward/foot_slip` | -0.008 |
| `Episode_Reward/foot_swing_height` | 1.040 |
| `Episode_Reward/gait_phase_regularity` | -0.073 |
| `Episode_Reward/joint_acc_l2` | -1.307 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.088 |
| `Episode_Reward/soft_landing` | -0.005 |
| `Episode_Reward/stand_still_motion` | -0.002 |
| `Episode_Reward/termination_penalty` | -0.002 |
| `Episode_Reward/torque_rate` | -0.020 |
| `Episode_Reward/track_angular_velocity` | 1.394 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s4__v11`

- **W&B:** [px06ulu0](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/px06ulu0)
- **State:** finished
- **Created:** 2026-07-01T15:47:05
- **Tags:** base-hard, batch-v11, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, seed-4, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 54.76 |
| `Train/mean_episode_length` | 943.24 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.055 |
| `Metrics/phase_delta_mean` | 0.002 |
| `Metrics/phase_delta_nominal_error_mean` | 0.899 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 0.685 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.055 |
| `Episode_Reward/upright` | 0.215 |
| `Episode_Reward/stand_still_pose` | -0.017 |
| `Episode_Termination/fell_over` | 0.958 |
| `Episode_Termination/time_out` | 7.583 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=57.48 (step 3997)
- mean_reward max=76.29 (step 2231)
- mean_episode_length last=965.96 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.121 |
| `Episode_Reward/action_rate_l2` | -0.057 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.231 |
| `Episode_Reward/body_ang_vel` | -0.001 |
| `Episode_Reward/command_progress_backslide` | -0.000 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.117 |
| `Episode_Reward/feet_distance` | -0.077 |
| `Episode_Reward/foot_clearance` | -0.002 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.002 |
| `Episode_Reward/foot_swing_height` | 1.143 |
| `Episode_Reward/gait_phase_regularity` | -0.057 |
| `Episode_Reward/joint_acc_l2` | -0.973 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.542 |
| `Episode_Reward/soft_landing` | -0.002 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.001 |
| `Episode_Reward/torque_rate` | -0.015 |
| `Episode_Reward/track_angular_velocity` | 1.468 |

</details>

### Batch v12

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.2__s1__v12`

- **W&B:** [lyhwmnll](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/lyhwmnll)
- **State:** finished
- **Created:** 2026-07-02T00:31:24
- **Tags:** base-hard, batch-v12, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, pd-tail-0.2, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 67.40 |
| `Train/mean_episode_length` | 992.09 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.218 |
| `Metrics/phase_delta_mean` | 0.006 |
| `Metrics/phase_delta_nominal_error_mean` | 0.923 |
| `Episode_Reward/phase_delta_nominal` | -0.146 |
| `Episode_Reward/track_linear_velocity` | 0.834 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.014 |
| `Episode_Reward/upright` | 0.212 |
| `Episode_Reward/stand_still_pose` | -0.009 |
| `Episode_Termination/fell_over` | 0.125 |
| `Episode_Termination/time_out` | 8.500 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=67.79 (step 3997)
- mean_reward max=82.99 (step 995)
- mean_episode_length last=995.49 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.045 |
| `Episode_Reward/action_rate_l2` | -0.020 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.239 |
| `Episode_Reward/body_ang_vel` | -0.001 |
| `Episode_Reward/command_progress_backslide` | -0.010 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.018 |
| `Episode_Reward/feet_distance` | -0.026 |
| `Episode_Reward/foot_clearance` | -0.008 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.006 |
| `Episode_Reward/foot_swing_height` | 1.149 |
| `Episode_Reward/gait_phase_regularity` | -0.071 |
| `Episode_Reward/joint_acc_l2` | -1.040 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.794 |
| `Episode_Reward/soft_landing` | -0.004 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.016 |
| `Episode_Reward/track_angular_velocity` | 1.554 |

</details>

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__s1__v12`

- **W&B:** [260z9ekp](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/260z9ekp)
- **State:** finished
- **Created:** 2026-07-02T00:31:28
- **Tags:** base-hard, batch-v12, clock_learned, critic-height-scan, gridsearch, joule-3e-4, pc-0.5, pd-tail-0.1, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 28.92 |
| `Train/mean_episode_length` | 867.41 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.298 |
| `Metrics/phase_delta_mean` | 0.009 |
| `Metrics/phase_delta_nominal_error_mean` | 0.958 |
| `Episode_Reward/phase_delta_nominal` | -0.067 |
| `Episode_Reward/track_linear_velocity` | 0.821 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.082 |
| `Episode_Reward/upright` | 0.197 |
| `Episode_Reward/stand_still_pose` | -0.092 |
| `Episode_Termination/fell_over` | 2.750 |
| `Episode_Termination/time_out` | 6.917 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=27.89 (step 3997)
- mean_reward max=68.75 (step 994)
- mean_episode_length last=873.28 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.106 |
| `Episode_Reward/action_rate_l2` | -0.048 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.210 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.016 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.366 |
| `Episode_Reward/feet_distance` | -0.113 |
| `Episode_Reward/foot_clearance` | -0.012 |
| `Episode_Reward/foot_flat` | -0.003 |
| `Episode_Reward/foot_slip` | -0.008 |
| `Episode_Reward/foot_swing_height` | 1.006 |
| `Episode_Reward/gait_phase_regularity` | -0.068 |
| `Episode_Reward/joint_acc_l2` | -1.149 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.039 |
| `Episode_Reward/soft_landing` | -0.005 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.003 |
| `Episode_Reward/torque_rate` | -0.021 |
| `Episode_Reward/track_angular_velocity` | 1.294 |

</details>

### Batch v13

#### `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__joule-1e-5__s1__v13`

- **W&B:** [l9wok1ss](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/l9wok1ss)
- **State:** finished
- **Created:** 2026-07-02T05:09:43
- **Tags:** base-hard, batch-v13, clock_learned, critic-height-scan, gridsearch, joule-1e-5, pc-0.5, pd-tail-0.1, seed-1, stand-0.15

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 68.98 |
| `Train/mean_episode_length` | 987.42 |
| `Metrics/phase_delta_nominal_ratio_mean` | 0.054 |
| `Metrics/phase_delta_mean` | 0.002 |
| `Metrics/phase_delta_nominal_error_mean` | 1.090 |
| `Episode_Reward/phase_delta_nominal` | -0.086 |
| `Episode_Reward/track_linear_velocity` | 0.725 |
| `Episode_Reward/foot_swing_height_landing` | 0 |
| `Episode_Reward/joule_heating` | -0.000 |
| `Episode_Reward/upright` | 0.211 |
| `Episode_Reward/stand_still_pose` | -0.008 |
| `Episode_Termination/fell_over` | 0.250 |
| `Episode_Termination/time_out` | 8.333 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=69.94 (step 3997)
- mean_reward max=84.65 (step 994)
- mean_episode_length last=1000 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.046 |
| `Episode_Reward/action_rate_l2` | -0.021 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.236 |
| `Episode_Reward/body_ang_vel` | -0.001 |
| `Episode_Reward/command_progress_backslide` | -0.001 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.023 |
| `Episode_Reward/feet_distance` | -0.021 |
| `Episode_Reward/foot_clearance` | -0.003 |
| `Episode_Reward/foot_flat` | -0.001 |
| `Episode_Reward/foot_slip` | -0.002 |
| `Episode_Reward/foot_swing_height` | 1.173 |
| `Episode_Reward/gait_phase_regularity` | -0.061 |
| `Episode_Reward/joint_acc_l2` | -0.956 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.794 |
| `Episode_Reward/soft_landing` | -0.002 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.015 |
| `Episode_Reward/track_angular_velocity` | 1.492 |

</details>

#### `clock_anneal__stand-0.15__pc-0.5__s1__v13`

- **W&B:** [ojozkbfs](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ojozkbfs)
- **State:** finished
- **Created:** 2026-07-02T05:09:25
- **Tags:** batch-v13, clock_anneal, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 54.99 |
| `Train/mean_episode_length` | 984.77 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.505 |
| `Episode_Reward/foot_swing_height_landing` | -0.053 |
| `Episode_Reward/joule_heating` | -0.021 |
| `Episode_Reward/upright` | 0.485 |
| `Episode_Reward/stand_still_pose` | -0.033 |
| `Episode_Termination/fell_over` | 0.167 |
| `Episode_Termination/time_out` | 6.500 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=56.11 (step 1995)
- mean_reward max=69.57 (step 995)
- mean_episode_length last=1000 (step 1995)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.015 |
| `Episode_Reward/action_rate_l2` | -0.010 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0.023 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.240 |
| `Episode_Reward/body_ang_vel` | -0.003 |
| `Episode_Reward/command_progress_backslide` | -0.051 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.041 |
| `Episode_Reward/feet_distance` | -0.096 |
| `Episode_Reward/foot_clearance` | -0.031 |
| `Episode_Reward/foot_flat` | -0.003 |
| `Episode_Reward/foot_slip` | -0.018 |
| `Episode_Reward/foot_swing_height` | 0 |
| `Episode_Reward/gait_phase_regularity` | -0.076 |
| `Episode_Reward/joint_acc_l2` | -1.072 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.410 |
| `Episode_Reward/soft_landing` | -0.011 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.000 |
| `Episode_Reward/torque_rate` | -0.014 |
| `Episode_Reward/track_angular_velocity` | 1.640 |

</details>

### Batch v14

#### `clock_anneal__stand-0.15__pc-0.5__base-hard__s1__v14`

- **W&B:** [jyksw3mg](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/jyksw3mg)
- **State:** finished
- **Created:** 2026-07-02T08:44:02
- **Tags:** base-hard, batch-v14, clock_anneal, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 33.43 |
| `Train/mean_episode_length` | 913.16 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.190 |
| `Episode_Reward/foot_swing_height_landing` | -0.036 |
| `Episode_Reward/joule_heating` | -0.044 |
| `Episode_Reward/upright` | 0.201 |
| `Episode_Reward/stand_still_pose` | -0.016 |
| `Episode_Termination/fell_over` | 1.083 |
| `Episode_Termination/time_out` | 7.208 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=36.11 (step 3997)
- mean_reward max=71.79 (step 994)
- mean_episode_length last=965.24 (step 3997)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.050 |
| `Episode_Reward/action_rate_l2` | -0.025 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0.017 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.227 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.042 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.053 |
| `Episode_Reward/feet_distance` | -0.081 |
| `Episode_Reward/foot_clearance` | -0.025 |
| `Episode_Reward/foot_flat` | -0.004 |
| `Episode_Reward/foot_slip` | -0.016 |
| `Episode_Reward/foot_swing_height` | 0 |
| `Episode_Reward/gait_phase_regularity` | -0.066 |
| `Episode_Reward/joint_acc_l2` | -1.298 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.472 |
| `Episode_Reward/soft_landing` | -0.009 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.001 |
| `Episode_Reward/torque_rate` | -0.017 |
| `Episode_Reward/track_angular_velocity` | 1.379 |

</details>

### Batch v15

#### `clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s1__v15`

- **W&B:** [ynquy630](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ynquy630)
- **State:** running
- **Created:** 2026-07-02T14:44:45
- **Tags:** 20k, base-hard, batch-v15, clock_anneal, gridsearch, joule-3e-4, pc-0.5, seed-1, stand-0.15, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 37.62 |
| `Train/mean_episode_length` | 955.97 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.254 |
| `Episode_Reward/foot_swing_height_landing` | -0.025 |
| `Episode_Reward/joule_heating` | -0.057 |
| `Episode_Reward/upright` | 0.212 |
| `Episode_Reward/stand_still_pose` | -0.022 |
| `Episode_Termination/fell_over` | 0.750 |
| `Episode_Termination/time_out` | 7.792 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=38.09 (step 14657)
- mean_reward max=67.77 (step 980)
- mean_episode_length last=964.14 (step 14657)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.093 |
| `Episode_Reward/action_rate_l2` | -0.057 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0.016 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.225 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.037 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.111 |
| `Episode_Reward/feet_distance` | -0.095 |
| `Episode_Reward/foot_clearance` | -0.023 |
| `Episode_Reward/foot_flat` | -0.006 |
| `Episode_Reward/foot_slip` | -0.012 |
| `Episode_Reward/foot_swing_height` | 0 |
| `Episode_Reward/gait_phase_regularity` | -0.064 |
| `Episode_Reward/joint_acc_l2` | -1.107 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.445 |
| `Episode_Reward/soft_landing` | -0.007 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.001 |
| `Episode_Reward/torque_rate` | -0.014 |
| `Episode_Reward/track_angular_velocity` | 1.468 |

</details>

#### `clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s2__v15`

- **W&B:** [rntq7onj](https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/rntq7onj)
- **State:** running
- **Created:** 2026-07-02T14:46:14
- **Tags:** 20k, base-hard, batch-v15, clock_anneal, gridsearch, joule-3e-4, pc-0.5, seed-2, stand-0.15, upright-0.5

| Metric | Summary (last logged) |
| --- | ---: |
| `Train/mean_reward` | 39.97 |
| `Train/mean_episode_length` | 955.63 |
| `Episode_Reward/phase_delta_nominal` | 0 |
| `Episode_Reward/track_linear_velocity` | 1.223 |
| `Episode_Reward/foot_swing_height_landing` | -0.024 |
| `Episode_Reward/joule_heating` | -0.046 |
| `Episode_Reward/upright` | 0.213 |
| `Episode_Reward/stand_still_pose` | -0.019 |
| `Episode_Termination/fell_over` | 1.125 |
| `Episode_Termination/time_out` | 7.167 |

**Training trajectory (`Train/mean_reward`, `Train/mean_episode_length`):**

- mean_reward last=40.38 (step 14657)
- mean_reward max=70.75 (step 980)
- mean_episode_length last=959.99 (step 14657)

<details><summary>Other episode reward summaries</summary>

| Metric | Value |
| --- | ---: |
| `Episode_Reward/action_acc_l2` | -0.056 |
| `Episode_Reward/action_rate_l2` | -0.041 |
| `Episode_Reward/actuation_power` | 0 |
| `Episode_Reward/air_time` | 0.015 |
| `Episode_Reward/angular_momentum` | -0.000 |
| `Episode_Reward/base_height` | 0.225 |
| `Episode_Reward/body_ang_vel` | -0.002 |
| `Episode_Reward/command_progress_backslide` | -0.042 |
| `Episode_Reward/cot_proxy` | 0 |
| `Episode_Reward/dof_pos_limits` | -0.054 |
| `Episode_Reward/feet_distance` | -0.091 |
| `Episode_Reward/foot_clearance` | -0.021 |
| `Episode_Reward/foot_flat` | -0.006 |
| `Episode_Reward/foot_slip` | -0.012 |
| `Episode_Reward/foot_swing_height` | 0 |
| `Episode_Reward/gait_phase_regularity` | -0.065 |
| `Episode_Reward/joint_acc_l2` | -1.139 |
| `Episode_Reward/limb_symmetry` | 0 |
| `Episode_Reward/pose` | 0.465 |
| `Episode_Reward/soft_landing` | -0.007 |
| `Episode_Reward/stand_still_motion` | -0.001 |
| `Episode_Reward/termination_penalty` | -0.001 |
| `Episode_Reward/torque_rate` | -0.015 |
| `Episode_Reward/track_angular_velocity` | 1.465 |

</details>

## Cross-batch comparison (`Train/mean_reward`, summary last)

| Batch | Run | State | mean_reward | episode_length | phase_delta_nominal_ratio |
| --- | --- | --- | ---: | ---: | ---: |
| v8 | `clock_learned__stand-0.15__pc-0.5__cur0__s1__v8` | finished | 94.22 | 991.77 | 0.000 |
| v8 | `clock_learned__stand-0.15__pc-0.5__cur1__s1__v8` | finished | 100.48 | 997.29 | -0.000 |
| v9 | `clock_anneal__stand-0.15__pc-0.5__s1__v9` | finished | 14.50 | 998.56 | — |
| v9 | `clock_learned__stand-0.15__pc-0.5__cur0__s1__v9` | finished | 63.50 | 981.91 | 0.664 |
| v9 | `clock_learned__stand-0.15__pc-0.5__cur1__s1__v9` | finished | 41.51 | 985.12 | 0.609 |
| v10 | `clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10` | crashed | 52.98 | 1000 | 0.001 |
| v10 | `clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10` | finished | 72.07 | 1000 | 0.167 |
| v11 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s1__v11` | finished | 47.31 | 989.22 | 0.045 |
| v11 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s2__v11` | finished | 38.29 | 938.95 | 0.314 |
| v11 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s3__v11` | finished | 33.02 | 890.23 | 0.247 |
| v11 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s4__v11` | finished | 54.76 | 943.24 | 0.055 |
| v12 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.2__s1__v12` | finished | 67.40 | 992.09 | 0.218 |
| v12 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__s1__v12` | finished | 28.92 | 867.41 | 0.298 |
| v13 | `clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__joule-1e-5__s1__v13` | finished | 68.98 | 987.42 | 0.054 |
| v13 | `clock_anneal__stand-0.15__pc-0.5__s1__v13` | finished | 54.99 | 984.77 | — |
| v14 | `clock_anneal__stand-0.15__pc-0.5__base-hard__s1__v14` | finished | 33.43 | 913.16 | — |
| v15 | `clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s1__v15` | running | 37.62 | 955.97 | — |
| v15 | `clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s2__v15` | running | 39.97 | 955.63 | — |

## Duplicate / superseded attempts

- **`clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10`** (3 attempts): finished [cxl0l9d8], crashed [ukz6aprc], crashed [lhlbam8a]
- **`clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10`** (3 attempts): finished [1tdz3rjv], crashed [h8hszyrf], finished [yz5baxda]
