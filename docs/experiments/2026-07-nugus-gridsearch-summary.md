# Nugus grid-search experiment summary (June–July 2026)

**Project:** `vincenttumm-the-university-of-newcastle/mjlab`  
**TensorBoard:** https://mjlab.4ai.systems (experiment groups `nugus_gridsearch_v3` … `nugus_gridsearch_v15`)  
**Document date:** 2026-07-03  
**Data sources:** W&B API (entity `vincenttumm-the-university-of-newcastle`, project `mjlab`), `kubectl -n mjlab` vcjob/pod logs, `scripts/k8s/gen-gridsearch.sh`, agent transcript `d15bf769-d3ca-43e3-bae2-e126adb7ca6f`.

Metric values below are **final-run summaries from W&B** unless noted as **in-progress (kubectl logs)**. W&B summary key used: `Train/mean_reward`, `Train/mean_episode_length`, `Metrics/phase_delta_nominal_ratio_mean` (when present).

---

## 1. Executive summary

The Nugus grid-search program on Kubernetes (Volcano queue `mjlab-train`) set out to:

1. **Eliminate shuffling** — use a fixed gait-clock scaffold (`clock_anneal`) or policy-owned phase (`clock_learned`) with staged reward handoff instead of a gameable clearance term alone.
2. **Compare curriculum variants** — `clock_anneal` (teacher clock in obs/rewards, annealed out), `clock_learned` (policy accumulates phase via `phase_delta`; nominal cadence penalty), and early matrix cells (`self_paced`, `clock_persist`).
3. **Tune standing / upright / cadence** — `STAND_W`, halved `UPRIGHT_W=0.5`, extended `PHASE_DELTA_STRONG_ITERS=1000` at weight `-5.0`, and later non-zero **phase-delta tail** weights so cadence does not collapse when the strong penalty ends.
4. **Harder locomotion** — `TRAINING_REGIME=hard_continue` ramps command velocity, pushes, and torso freedom; tested both **resume-from-checkpoint** (v10 stage A) and **single-run base→hard** (v11–v15) with `CONT_BASE_STEP=48000` (2000 iters × 24 steps/iter).
5. **Critic and energy ablations** — `CRITIC_HEIGHT_SCAN=true` on flat envs (v10+ learned runs), `JOULE_W=1e-5` vs default `3e-4` (v13).

**Headline outcomes (final W&B summaries):**

| Direction | Best run(s) | Mean reward | Notes |
|-----------|-------------|-------------|-------|
| Flat `clock_learned`, 2k iters | v8 cur1 `v21878z8` | **100.48** | Strong early penalty at commit `2229f92`; phase still collapsed early → led to v9 |
| Flat `clock_learned`, v9 settings | v9 cur0 `ift9sd2w` | **63.50** | Chosen base for v10 continuation; `pd_ratio` 0.66 at end |
| Flat retrain + height-scan critic | v10 hs `yz5baxda` | **72.07** | v9-equivalent fresh train; best completed v10 job |
| Base→hard, learned + pd-tail | v12 pd-tail −0.2 `lyhwmnll` | **67.40** | Large gain vs tail −0.1 (28.92); addresses v11 cadence collapse |
| Base→hard, low joule | v13 cl joule 1e-5 `l9wok1ss` | **68.98** | Similar to v12 −0.2 with weaker heating penalty |
| Flat `clock_anneal`, 2k | v13 ca `ojozkbfs` | **54.99** | v9-equivalent baseline |
| Base→hard, `clock_anneal` 4k | v14 `jyksw3mg` | **33.43** | Hard phase reduced reward vs flat 2k baseline |
| Base→hard, `clock_anneal` 20k | v15 s1/s2 `ynquy630` / `rntq7onj` | **38–40** (stopped) | Flatlined iter ~4k–15k; fell_over 0.5–1.1/ep — **stop per F2** |

Resume-based hard continuation (v10 stage A) **crashed repeatedly**; the program shifted to single-run base→hard (v11+) and `clock_anneal` long runs (v14–v15). **v15 stopped** (2026-07-03): extra compute past ~4k iters did not recover pre-ramp performance; see §3 v15 and sim2real plan F2.

### Standing constraints (future batches)

Per sim2real plan (`docs/plans/sim2real-training-regime/README.md`):

- **`clock_anneal` only** — retire `clock_learned` until hardware walking works (F3/B4).
- **`CRITIC_HEIGHT_SCAN=true`** — v10 showed fell_over → 0; v14/v15 dropped it and regressed.
- **`JOULE_W=1e-5`**, not default `3e-4` — v13 evidence; v12 tail −0.1 collapsed at 3e-4.
- **Use fixed eval (E0.1) for comparisons** — reward curves are not comparable across curriculum stages or physics changes (F1).

**v16-short (Phase 0 validation, 2026-07-03):** `BATCH=v16-short` → `scripts/k8s/gen_v16_short/mj-gs-v16-short-ca-hs-joule-1e5.yaml` (500 iters). **Completed** @ `a1af0d4` — gate **PASS** @ iter ~336 (see §8); W&B `gr1hb5uh`, final reward **26.75** @ iter 499; fixed eval **falls_per_min 0.0** overall and at `(0.5,0,0)` / `(0,0,0)` (local smoke, 1 env/cmd — see §8).

**v16 (Phase 0 smoke, 2026-07-03):** `BATCH=v16` → `scripts/k8s/gen_v16/mj-gs-v16-ca-hs-joule-1e5.yaml` (2000 iters). **Completed** @ `a1af0d4` — walk gate **PASS** @ iter ~630–960 (kubectl); final reward **3.08**, ep_len **165** @ iter 1999 (late regression post ~iter 1000 curriculum); W&B `shhm98rd`. **v17 queued** (5 cells) after gate + completion.

---

## 2. Timeline / batches

Shared defaults unless overridden: `PHASE_C_FRAC=0.5`, flat task `Mjlab-Velocity-Flat-Nubots-Nugus`, `GAIT_PERIOD=0.7`, `JOULE_W=3e-4`, seed 1, W&B project `mjlab`, experiment name `nugus_gridsearch_<batch>`.

| Batch | Intention | Key hyperparams | K8s job(s) | W&B run name / ID | Final metrics (W&B) | Outcome |
|-------|-----------|-----------------|------------|-------------------|---------------------|---------|
| **v3** | Initial 12-cell matrix: `clock_anneal` / `self_paced` / `clock_persist` × `STAND_W` {0.1,0.3} × `pc` {0.5,0.7} | 1250 iters | (TTL expired) | e.g. `76kregdc` clock_anneal sw0.1 pc0.5 | reward **72.74**, ep_len **1000** | Best early `clock_anneal` pc-0.5; used as v4 resume sources (`7fivy5q7`, `eyiowvgo`) |
| **v4** | Resume two v3 `clock_anneal` pc-0.5 runs to ~2000 iters | `RESUME=true`, `WANDB_RUN_PATH` from v3 | `mj-gs-v4-ca-sw01-p05-cont`, `…-sw03-…` | **No W&B runs** (failed at init) | metrics not retrieved | Failed: tyro `--agent.resume` CLI bug (see §4) |
| **v5** | 2×2: `CURRENT_OBS` × `SILENCE_CLOCK` on `clock_anneal` | 2000 iters, `STAND_W=0.15` | (TTL expired) | `l9c9iicz` cur1 sil0 | reward **78.40**, ep_len **997.73** | Current obs + silence grid; best cell cur1/sil0 |
| **v6** | Command resample min 3.0 vs 0.0 | `clock_anneal`, 2000 iters | Not deployed initially | **No tagged runs** | metrics not retrieved | Generator added; jobs not launched in this window |
| **v7** | `clock_learned` vs `clock_anneal` | 2000 iters, `PHASE_ITERATIONS=2000` | (TTL expired) | `19jgdqlx` cl, `j8r4iqmi` ca | cl **93.19** / ca **63.25** | Learned variant scored higher but early asymmetry vs anneal noted |
| **v8** | Strong early `phase_delta_nominal` at git `2229f92`; cur0 vs cur1 | 2000 iters, `-5.0` strong stage (100 iters at this commit) | (TTL expired) | `57gr0muv` cur0, `v21878z8` cur1 | **94.22** / **100.48** | High flat reward but phase step → 0 too soon → v9 |
| **v9** | Strong penalty **1000 iters**, `UPRIGHT_W=0.5`, + `clock_anneal` baseline | 3 jobs | (TTL expired) | `ift9sd2w` cur0, `ffcdmlkz` cur1, `xm5t9ilu` ca | **63.50** / **41.51** / **14.50** | cur0 selected for v10; empty-env + curriculum-order bugs fixed mid-batch |
| **v10** | (A) hard resume from v9 cur0; (B) flat hs-critic retrain | `TRAINING_REGIME=hard_continue`, `CRITIC_HEIGHT_SCAN` true/false | `mj-gs-v10-cl-cur0-hs` (cont jobs removed) | hs: `yz5baxda`; cont attempts: `cxl0l9d8` (finished), `ukz6aprc`, `lhlbam8a` (crashed) | hs **72.07**; cont best finished **49.55** | Stage B succeeded; stage A resume unreliable |
| **v11** | Overnight base→hard, 4 seeds, hs-critic | 4000 iters, `CONT_BASE_STEP=48000` | (TTL expired) | `16cbg6lm` s1 … `px06ulu0` s4 | **47.31** / **38.29** / **33.02** / **54.76** | Cadence collapsed ~iter 1000 when strong pd penalty ended |
| **v12** | v11-like + **pd-tail** −0.2 vs −0.1 | 4000 iters | `mj-gs-v12-cl-pd-tail-0.2`, `…-0.1` | `lyhwmnll`, `260z9ekp` | **67.40** / **28.92** | Tail −0.2 clearly better |
| **v13** | Low joule 1e-5 (learned base→hard) + v9-like `clock_anneal` 2k | 4000 / 2000 iters | `mj-gs-v13-cl-joule-1e5-pd01`, `mj-gs-v13-ca` | `l9wok1ss`, `ojozkbfs` | **68.98** / **54.99** | Low joule matches v12-quality learned hard run |
| **v14** | `clock_anneal` base→hard single run, legacy critic | 4000 iters | `mj-gs-v14-ca-base-hard` | `jyksw3mg` | **33.43**, ep_len **913.16** | Hard regime hurt vs flat 2k anneal |
| **v15** | v14 extended to **20k** iters, seeds 1–2 | `MAX_ITERATIONS=20000` | `mj-gs-v15-ca-base-hard-20k`, `…-s2` | `ynquy630` s1, `rntq7onj` s2 | **38–40** plateau / fell **0.5–1.1**/ep | **Stopped** per F2 — flatlined ~4k–15k; do not relaunch |
| **v16-short** | Phase 0 validation — same as v16, shorter | `clock_anneal`, **500** iters, `JOULE_W=1e-5`, hs-critic, `TRAINING_REGIME=base` | `mj-gs-v16-short-ca-hs-joule-1e5` | `gr1hb5uh` | reward **26.75**, ep_len **851** @ iter 499 | **PASS** — completed 2026-07-03 |
| **v16** | Phase 0 smoke — post-E0.2/A3/C1 physics base | `clock_anneal`, 2k iters, `JOULE_W=1e-5`, `CRITIC_HEIGHT_SCAN=true`, `TRAINING_REGIME=base` | `mj-gs-v16-ca-hs-joule-1e5` | `shhm98rd` | reward **3.08**, ep_len **165** @ iter 1999; peak **~53** @ iter ~960 | **Completed** @ `a1af0d4` |

**Legacy job still on cluster:** `mjlab-gs-clock-anneal-joule-1e-4-pc-0-5-s1` → W&B `ufk65r9v` (v3-era naming, 1250 iters, final summary reward **−0.17**). Pod logs at iter **1187/1250**: mean reward **−25.40** (in progress, not final).

---

## 3. Per-run sections

### v3 — initial matrix (`nugus_gridsearch_v3`)

**Intention:** Compare gait strategies and stand-weight / phase-C timing on the original 12-cell grid.

**Config:** `MJLAB_VARIANT ∈ {clock_anneal, self_paced, clock_persist}`, `STAND_W ∈ {0.1, 0.3}`, `PHASE_C_FRAC ∈ {0.5, 0.7}`, `MAX_ITERATIONS=1250`.

**Notable runs (W&B final summaries):**

| Run ID | Variant | STAND_W | pc | Mean reward | Ep len |
|--------|---------|---------|-----|-------------|--------|
| `76kregdc` | clock_anneal | 0.1 | 0.5 | 72.74 | 1000 |
| `7fivy5q7` | clock_anneal | 0.1 | 0.5 | 59.91 | 994.62 |
| `eyiowvgo` | clock_anneal | 0.3 | 0.5 | 65.08 | 1000 |
| `espatus7` | clock_anneal | 0.3 | 0.5 | 87.46 | 994.29 | state: crashed |

**Outcome:** `clock_anneal` at pc-0.5 outperformed `self_paced` / `clock_persist`. pc-0.7 and higher stand weights were deprioritized in later batches.

---

### v4 — continuation (failed)

**Intention:** Add 750 iters to the two best v3 `clock_anneal` pc-0.5 runs (`7fivy5q7`, `eyiowvgo`) with frozen phase boundaries.

**Config:** `RESUME=true`, `WANDB_RUN_PATH` set per cell, `MAX_ITERATIONS=1250` (additive).

**Metrics:** not retrieved — jobs failed before W&B run creation.

---

### v5 — current obs × clock silence

**Intention:** Test actuator-current observations and silencing the gait-clock observation on `clock_anneal`.

| Run ID | CURRENT_OBS | SILENCE_CLOCK | Mean reward | Ep len |
|--------|-------------|---------------|-------------|--------|
| `t4sv9kq2` | 0 | 0 | 69.13 | 1000 |
| `x51sayjr` | 0 | 1 | 74.48 | 1000 |
| `l9c9iicz` | 1 | 0 | **78.40** | 997.73 |
| `u5sc6wum` | 1 | 1 | 77.73 | 1000 |

---

### v6 — rapid command resampling

**Intention:** A/B `RESAMPLE_MIN` 3.0 vs 0.0 s on `clock_anneal`.

**Metrics:** not retrieved — no `batch-v6` W&B runs found.

---

### v7 — clock_learned vs clock_anneal

| Run ID | Variant | Mean reward | Ep len | pd_ratio |
|--------|---------|-------------|--------|----------|
| `19jgdqlx` | clock_learned | **93.19** | 992.89 | 0.00023 |
| `j8r4iqmi` | clock_anneal | 63.25 | 995.66 | — |

**Note:** High learned reward partly reflects different early obs/reward structure (policy-owned phase vs teacher clock), not apples-to-apples locomotion quality.

---

### v8 — strong early penalty (commit `2229f92`)

**Intention:** Pin `2229f92` (−5.0 nominal penalty for first **100** iters); compare `CURRENT_OBS` 0 vs 1.

| Run ID | CURRENT_OBS | Mean reward | Ep len | pd_ratio |
|--------|-------------|-------------|--------|----------|
| `57gr0muv` | 0 | 94.22 | 991.77 | 0.000074 |
| `v21878z8` | 1 | **100.48** | 997.29 | −0.00012 |

**Observation:** Policy still drove phase step toward zero after the short strong window → motivated v9 (1000-iter strong stage).

---

### v9 — extended strong penalty + upright cut

**Intention:** `PHASE_DELTA_STRONG_ITERS=1000`, `UPRIGHT_W=0.5`, `PROGRESS_BACKSLIDE_W=-0.5`; three cells: learned cur0/cur1 + `clock_anneal`.

| Run ID | Cell | Mean reward | Ep len | pd_ratio |
|--------|------|-------------|--------|----------|
| `ift9sd2w` | cl cur0 | **63.50** | 981.91 | **0.664** |
| `ffcdmlkz` | cl cur1 | 41.51 | 985.12 | 0.609 |
| `xm5t9ilu` | ca | 14.50 | 998.56 | — |

**Trajectory (kubectl / W&B):** User-selected base for harder training: `2026-07-01_07-25-50_clock_learned__stand-0.15__pc-0.5__cur0__s1__v9`.

---

### v10 — hard continuation + hs-critic retrain

**Intention:**

- **Stage A:** `RESUME=true` from v9 cur0 (`ift9sd2w`), `TRAINING_REGIME=hard_continue`, legacy critic, +2000 iters.
- **Stage B:** Fresh v9-equivalent train with `CRITIC_HEIGHT_SCAN=true`.

| Run ID | Stage | State | Mean reward | Ep len | pd_ratio |
|--------|-------|-------|-------------|--------|----------|
| `cxl0l9d8` | A cont (early) | finished | 49.55 | 964.99 | 0.042 |
| `ukz6aprc` | A cont retry | crashed | 58.73 | 989.09 | 0.376 |
| `lhlbam8a` | A cont retry | crashed | 52.98 | 1000 | 0.0013 |
| `yz5baxda` | B hs-critic | finished | **72.07** | 1000 | 0.167 |

**K8s:** `mj-gs-v10-cl-cur0-hs` — pod **Succeeded** (2000/2000); vcjob status may still show Running until TTL.

**Stage B trajectory (kubectl logs):** iter 1999/2000 — mean reward **72.07**, ep_len **1000.00**; `track_linear_velocity` episode reward **1.41** at last log block.

---

### v11 — overnight base→hard (4 seeds)

**Intention:** Single 4000-iter run: base for 2000 iters then `hard_continue` without resume; hs-critic.

| Seed | Run ID | Mean reward | Ep len | pd_ratio |
|------|--------|-------------|--------|----------|
| 1 | `16cbg6lm` | 47.31 | 989.22 | 0.045 |
| 2 | `u5mbohzy` | 38.29 | 938.95 | 0.314 |
| 3 | `4jx3q9es` | 33.02 | 890.23 | 0.247 |
| 4 | `px06ulu0` | **54.76** | 943.24 | 0.055 |

**Issue:** Performance strong until ~iter 1000, then cadence slowed when `phase_delta_nominal` strong stage ended → v12 tail penalty.

---

### v12 — phase-delta tail weights

**Intention:** Same as v11 but `PHASE_DELTA_TAIL_W ∈ {-0.2, -0.1}` after 1000-iter strong start.

| Tail | Run ID | Mean reward | Ep len | pd_ratio | K8s job |
|------|--------|-------------|--------|----------|---------|
| −0.2 | `lyhwmnll` | **67.40** | 992.09 | 0.218 | `mj-gs-v12-cl-pd-tail-0.2` Completed |
| −0.1 | `260z9ekp` | 28.92 | 867.41 | 0.298 | `mj-gs-v12-cl-pd-tail-0.1` Completed |

**Trajectory (kubectl, final iters):** pd-tail −0.1 iter 3999 — reward **28.92**, ep_len **867.41**; pd-tail −0.2 iter 3999 — reward **63.64**, ep_len **972.09** (log snapshot near end).

---

### v13 — joule 1e-5 + clock_anneal baseline

| Cell | Run ID | Mean reward | Ep len | pd_ratio |
|------|--------|-------------|--------|----------|
| cl, joule 1e-5, pd-tail −0.1 | `l9wok1ss` | **68.98** | 987.42 | 0.054 |
| ca, v9-like 2k | `ojozkbfs` | 54.99 | 984.77 | — |

---

### v14 — clock_anneal base→hard (4k)

**Run:** `jyksw3mg` — mean reward **33.43**, ep_len **913.16** (final W&B).  
**K8s:** `mj-gs-v14-ca-base-hard` Completed.

**Interpretation:** Extending `clock_anneal` through hard_continue in one 4k run yielded lower total reward than flat 2k v13 ca (**54.99**), with shorter episodes.

---

### v15 — clock_anneal base→hard (20k, 2 seeds) — **STOPPED / FINAL**

**Intention:** Hold final hard parameters from ~iter 3000 through 20k for robustness / long-horizon behavior.

**Outcome (F2):** Both seeds **flatlined** at mean reward **38–40** from iter ~4k through ~15k. `Episode_Termination/fell_over` oscillated **0.5–1.1 per episode** while `track_linear_velocity` held ~1.25–1.28 — the hard-stage problem is stability, not tracking. Extra compute does not recover the pre-ramp peak (~70 @ iter ~860). **Recommendation: stop; do not extend or relaunch.**

| Seed | Run ID | W&B state | Mean reward (final) | Ep len | fell_over/ep (@15k sample) |
|------|--------|-----------|---------------------|--------|----------------------------|
| 1 | `ynquy630` | stopped (rec.) | 37.99 | 953.16 | 0.46–0.67 |
| 2 | `rntq7onj` | stopped (rec.) | 41.11 | 966.15 | 0.88–1.12 |

**Trajectory (W&B history samples):** s1 — 3916: 35.9/1.08, 7778: 39.3/0.50, 15105: 38.9/0.46; s2 — 3916: 35.2/0.88, 7778: 38.3/0.88, 15105: 39.7/1.12 (format: reward / fell_over).

**K8s:** `mj-gs-v15-ca-base-hard-20k`, `mj-gs-v15-ca-base-hard-20k-s2` — **deleted** 2026-07-03 at ~iter 16200/20000 (orchestration agent); freed 8 GPUs on `mjlab-train`.

---

## 4. Failures and fixes

| Issue | Affected runs | Fix |
|-------|---------------|-----|
| **Resume CLI bug** — bare `--agent.resume` caused tyro to consume `--wandb-run-path` as the resume value | v4 (`mj-gs-v4-ca-*-cont`) | `entrypoint.sh`: `--agent.resume True`; cluster ConfigMap updated |
| **Empty env vars** — `PHASE_DELTA_STRONG_W=""` etc. crashed import | v9 initial launch (all 3 jobs failed) | `_env_float`/`_env_int`/`_env_bool` treat `""` as unset; entrypoint skips empty exports; v9 re-queued |
| **Curriculum stage ordering** — strong pd window to iter 1000 conflicted with short phase stages | v9 `clock_learned` jobs (2 failed, ca ok) | Curriculum ordering fix (`05bc2bd`); learned jobs re-queued |
| **`command_progress_backslide` disabled** — `PROGRESS_BACKSLIDE_W` defaulted to 0, not wired in k8s | v3–v8 grid runs | Default **−0.5**; wired in template + v9+ generators |
| **GIT_COMMIT pin not on remote** — shallow fetch by SHA failed | v10 early attempts (`6c77e8a` pin) | Push pin commits to `origin/add-phase-clock`; full 40-char SHA fetch in entrypoint |
| **`actuator_torque_rate_l2` init crash** | Early grid jobs on `8caf573` | Fixed `dfcad18` / later branch tip |
| **Resume + hard_continue unreliable** | v10 stage A (`cxl0l9d8`, `ukz6aprc`, `lhlbam8a`) | Pivoted to single-run base→hard (v11+); resume path still available but not used for mainline after v10 |
| **Phase cadence collapse after iter 1000** | v11 all seeds | v12 `PHASE_DELTA_TAIL_W` non-zero tail; −0.2 best |
| **v10 stage A unpushed pin / cluster apply** | First v10 cont + hs pair | Re-applied with remote SHAs; only hs job retained on cluster |

---

## 5. Current cluster status

Snapshot: `kubectl get vcjob -n mjlab` on **2026-07-03 ~14:22 JST** (5-min poll loop, ~90m elapsed).

| vcjob | Status | Running pods | Notes |
|-------|--------|--------------|-------|
| `mj-gs-v10-cl-cur0-hs` | Running | 0 | Pod Completed; vcjob status lag |
| `mj-gs-v12-cl-pd-tail-0.1` | Completed | 0 | |
| `mj-gs-v12-cl-pd-tail-0.2` | Completed | 0 | |
| `mj-gs-v13-ca` | Completed | 0 | |
| `mj-gs-v13-cl-joule-1e5-pd01` | Completed | 0 | |
| `mj-gs-v14-ca-base-hard` | Completed | 0 | |
| `mj-gs-v15-ca-base-hard-20k` | **Deleted** | 0 | Stopped ~iter 16200/20000 |
| `mj-gs-v15-ca-base-hard-20k-s2` | **Deleted** | 0 | Stopped ~iter 16200/20000 |
| `mj-gs-v16-short-ca-hs-joule-1e5` | **Completed** | 0 | Gate **PASS**; W&B `gr1hb5uh`, 500/500 iters |
| `mj-gs-v16-ca-hs-joule-1e5` | **Completed** | 0 | 2000/2000; W&B `shhm98rd`; walk gate PASS @ ~630 |
| `mj-gs-v17-all` | **Running** | 1 | iter **1956**/4000, reward **10.28**, ep_len **302.65** (kubectl ~05:22 UTC) |
| `mj-gs-v17-commands` | **Running** | 1 | iter **2025**/4000, reward **3.07**, ep_len **141.19** (kubectl ~05:22 UTC) |
| `mj-gs-v17-phasec` | **Pending** | 0 | Queue `mjlab-train`; 8 GPU cluster, 4 GPU/job |
| `mj-gs-v17-pushes` | **Pending** | 0 | Waiting for GPU |
| `mj-gs-v17-upright` | **Pending** | 0 | Waiting for GPU |
| `mjlab-gs-clock-anneal-joule-1e-4-pc-0-5-s1` | Running | 0 | Legacy v3-era; pod Completed |
| `mjlab-train` | Completed | 0 | Non-grid job |

Queue `mjlab-train`: v16 full holds **4×4090**; failed stale `mj-gs-v16-ca-hs-joule-1e5` deleted before re-apply.

---

## 8. v16 orchestration log (2026-07-03)

### Local validation (passed)

| Check | Result |
|-------|--------|
| `pytest` (nugus eval, config audit, mirror map, actuator friction, obs vector) | **34 passed**, 1 skipped |
| Phase 2 code | `mirror_map.py`, `NugusOnPolicyRunner`, `sim2sim_eval.py` present |
| Manifests | `BATCH=v16-short\|v16 ./scripts/k8s/gen-gridsearch.sh -o …`; `GIT_COMMIT` from configmap |

### Launched: v16-short @ `a1af0d4` (2026-07-03)

| Item | Value |
|------|-------|
| Commit | `a1af0d4fb8aef683f80f6fec9b6d4e63613d5ac0` on `add-phase-clock` (pushed) |
| ConfigMap | `mjlab-train-config` `GIT_COMMIT` updated + applied |
| Volcano job | `mj-gs-v16-short-ca-hs-joule-1e5` — **Completed** (recreated; first pod used stale `8581f7e`) |
| W&B run | `gr1hb5uh` — `clock_anneal__stand-0.15__pc-0.5__joule-1e-5__hs__s1__v16-short` / `nugus_gridsearch_v16-short` |
| Local tests | 48 passed, 1 skipped (`pytest` nugus suite) |
| Repo pin (follow-up) | `d91d439` — ConfigMap + manifest GIT_COMMIT tracked in git |

**v16 full queued** — gate **PASS** @ iter ~336; deleted failed `mj-gs-v16-ca-hs-joule-1e5`, `BATCH=v16 ./scripts/k8s/gen-gridsearch.sh -o scripts/k8s/gen_v16`, `kubectl apply -f scripts/k8s/gen_v16/` (2026-07-03 ~11:16 JST).

### Gate decision (2026-07-03 ~11:16 JST)

| Signal | Value |
|--------|-------|
| vcjob / pod | **Completed** (`mj-gs-v16-short-ca-hs-joule-1e5`, 500/500 iters) |
| Progress @ gate (~300) | **~336/500** — reward **~1.6–1.9**, ep_len **~78** |
| Mean reward trajectory | **−9.89 @ iter 4** → **−0.65 @ ~90** → **~0.8 @ ~291** → **26.75 @ iter 499** |
| Mean episode length @ finish | **851** steps @ iter 499 |
| NaNs / crash | **None** in logs |
| W&B | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/gr1hb5uh |
| Checkpoint (W&B) | `model_499.pt` + ONNX uploaded |

**Gate status:** **PASS** — reward trending up, stable training, reasonable ep length growth.

### Fixed eval (`nugus_eval`, `model_499.pt`, 2026-07-03)

Local run @ git `6fa6b4c` (+ `find_sites` id unpack fix): checkpoint from W&B `gr1hb5uh`, seed 7, 30s episodes. **No CUDA locally** — used `--envs-per-command 1` (10 envs total); default 2560-env eval did not finish on CPU in reasonable time. Metrics written to `/tmp/v16-short-eval.json`; W&B summary updated.

| Scope | `falls_per_min` |
|-------|-----------------|
| cmd `(0.5, 0, 0)` | **0.0** |
| cmd `(0, 0, 0)` | **0.0** |
| **overall** | **0.0** |

Other overall: `lin_vel_rmse` **0.33**, `ang_vel_rmse` **0.33**, `slip_vel` **0.032**, `swing_height_err` **0.077** (n=1 env/command — high variance; re-run at 256 envs/command on GPU for production numbers).

### v16 full launch (post-gate)

| Item | Value |
|------|-------|
| vcjob | `mj-gs-v16-ca-hs-joule-1e5` — **Completed** (`GIT_COMMIT=a1af0d4`, 96m wall) |
| Iters | 2000 (`MAX_ITERATIONS=2000`) |
| Experiment | `nugus_gridsearch_v16` |
| W&B run | `shhm98rd` — https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/shhm98rd |


### v16 full progress snapshots (kubectl)

| Time (UTC ~) | Iter | Mean reward | Ep length | fell_over | NaN/crash | Notes |
|--------------|------|-------------|-----------|-----------|-----------|-------|
| 2026-07-03 02:26 | **~175/2000** | **~0.1** | **~27** | ~309 | **None** | Early base; learning |
| 2026-07-03 02:46 | **~630/2000** | **~46.8** | **~959** | **~0.67** | **None** | **Walk gate PASS** — stable gait |
| 2026-07-03 03:04 | **~1008/2000** | **~39** | **~889** | **~1.3** | **None** | Still walking; post-iter-1000 curriculum stress |
| 2026-07-03 03:28 | **~1515/2000** | **~33** | **~889** | **~2.5** | **None** | Reward easing; ep_len high |
| 2026-07-03 03:51 | **1999/2000** | **3.08** | **165** | (rising) | **None** | **Completed** — vcjob `Completed`; checkpoint expected on W&B |

**Walk gate (~iter 1000):** **PASS** on training metrics (reward **>40**, ep_len **>900**, `fell_over` **<2** @ iter 630–960). Fixed-eval `nugus_eval.py` on `shhm98rd` **not run** locally (needs GPU/W&B artifact pull; prior `heading_command` cfg issue noted in §8 — retry when convenient).

### v17 decoupling grid (B1, 2026-07-03)

| Item | Value |
|------|-------|
| Trigger | v16 walk gate PASS @ ~630 + v16 full **Completed** 2000/2000 |
| Manifests | `BATCH=v17 ./scripts/k8s/gen-gridsearch.sh -o scripts/k8s/gen_v17` → 5 YAMLs |
| Applied | `kubectl apply -f scripts/k8s/gen_v17/` @ `a1af0d4` |
| Cells | `commands`, `pushes`, `upright`, `phasec`, `all` — 4000 iters, `CONT_BASE_STEP=48000` (iter 2000) |
| Cluster | **2× Running**, **3× Pending** (8 GPU cluster, 4 GPU/job) |

**W&B runs (live):**

| Cell | Run ID | URL |
|------|--------|-----|
| all | `ts21daeb` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ts21daeb |
| commands | `9nzuqlnr` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/9nzuqlnr |
| phasec / pushes / upright | — | Pending launch |

**Progress snapshots (kubectl, UTC ~):**

| Time | Cell | Iter | Mean reward | ep_len | fell_over/ep | Notes |
|------|------|------|-------------|--------|--------------|-------|
| 04:05 | all | 266 | 0.76 | — | ~230 | Early base |
| 04:05 | commands | 273 | 0.58 | — | ~228 | Early base |
| 04:10 | all | 384 | **28.49** | — | — | Walk-gate region; reward climbing |
| 04:10 | commands | 393 | **8.54** | — | — | Slower lift vs all |
| 05:22 | all | 1956 | **10.28** | 302.65 | ~21 | Pre-CONT_BASE; degraded from peak |
| 05:22 | commands | 2025 | **3.07** | 141.19 | ~33 | **Past CONT_BASE** (cmd widening active) |

Pods: `mj-gs-v17-all-train-0` (munin), `mj-gs-v17-commands-train-0` (hugin). Both show the familiar base→pre-hard degradation (peak ~28 @ iter ~384, collapse to single digits by iter ~2000). `commands` crossed CONT_BASE first with **higher** falls (~33 vs ~21) and **lower** reward (~3 vs ~10) — command widening alone may already be destabilizing, but B1 requires fixed eval (`nugus_eval` @ 0.75 vs 0.3 m/s) on all 5 checkpoints. **v18 not queued.**

**Pending:** `nugus_eval` per cell after all 5 complete; destabilizer call deferred.

---

## 6. Data availability notes

| Source | Status |
|--------|--------|
| W&B API | **Available** for batches v3, v5, v7–v15 (tag `batch-v*`). Final summaries used throughout §2–3. |
| W&B API | **No runs** for v4, v6 (never completed / not launched). |
| W&B history (per-iter curves) | **Not retrieved** — pandas unavailable in fetch environment; iteration snapshots from **kubectl logs** where cited. |
| kubectl logs | **Available** for vcjobs still within TTL (~24h after finish) and all running pods. |
| v8–v11 pods | **Expired** (TTL); metrics from W&B only. |

---

## 7. Quick reference — W&B URLs (batches v8–v15)

| Batch | Run | URL |
|-------|-----|-----|
| v8 | cur0 `57gr0muv` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/57gr0muv |
| v8 | cur1 `v21878z8` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/v21878z8 |
| v9 | cur0 `ift9sd2w` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ift9sd2w |
| v9 | cur1 `ffcdmlkz` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ffcdmlkz |
| v9 | ca `xm5t9ilu` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/xm5t9ilu |
| v10 | hs-critic `yz5baxda` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/yz5baxda |
| v10 | hard-cont (finished) `cxl0l9d8` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/cxl0l9d8 |
| v11 | seeds `16cbg6lm`, `u5mbohzy`, `4jx3q9es`, `px06ulu0` | /runs/16cbg6lm … /runs/px06ulu0 |
| v12 | pd-tail −0.2 `lyhwmnll` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/lyhwmnll |
| v12 | pd-tail −0.1 `260z9ekp` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/260z9ekp |
| v13 | joule 1e-5 `l9wok1ss` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/l9wok1ss |
| v13 | ca `ojozkbfs` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ojozkbfs |
| v14 | `jyksw3mg` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/jyksw3mg |
| v15 | s1 `ynquy630` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/ynquy630 |
| v15 | s2 `rntq7onj` | https://wandb.ai/vincenttumm-the-university-of-newcastle/mjlab/runs/rntq7onj |
