# v16 collapse analysis (2026-07-03) and re-baseline spec (v16b)

Post-mortem of the first batch on the Phase-0 physics. Supersedes the v16/v17
launch sequence in `03-track-b-curriculum.md` until a stable base exists.

## What happened

| Run | Result |
|---|---|
| v16-short (`gr1hb5uh`, 500 it) | rw 26.7, ep 851, **fell 3.54/ep** (v13 @ ~436 it: fell 0.54) |
| v16 full (`shhm98rd`, 2000 it) | peaked rw 43.6 / fell 0.71 @ it 570 → dip at p2 (it≈1000) → partial recovery 32.8 @ 1477 → **collapse from p3 (it≈1700): rw 3.1, ep_len 165, fell 43.8/ep at end** |
| v17 hard-commands / hard-all | launched on this base — results meaningless; killed |

## Per-term decomposition of the v16 full run (sampled from W&B)

```
it    rw     ep_len fell  trkLin  upright swing  joule    torq_rate act_rate
 570  43.6   956    0.71  1.04    0.455   0.455  0        0         -0.155
1054   9.7   849    1.92  1.03    0.421   0.105  -2.6e-4  -0.020    -0.084
1477  32.8   799    2.33  0.90    0.396   0.102  -3.3e-4  -0.013    -0.042
1751  21.1   588    5.21  0.48    0.295   0      -3.2e-4  -0.011    -0.033
1990   4.7   199   38.9   0.13    0.081   0      -0.8e-4  -0.003    -0.013

Metrics/peak_height_mean: 0.003 @ it 510 → 0.012 @ 1577 → 0.029 @ 1694
(target: 0.08 m)
```

## Findings

**V1 — The energy/joule hypothesis is acquitted.** Phase-C penalty
magnitudes are negligible in the reward sum (joule ~−3e-4, torque_rate
~−0.02, both ≪ the ±1.0-scale task terms). The joule term is not the
problem. Keeping `JOULE_W=1e-5` stands.

**V2 — The swing target is physically infeasible on the new actuators.**
Achieved peak swing height is 0.3–2.9 cm against the 0.08 m target. With
0.7 s period × 0.45 swing ratio, half-swing is ~0.16 s; lifting a foot 8 cm
needs ~0.9 rad of knee travel → ~5.5 rad/s mean knee velocity, while the
XH540 legs are now (correctly) capped at ~3.2 rad/s no-load with torque
derating on the way there. The policy physically cannot perform the
commanded gait; it shuffles at ~1 cm clearance instead.

**V3 — The clock_anneal→0 handoff is the collapse trigger.** While the
swing reward exists, the shuffle-gait equilibrium is held together by it
(dips at p1/p2 stage boundaries, visible above at it≈1000). At p3
(it≈1700) the weight hits 0: nothing rewards foot lift at all, clearance
degrades, trips cascade (fell 5.2 → 38.9). Note v9's clock_anneal baseline
collapsed post-p3 on the OLD physics too (peak 59 → 14.5) — the
anneal-to-zero design was always fragile; harder physics made it fatal.
Published systems do not do this: the in-repo G1 velocity task, Humanoid-Gym,
Berkeley Humanoid, and unitree_rl_gym all keep their gait-shaping rewards
(air time / periodic force-velocity) **permanent**; curricula ramp
terrain/commands, never the reward stack.

**V4 — The new physics/DR makes the base task ~4× fallier but workable**
(fell 0.71 vs 0.17 at comparable iters, tracking fine). The base
degradation is tolerable; contributors to trim while stabilizing: payload
range (−0.3,+0.5) kg is ±27% of the 1.833 kg torso.

**V5 — The XH540 velocity limit was set ~25–35% too low (units confirmed
correct).** Units first: `velocity_limit` in `DcMotorActuator` is compared
directly against MuJoCo joint velocity, i.e. **rad/s** — the G1 config's
values of 20–37 are genuine rad/s for Unitree's quasi-direct-drive
actuators (~10× faster than Dynamixels), NOT rpm. For the XH540-W270 the
correct no-load speed is **39–46 rpm (12 V → 14.8 V) = 4.1–4.8 rad/s**;
Phase 0 used 3.2 (a 30 rpm figure — wrong). Fix: set the limit from the
measured bus voltage (4S LiPo ≈ 14.8 V nominal → ~4.8 rad/s XH540; scale
MX-106/MX-64 similarly from their e-Manual rows). Even at 4.8 rad/s the
0.7 s / 0.08 m gait (~5.5 rad/s mean knee) remains infeasible — V2's
conclusion stands, only less severely.

Also VERIFY the torque-speed anchor: Phase 0 set
`saturation_effort = effort_limit`, which makes available torque decay
linearly from the rated limit at zero speed — pessimistic in mid-range.
The DC-motor model wants `saturation_effort` = STALL torque at bus voltage
and `effort_limit` = the rated/firmware torque cap (look both up on the
ROBOTIS e-Manual per servo; stall is typically noticeably above rated).

Best calibration for all of this: log joint-velocity/torque profiles from
the EXISTING classical walk engine on the real robot — ground truth for
what the servos actually deliver, and it bounds the feasible gait envelope
directly.

**V6 — The v16-short "gate PASS" was under-powered** (nugus_eval run on CPU
with 1 env per command). Gate decisions need the full eval spec (≥256
envs/command) on cluster checkpoints.

## Re-baseline spec: v16b

Single change-set, then re-run the 2000-iter base:

1. **Variant `clock_persist`** (exists, never run in v8+): permanent swing
   reward 0.75, air_time 0.08, no anneal, no landing handoff. Alternative
   if persist misbehaves: clock_anneal with a floor of 0.25 instead of 0.
   (Variable gait timing — the original motivation for removing the clock —
   is NOT abandoned; the supported path is `07-gait-timing-strategy.md`,
   which layers commanded-period variation on top of this stable base.)
2. **Corrected velocity limits first (V5):** XH540 → 4.1–4.8 rad/s per
   measured bus voltage; rescale MX-106/MX-64 from e-Manual; revisit
   `saturation_effort` = stall torque. THEN feasible gait target:
   `GAIT_PERIOD=1.0`, swing `target_height=0.05`; revisit 0.06–0.08 only
   after saturation telemetry (item 3) shows headroom. Feasibility rule of
   thumb: required mean swing joint velocity ≤ ~60% of no-load limit.
3. **Saturation telemetry:** add `Metrics/vel_sat_frac_<group>` — fraction
   of steps with |qd| > 0.9 × velocity_limit per actuator group — so
   feasibility is measured, not inferred.
4. **Trim DR while re-baselining:** payload (−0.2,+0.2) kg, link_mass
   (0.90,1.10). Re-widen after a stable base exists (Track A re-centering).
5. **Flatten Phase-C:** set joule 1e-5 / joint_acc −1e-4 / torque_rate
   −1e-3 / soft_landing −0.01 as PERMANENT weights from it 0 and delete the
   Phase-C ramp for this variant (V1 shows they're affordable; the staged
   ramp only adds nonstationarity). base_height +0.3 also permanent.
6. Everything else per v16 (hs-critic, joule 1e-5, stand 0.15, seed 1;
   optionally seed 2).

**Success signal:** fixed eval (proper, cluster) falls_per_min at (0.5,0,0)
within 2× of a v13 checkpoint evaluated under the SAME eval build, and
`peak_height_mean` ≥ 0.7 × target. Then relaunch the v17 decoupling grid on
the v16b base.

## Where to copy working curricula/weights from (all public)

- **In-repo:** `src/mjlab/tasks/velocity/config/g1/env_cfgs.py` — same
  framework, tuned by upstream; permanent rewards, terrain+command curricula
  only. The closest directly-diffable reference.
- **Humanoid-Gym** (github.com/roboterax/humanoid-gym) — XBot-S/L reward
  scales and DR ranges in `humanoid/envs/custom/humanoid_config.py`;
  sim2real-proven; periodic gait rewards permanent.
- **unitree_rl_gym** (github.com/unitreerobotics/unitree_rl_gym) — G1/H1/Go2
  configs.
- **Berkeley Humanoid** (github.com/HybridRobotics/isaac_berkeley_humanoid)
  — mid-size (85 cm) humanoid IsaacLab config; closest size class with
  public config.
- **Booster Gym** (github.com/BoosterRobotics/booster_gym) — T1 full
  YAML: rewards, DR, push schedule.
- Common pattern across all: **fixed reward stack for the whole run;
  difficulty comes from terrain/command/push curricula; energy and
  smoothness penalties small and constant.** None anneal gait shaping to
  zero mid-run.
