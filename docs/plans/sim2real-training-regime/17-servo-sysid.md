# 17 — Servo system identification from hardware walking data

**Data**: `sensor_data_moving.json` (763 MB NUClear log, 2026-06-03):
25,675 paired `RawSensors`/`Sensors` records, 293 s of walking at ~91 Hz.
`RawSensors.servo.*` carries `presentCurrent` (A), `presentPWM` (raw
ticks, ±885 = ±100% duty), `presentPosition` (rad, convention-corrected),
`presentVelocity`, per-servo `voltage` (bus sag 15.4→11.4 V) and
temperature. Extracted arrays cached at
`~/.claude/jobs/6f7aa84e/tmp/sysid_raw.npz`.

## Three data bugs in the NUbots sensor pipeline

1. **`presentVelocity` is rev/s, not rad/s.** `Convert.cpp:147` does
   `raw * 0.229 / 60` (0.229 rpm/tick, /60 → rev/s) and nothing
   multiplies by 2π. Verified against the position derivative:
   amplitude ratio 0.13–0.145 ≈ 1/2π across all moving joints.
2. **Side-dependent sign flip.** corr(presentVelocity, dpos/dt) is
   −0.99 for {rKnee, lHipPitch, rAnklePitch, rAnkleRoll, lAnkleRoll,
   rShoulderPitch, rHipYaw, lHipYaw} and +0.99 for their mirrors: the
   position converter applies the per-servo direction map, the velocity
   converter does not. Any consumer mixing the two gets mirrored
   velocities on half the joints.
3. **~55 ms latency** on `presentVelocity` relative to the position
   derivative (Dynamixel windowed velocity estimate + bus timing; arms
   read at ~33 ms, hip yaw ~44 ms — bus scheduling order visible). A
   deployed policy consuming this register gets a five-control-step-old
   velocity.

**Deployment implication**: derive joint velocity from position on the
robot (as the sim policy's obs pipeline effectively does) or fix the
conversion and accept the lag; do not feed the raw register.

## Fits (swing-phase, raw-frame, position-derived ω)

Electrical `duty·V = R·I + K·ω` per joint, then mechanical
`K·I = J·ω̇ + b·ω + tc·sign(ω)` on swing-phase samples (foot up,
|ω| > 0.3). Pooled over the 10/12 leg joints passing quality gates
(elec R² > 0.9, mech R² > 0.4):

| param | value | quality |
|---|---|---|
| R | 2.25 ± 0.36 Ω | strong (elec R² 0.85–0.99) |
| K (back-EMF) | 2.68 ± 0.18 N·m/A | strong; spec "2.0" is stall-derived (gear losses baked in) |
| J (reflected) | 0.0496 ± 0.0108 kg·m² | robust across joints (0.034–0.069) |
| b (viscous) | 0.47 ± 0.66 N·m·s/rad | poorly identified — swing gravity unmodeled |
| tc (Coulomb) | −0.31 ± 0.78 N·m | poorly identified — same confound |

The earlier external fit (R 2.73, K 1.88, J 0.0346, b 0.304, tc −0.131)
used `presentVelocity·2π` as ω — attenuated (~0.87×), lagged 55 ms, and
sign-mixed per bug 2 — which biases K down and J low; its K "matching
the stall spec" matches the wrong constant (stall K_t embeds gear
efficiency; the electrical equation identifies back-EMF K_e).

## Sim gaps found and actions taken

- **`ARMATURE_XH540` was 0.0266 (the MX-106 value, copied) vs measured
  0.0496 — 1.9× low, outside the ±20% DR band.** No training env ever
  saw realistic leg inertia. → corrected to 0.0496 (commit with this
  doc); per-joint spread 0.034–0.069 sits inside DR ±20% around it.
- `_NUGUS_CURRENT_KT` legs 2.0 → 2.68 (observation model should use the
  electrical constant; present-current reads electrical amps).
- `FRICTIONLOSS_XH540`/`VISCOUS_DAMPING_XH540` (0.124/0.042, gear-scaled
  from Rhoban BAM MX-106) stay: b/tc are not identified well enough
  from this log to overrule them. A bench log (single leg unloaded,
  chirp/steps) would identify them cleanly if wanted.
- Current-obs plumbing already existed (`CURRENT_OBS=1`: τ/K_t with
  2.69 mA quantization, noise+bias, per-servo gain/offset DR event) —
  v47 enables it on the corrected model.

## Sanity cross-checks

- τ_max = K·I_limit = 2.68 × 5.22 A (goalCurrent register 1941) ≈ 14
  N·m vs spec stall 11.7 — consistent once gear losses are counted.
- Walking knee current peaks 4.5 A vs 4.9 A stall rating: the real
  robot walks near its current budget; the joule/λ-live watts budget
  (R10) has hardware numbers to anchor to now.
- Hottest servo: rKnee 47 °C after 5 min of walking (also the joint
  with the worst mechanical fit — load-heaviest).

## Observation-noise calibration (added same day, user prompt)

High-frequency residual (signal minus 11-tap Hanning smooth) across the
full walking log — i.e., what a policy's input actually carries on
hardware, including structure-borne vibration the rigid-body sim never
generates:

| channel | sim (before) | measured walking hf | sim (after) |
|---|---|---|---|
| gyro (rad/s) | 0.02/0.03/0.03 ("×10 safety") | 0.130/0.163/0.156, kurt 8–26 | 0.13/0.16/0.16 |
| projected gravity | 3.9e-3/4.3e-3/5.9e-4 | 4.6e-3/3.4e-3/3.0e-3 (fused Htw) | 5e-3/4e-3/3e-3 |
| joint pos (rad) | 0.01 | 0.0037 | 0.01 (kept, margin) |
| joint vel (rad/s) | 0.05 | 0.14–0.21 (position-derivative @ 91 Hz) | 0.15 |
| current (A) | 0.05 white + 0.02 bias | 0.25–0.54 legs / 0.12–0.29 arms, kurt to 24; offsets ±0.095 | 0.25 white; offset DR ±0.1 already covered |
| accel (m/s²) | (not an actor obs) | 1.27/1.39/1.85, kurt 5–42 | n/a |

Lesson: bench-static noise measurements under-read walking reality ~5×
even after "×10 safety" inflation, because the dominant term is
vibration, not sensor floor. Tails are heavy everywhere; Gaussian at
matched std is the available first-order match. If a future run shows
noise-related brittleness, the next step is spike/burst noise (matched
kurtosis), not bigger Gaussians.
