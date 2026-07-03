# Track A — Actuator & DR fidelity

The literature is unambiguous that on hobby-servo platforms actuator fidelity
is the make-or-break sim-to-real factor (see
`references/bam-actuator-models.md`, `references/systematic-sim2real-pace.md`,
`references/small-humanoid-precedents.md`). E0.2 (Phase 0) fixed the
correctness bugs; this track raises fidelity.

Ordering: A1 can land with Phase 0 (it refines E0.2b's baselines). A4 needs
robot access but no GPU. A2 needs a bench rig. A5 is a training-time
comparison arm.

---

## A1 — BAM-lite actuator parameters

**What:** replace guessed friction/damping baselines with Rhoban's published
BAM-identified parameters for the MX-64 and MX-106 (the exact servos in the
NUgus arms/head and hip-yaw), and interpolate a starting point for the
XH540-W270. Then re-center DR: scale ranges ±25% around identified values
instead of wide guesses.

**How:**
1. Clone https://github.com/Rhoban/bam. Identified parameter JSONs live in
   the repo per actuator/model (M1 Coulomb-viscous … M6 extended). Extract
   for MX-64 and MX-106: Coulomb friction, viscous damping, armature/rotor
   inertia, and (for M5/M6) the load-dependent and Stribeck terms.
2. Map to the mjlab actuator cfg: Coulomb → `frictionloss`, viscous →
   passive joint `damping`, rotor inertia → `armature` (compare with the
   current 0.0266/0.01195 values — if BAM's differ substantially, prefer
   BAM's and note the change). MuJoCo's dof friction is Coulomb-only, so the
   Stribeck/load-dependent extensions cannot be expressed directly — see
   step 4.
3. XH540-W270: no published BAM params. Start from MX-106 values scaled by
   gear ratio ratio (W270 = 270.4:1 vs MX-106 225:1) and flag PROVISIONAL;
   A2 replaces this.
4. Optional stretch (only if replay error in verification stays poor): BAM's
   extended models can be approximated in training by *randomizing*
   frictionloss over the range the load-dependent model spans across the
   gait's load profile. Compute that range from the BAM model equations at
   representative loads (±effort/2); use it as the DR range instead of ±25%.

**Verification (replay error — this is the money metric):** extend the E0.2
hanging-leg test into `scripts/actuator_replay.py`: drive one simulated joint
with a logged position-target trajectory and compare simulated vs (later)
real joint response. Until hardware logs exist (D2), compare old model vs
BAM model on a chirp: expect visibly slower, lag-ier tracking. After D2,
compute MAE(sim, real) per joint — BAM should cut it ~2× vs Coulomb-viscous
per the paper; if it doesn't, our mapping is wrong.

**Cost:** low-medium (no bench). **Depends:** E0.2. **Conflicts:** none.

---

## A2 — Bench identification of our actual servos

**What:** run the BAM identification protocol (pendulum rig + logged
trajectories, documented in their repo/paper) on at least one MX-64, MX-106,
and XH540-W270 from the actual robots — capturing our units' wear, voltage,
and firmware mode.

**Why not skip:** A1 borrows Rhoban's units. The systematic-sim2real study
found firmware mode changes effective dynamics (identify and deploy in the
SAME mode — for Dynamixels: same control mode, same return-delay-time, same
voltage as in-game). Wear on RoboCup servos is significant.

**Output:** per-class identified {frictionloss, damping, armature}; replace
A1 values; narrow DR to ±15% around them.

**Cost:** medium (bench time, a few days). **Depends:** A1 harness for
comparison; robot hardware. **Signal:** replay MAE on held-out trajectories.

---

## A4 — Measure and match real control-chain latency

**What:** measure the actual command→motion and sense→policy latencies on the
NUgus software stack (NUClear reactors + Dynamixel bus), then set sim delay
ranges around measurements instead of guesses.

Current sim guesses: actuator command delay 1–3 physics steps (5–15 ms,
`nugus_constants.py`); obs delays gyro/gravity 0–40 ms, joint pos/vel
20–60 ms (`env_cfgs.py` obs section).

**How (robot side):**
1. Command latency: at a known tick, step a joint target by a small delta;
   record bus timestamps of the command write and the first present-position
   sample that moves (and/or present-current spike). Repeat ~100×, take the
   distribution. The systematic study found ~7.5 ms typical for pro
   hardware; Dynamixel serial chains are usually worse — expect 10–30 ms,
   possibly asymmetric across the chain order.
2. Observation age: timestamp each pipeline stage (servo read, IMU read,
   filter output, policy input) for the same tick; compute age of data at
   policy execution.
3. Jitter: record the distribution, not just the mean — sim delay ranges
   should span p5–p95.

**Sim change:** set `delay_min_lag`/`delay_max_lag` (units: 200 Hz physics
steps, i.e. 5 ms each) and the obs-delay ranges to bracket the measured
p5–p95. If measured command latency exceeds 15 ms, today's sim range is too
optimistic — this alone can break transfer (see
`references/reward-curriculum-symmetry.md`, delay section).

**Cost:** low, robot-side only. **Depends:** nothing (do anytime).

---

## A5 — Torque-space perturbation injection (comparison arm)

**What:** during training, add filtered random torque noise per joint
(Ornstein-Uhlenbeck or low-pass-filtered white noise, magnitude a few % of
effort limit) to emulate unmodeled actuator dynamics. From arXiv 2504.06585
(see `references/reward-curriculum-symmetry.md`).

**How:** implement as an interval/perpetual event applying `qfrc_applied` on
motor dofs, or as an actuator-level additive term. Parameters: correlation
time ~50–100 ms; magnitude sweep {2%, 5%} of per-joint effort.

**Run as a comparison, not stacked by default:** cells {A1 only, A5 only,
A1+A5} on the v16 base, 2k iters each. It partially duplicates what A1/A2
model explicitly; the question is whether it adds robustness beyond them.

**Signal:** fixed-eval falls_per_min and (after D2) hardware tracking error.
**Cost:** low-medium. **Depends:** v16 base; independent of A1 mapping
correctness (that's its virtue).
