# Track D — Close the hardware loop

No policy has ever run on the physical NUgus and no hardware log exists in
the repo. Until one does, every DR range and latency estimate is
unconstrained guesswork. **D2's plumbing should start now, in parallel with
all sim work — it needs zero GPU time and unblocks A2/A4 fitting.**

---

## D1 — Sim-to-sim validation gate

**What:** run the exported ONNX policy in *plain* MuJoCo (not Warp) with the
same MJCF + the A1/A2 actuator parameters, and compute the E0.1 eval metrics.
This is the standard robustness gate (Humanoid-Gym popularized Isaac→MuJoCo;
ours is Warp→vanilla-MuJoCo, catching solver/integration discrepancies plus
any train-time-only obs processing).

**Deliverable:** `scripts/sim2sim_eval.py`:
- Load ONNX (exported by `MjlabOnPolicyRunner`; verify the export bundles
  obs normalization — if not, export/apply the normalizer stats manually).
- Rebuild the actor observation vector from raw `mujoco.MjData` in the SAME
  order as training (source the term list from the env cfg, don't hand-copy;
  `tests/test_nugus_observation_vector.py` documents the layout).
- Reproduce: 50 Hz control / 200 Hz physics, action scale ≈0.247 +
  default-offset mapping, gait clock from wall time, sensor delays via a
  ring buffer.
- Run the E0.1 command grid; emit the same JSON metrics.

**Gate:** a policy whose falls_per_min degrades >2× from Warp to vanilla
MuJoCo does not go to hardware.

**Cost:** low-medium (obs reconstruction is the fiddly part — reuse it for
D2's onboard runtime).

---

## D2 — First deployment + data capture (highest-value item in the plan)

**What:** deploy the best stable policy (v13's final checkpoint today; the
v17/v18 winner once available) at conservative commands, harnessed/tethered,
and capture logs. **A failed walk with good logs is a success** — the logs
constrain A1/A2/A4 and every DR range.

### Deployment runtime (NUbots side)

- ONNX inference in the NUClear pipeline at 50 Hz; actor obs assembled
  exactly as in D1 (share that code).
- Gait clock: wall-clock phase at period 0.7 s, sin/cos; zero it when
  commanded to stand (matching the training-time standstill masking —
  read `gait_clock` in the task's observations module for exact semantics).
- Actions → position targets: `default_pose + 0.247 × action` per joint,
  clamped to soft limits (0.9 × range). Servos in position mode with the
  same kp the sim models (Dynamixel P-gain register equivalent of
  kp≈56/31 — convert units carefully: MuJoCo kp is Nm/rad; the Dynamixel
  register is a dimensionless firmware gain. Derive the conversion from the
  servo's stall torque and register scaling, then VERIFY on the bench with
  a step response comparing sim vs real rise time — mismatched effective kp
  is the classic silent transfer-killer on position-controlled servos).
- Safety: torque-off on |tilt| > 50° (matches training termination), on
  command, and on comms dropout. Human with kill switch. Start suspended,
  then supported standing, then unsupported at (0.3, 0, 0).

### Logging spec (50 Hz minimum, one file per trial)

| Field | Purpose |
|---|---|
| t (monotonic, per pipeline stage) | latency fitting (A4) |
| q_target[20] (policy output, post-clamp) | replay input (A1/A2) |
| q_meas[20], qd_meas[20] | replay target; friction/velocity-sat fitting |
| present_current[20] | torque estimate; effort-limit realism; CURRENT_OBS validation |
| IMU gyro, accel, orientation quat | obs-noise validation; fall analysis |
| policy obs vector as fed (post-noise/delay handling) | reproduce inference offline exactly |
| commanded twist | eval bookkeeping |
| servo error/status registers, bus voltage | brownout/overload detection |

### Trial protocol

1. Bench: single leg, sinusoid + logged-policy trajectories → A1/A2 replay
   fitting.
2. Suspended full robot: run policy in air; verify no oscillation, sane
   ranges, correct clock behavior at stand.
3. Supported stand → unsupported stand (command (0,0,0)); watch the
   stand-still masking behaves.
4. Walk at (0.3,0,0) × 30 s × ≥5 trials; then (0.5,0,0), then yaw.
5. After every session: replay logs through `scripts/actuator_replay.py`
   (A1) and the latency fitter (A4); update sim parameters; note deltas in
   the experiments doc.

**Signal:** trial survival time and lin-vel tracking vs the D1 sim-to-sim
numbers — the gap between those two IS the remaining sim-to-real gap and
becomes the plan's steering metric.

---

## D3 — SILENCE_CLOCK (deprioritized; conditional)

The `SILENCE_CLOCK` variant knob (fade the clock observation to zero on the
anneal schedule, so the policy can't depend on a phase signal at deployment)
exists but has never been run (v8–v15 all use the clock obs). It matters
ONLY if the robot software will not provide a gait clock at 50 Hz. It can
(it's software — D2 provides it trivially), so: skip, unless a clock-free
policy becomes a goal in itself. If run: single cell on the winning Track-B
base, compare fixed-eval stability; expect some loss (the clock is
information); decide if clock-free operation is worth it.
