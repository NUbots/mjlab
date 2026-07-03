# Phase 0 — Foundations (the "v16 base")

Everything here must land **before** any comparative experiment, because
E0.2/A3 change the physics and C1 changes the critic. Together these define
the new base config all later batches build on. Work through in order; each
item has its own verification.

Reference for every file path: `references/codebase-map.md`.

---

## E0.1 Fixed evaluation harness

**Problem:** reward curves are not comparable across curriculum stages or
physics changes (finding F1). All experiment comparisons in this plan use
these eval metrics, never `Train/mean_reward`.

**Deliverable:** `scripts/nugus_eval.py` — loads a checkpoint, runs frozen
eval conditions, prints/logs metrics.

### Specification

- Build the env exactly as training does (same task id, same variant env
  vars) but with:
  - fixed seed (`--seed`, default 7),
  - episode length 30 s,
  - command resampling disabled — commands set explicitly per env group,
  - pushes at the standard (×1.0) interval/magnitude,
  - DR enabled but deterministic via the fixed seed (we want the DR
    *distribution*, held constant across evaluations).
- Command grid, one group of envs per command (≥256 envs per command):
  `(0.3,0,0) (0.5,0,0) (0.75,0,0) (-0.3,0,0) (0,0.3,0) (0,-0.3,0)
  (0,0,0.5) (0,0,-0.5) (0,0,0) (0.5,0.3,0.5)`
- Metrics (computed from sim state, NOT reward terms):
  - `eval/lin_vel_rmse` — RMS of ‖cmd_xy − v_xy‖ over non-fallen steps
  - `eval/ang_vel_rmse` — RMS of |cmd_z − ω_z|
  - `eval/falls_per_min` — fell_over terminations / simulated minutes
  - `eval/mean_ep_len_s`
  - `eval/slip_vel` — mean foot xy speed while in contact
  - `eval/swing_height_err` — |peak swing height − 0.08| mean
  - per-command breakdown of the above (falls at (0.75,0,0) vs (0,0,0) is
    the diagnostic split for track B)
- Output: JSON to stdout/file, and `wandb` summary update under `Eval/*` when
  given `--wandb-run-path entity/project/run_id` (checkpoints are saved every
  250 iters; evaluating the last 3 checkpoints of a run gives a
  late-training trend).

### Implementation notes

- Follow the existing play-mode pattern: `env_cfgs.py` already has a play
  branch that disables pushes and extends episodes — do NOT reuse it
  verbatim (it removes pushes; eval keeps them).
- Command override: the command term is `UniformVelocityCommand`
  (`src/mjlab/tasks/velocity/mdp/velocity_command.py`). Simplest override:
  set `resampling_time_range=(1e9, 1e9)` and write the command tensor
  directly after reset; check how `rel_standing_envs` masks zero commands so
  the (0,0,0) group is treated as standing.
- Policy loading: reuse the runner's checkpoint load
  (`src/mjlab/rl/runner.py`, `MjlabOnPolicyRunner`) or the exported ONNX.
  Torch checkpoint is easier (obs normalizer state comes with it).

### Verification

- Run twice with the same checkpoint/seed → identical metrics.
- Run on a v13 checkpoint and a v15 checkpoint: v15 should show clearly
  higher `falls_per_min` (matches F1); sanity-checks the harness.

---

## E0.2 Servo model correctness

### E0.2a Velocity saturation (gap G1 — highest-impact single fix)

**Problem:** `BuiltinPositionActuatorCfg(velocity_limit=30)` in
`src/mjlab/asset_zoo/robots/nugus/nugus_constants.py` (3 sites, all marked
TODO) is dead config: `BuiltinPositionActuator`
(`src/mjlab/actuator/builtin_actuator.py`) never reads `velocity_limit`.
Only `DcMotorActuator` (`src/mjlab/actuator/dc_actuator.py`) implements a
torque-speed curve (`velocity_limit_motor`, `_vel_at_effort_lim`). The sim
therefore allows unbounded joint speeds; real servos saturate hard.

**Change (preferred):** migrate the three NUgus actuator configs from
`BuiltinPositionActuatorCfg` to the `DcMotorActuatorCfg`/`IdealPdActuatorCfg`
family, keeping identical kp/kd/effort/armature and adding real velocity
limits:

| Servo | Joints | kp | kd | effort (Nm) | armature | velocity_limit (rad/s) |
|---|---|---|---|---|---|---|
| MX-106 | hip_yaw | 56.052 | 1.6548 | 11.086 | 0.0266 | ~4.7 (45 rpm @12V) VERIFY |
| XH540-W270 | hip_roll/pitch, knee, ankle | 56.052 | 1.6548 | 11.086 | 0.0266 | ~3.2 (30 rpm @12V) VERIFY |
| MX-64 | shoulders, elbows, neck, head | 31.1558 | 0.6782 | 6.1621 | 0.01195 | ~6.6 (63 rpm @12V) VERIFY |

VERIFY means: check the ROBOTIS e-Manual datasheet for the exact part number
and supply voltage NUgus uses (values above are 12 V no-load; under load the
torque-speed line means less). If NUgus runs a higher bus voltage, scale
accordingly. The DC-motor model wants the no-load speed as `velocity_limit`;
it derives available torque linearly between stall and no-load.

**Fallback if DcMotor migration destabilizes sim:** `BuiltinPositionActuator`
uses MuJoCo's implicit actuator integration (stable with stiff gains at
200 Hz); an explicit ideal-PD torque computation may need smaller timesteps
or gain retuning. If instability appears (NaNs, chattering at rest), the
fallback is to keep the builtin actuator and add velocity-dependent effort
clamping: scale each joint's forcerange by
`clamp(1 - |qd|/v_noload, 0, 1)` per step (an interval/perpetual event or a
small actuator subclass). Less principled, much less invasive.

**Also:** keep `delay_min_lag=1, delay_max_lag=3` (5–15 ms actuator delay)
in the migrated configs — do not lose it in the swap. And delete the
`velocity_limit=30  # TODO` comments either way (they misleadingly imply the
value is used).

### E0.2b Nonzero joint friction baselines (gap G2)

**Problem:** the DR event `joint_friction` (`env_cfgs.py`, event section;
implementation `src/mjlab/envs/mdp/dr/joint.py:88`, targets
`dof_frictionloss`) scales a baseline of 0 → silent no-op, and the sim has
zero Coulomb friction on friction-dominated hobby gearboxes.

**Change:**
1. Set `frictionloss` in each actuator cfg (the builtin/ideal-PD cfgs both
   plumb it to the joint) using BAM-identified Coulomb terms as baselines.
   Get exact numbers from the BAM repo params
   (see `references/bam-actuator-models.md`, "Getting the parameters");
   expected order of magnitude 0.1–0.4 Nm for MX-class servos. Mark the
   chosen values with a comment citing the BAM param file used.
2. Widen the `joint_friction` DR scale range from (0.8, 1.2) to **(0.5, 1.5)**
   until bench identification (A2) narrows it — the baseline is borrowed
   from Rhoban's units, not ours.
3. Add a `joint_damping` DR term (helper exists: `dr.joint_damping`) scale
   (0.8, 1.2) — viscous friction is the other BAM friction component; give
   it a small nonzero baseline from the same BAM params (distinct from the
   PD `damping` gain — this is the passive joint field).

**Solver headroom:** frictionloss adds a constraint per DOF. The flat env
trims `njmax=300` (`env_cfgs.py`, flat overrides). 20 extra friction
constraints per robot may not fit; if MuJoCo warns/overflows, raise `njmax`
and re-benchmark envs-per-GPU (8192 was the RTX 4090 sweet spot; it may
drop).

### Verification for E0.2

- Unit test: build the NUgus articulation, assert
  `model.dof_frictionloss[motor_dofs] > 0` and that the DR event changes it
  across two resets.
- Behavior test: track a 2 rad/s sinusoidal position target on a hanging leg
  joint in sim; with saturation the tracked amplitude must lag/attenuate vs
  the old model. (This becomes the replay-error harness for A1/A2 later.)
- Training smoke: 300-iter run; expect slower early progress than pre-fix
  runs — reward drop vs old curves is EXPECTED and is the fix working. Only
  investigate if it fails to walk at all by iter ~1000 in a 2k run.

---

## A3 Mass and payload randomization (gap G4)

**Change** in the NUgus events section of `env_cfgs.py` (next to `base_com`):

- `link_mass`: `dr.body_mass`, mode reset, operation scale, uniform
  (0.85, 1.15), on major bodies (torso, thighs, shanks, feet, arms — use a
  body-name regex; skip the tiny sub-links if any).
- `payload`: `dr.body_mass`, mode reset, operation add, uniform
  (−0.3, +0.5) kg on torso only (battery state, cabling, future compute).

CTS and most successful recipes use ±20% link mass; we start ±15% because
the CAD masses are believed decent. `dr.body_mass` exists in
`src/mjlab/envs/mdp/dr/` (see `__init__.py` exports) — config-only change.
Check whether mass edits require inertia co-scaling in mjlab's DR helper
(`pseudo_inertia` also exists if inertia scaling is desired later; skip for
now).

**Verification:** unit test asserting torso mass differs across resets and
total mass stays within expected bounds.

---

## C1 Critic observes the sampled DR parameters (gap G7, motivated by F4)

**Change:** new observation term (put next to the other NUgus obs helpers in
`src/mjlab/tasks/velocity/config/nugus/` or the task's observations module)
returning, per env, a flat vector of *dimensionless ratios to defaults*:

- kp scale, kd scale (actuator gain / cfg default) — 2 per actuator group or
  per joint, follow whatever the `pd_gains` DR granularity is
- effort-limit ratio (current forcerange / default) — captures both the reset
  DR and the intra-episode `effort_drift`
- armature ratio, frictionloss ratio (per joint)
- torso mass ratio and COM offset (xyz, meters)
- foot friction values

Add ONLY to the critic obs group (`enable_corruption=False` side). The actor
group is untouched → actor dimension unchanged → old actor checkpoints stay
loadable for play, though full resume needs matching critic dims (fresh runs
anyway in v16).

**Implementation notes:** DR events write into warp model fields; read the
same fields the DR functions write (`dof_frictionloss`, `dof_armature`,
actuator gain/forcerange, `body_mass`, `body_ipos`) and divide by the
defaults captured at build time. Normalize to ~O(1). ~50–80 dims total.

**Verification:** unit test in the style of
`tests/test_nugus_observation_vector.py`: term present in critic group,
absent from actor group; values change across resets; effort ratio drifts
downward within an episode when `effort_drift` fires.

---

## No-op audit (bug-pattern insurance)

Two silent no-ops have already cost batches (context doc §4). Add
`tests/test_nugus_config_audit.py`:

1. For every DR event with operation "scale", assert the targeted model field
   is nonzero for the targeted dofs/bodies at build time.
2. Build the env, run 200 random-action steps, and assert every reward term
   with weight ≠ 0 produced a nonzero value at least once (catches
   mis-scoped body/joint regexes and dead terms). Allow an explicit skip-list
   for terms that legitimately need walking to fire (e.g. air_time) — but
   keep the list in the test, visible.
3. Assert every env var consumed by `env_cfgs.py` (grep `_env_` helpers)
   appears in `scripts/k8s/volcano-train-job.template.yaml` — catches the
   `PROGRESS_BACKSLIDE_W` class of bug where a knob exists but is never
   exported.

---

## v16 smoke batch

After all of the above: one clock_anneal run, 2000 iters, base task (no
hard_continue), `CRITIC_HEIGHT_SCAN=true`, `JOULE_W=1e-5`, seed 1, via
`scripts/k8s/gen-gridsearch.sh` (add a `gen_v16` generator following the
v13–v15 pattern; pin the new GIT_COMMIT in `scripts/k8s/configmap.yaml`).

**Success:** walks by iter ~1000 on the fixed eval (falls_per_min < 2 at
(0.5,0,0)); reward curve shape similar to v13's base stage even if the
absolute value is lower. **Failure:** no walking by iter 1500 → bisect E0.2a
(velocity limits too tight / actuator migration issue) before anything else;
temporarily relax `velocity_limit` ×1.5 to test that hypothesis.

Record the v16 result in the experiments doc; v16 metrics are the new
baseline all Track A/B/C cells compare against.
