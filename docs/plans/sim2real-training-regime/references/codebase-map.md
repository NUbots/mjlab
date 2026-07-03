# Codebase map (NUgus task, as of commit `adf2023`, branch `add-phase-clock`)

Line numbers drift; regex/grep from the anchors given here. Everything NUgus
is net-new on this branch relative to `main`.

> **UPDATE (post `a1af0d4`, Phase 0):** actuators migrated
> `BuiltinPositionActuatorCfg` → `DcMotorActuatorCfg` with velocity limits
> (rad/s), BAM frictionloss/viscous-damping baselines, and
> `saturation_effort`; new DR events `link_mass`/`payload`/`joint_damping`;
> critic DR-param obs in `config/nugus/dr_observations.py`; mirror map in
> `config/nugus/mirror_map.py`; runner hooks (`ENTROPY_DECAY`, `LR_CAP`,
> `GAMMA`, `MIRROR_AUG`) in `config/nugus/runner.py`; eval harnesses
> `scripts/nugus_eval.py`, `scripts/sim2sim_eval.py`;
> `HARD_COMPONENTS` env var gates hard-stage pieces. The actuator sections
> below describe the PRE-Phase-0 state; trust the diff of `a1af0d4` and
> docs 06–07 over this file where they conflict. Known Phase-0 issue: the
> XH540 velocity limit 3.2 rad/s is too low (true no-load 4.1–4.8 rad/s;
> doc 06 §V5).

## Entry points

| What | Where |
|---|---|
| Task registration (`Mjlab-Velocity-{Rough,Flat}-Nubots-Nugus`) | `src/mjlab/tasks/velocity/config/nugus/__init__.py` |
| NUgus env config (rewards, DR, curriculum, obs — ~1000 lines, env-var driven) | `src/mjlab/tasks/velocity/config/nugus/env_cfgs.py` (`nubots_nugus_rough_env_cfg()`; flat wraps rough) |
| PPO/runner config | `src/mjlab/tasks/velocity/config/nugus/rl_cfg.py` |
| Base velocity env the above mutates | `src/mjlab/tasks/velocity/velocity_env_cfg.py` (`make_velocity_env_cfg`) |
| Robot constants (actuators, keyframe, action scale) | `src/mjlab/asset_zoo/robots/nugus/nugus_constants.py` |
| MJCF | `src/mjlab/asset_zoo/robots/nugus/xmls/nugus.xml` |
| Env-var helpers (`MJLAB_VARIANT`, `JOULE_W`, …) | `env_cfgs.py` `_env_*` functions near the top |

## Robot / actuation facts

- 1 free joint + **20 actuated hinges** + 20 passive `*_backlash` siblings
  (damping 0.001, range ±0.005 rad — gear play). Motor-joint regex:
  `^(?!.*_backlash$).*` (`nugus_constants.py`).
- Actuators are `BuiltinPositionActuatorCfg` (MuJoCo native position servo;
  the XML has NO `<actuator>` block — "from xml" comments in
  nugus_constants are stale):
  - MX-106 (hip_yaw) & XH540-W270 (legs): kp 56.052, kd 1.6548,
    effort 11.086 Nm, armature 0.0266
  - MX-64 (arms, neck, head): kp 31.1558, kd 0.6782, effort 6.1621 Nm,
    armature 0.01195
  - All: `delay_min_lag=1, delay_max_lag=3` (units = 200 Hz physics steps
    → 5–15 ms), `soft_joint_pos_limit_factor=0.9`,
    `velocity_limit=30  # TODO` (**DEAD CONFIG** — see below).
- **`BuiltinPositionActuator` (`src/mjlab/actuator/builtin_actuator.py`)
  never reads `velocity_limit`.** Only `DcMotorActuator`
  (`src/mjlab/actuator/dc_actuator.py`) implements torque-speed limiting
  (`velocity_limit_motor`, `_vel_at_effort_lim`).
- **No `frictionloss` set anywhere for motor joints** → baseline 0.
- Masses: torso 1.833 kg, head 0.474, upper_leg ~0.313, ankle 0.330,
  foot 0.246, arms ~0.3/link; total ≈ 5–6 kg. Keyframe:
  `STAND_BENT_KNEES_KEYFRAME` z=0.4738.
- Sensors in XML: IMU site (gyro/velocimeter/accel/framequat),
  `subtreeangmom root_angmom`, per-foot force+velocity, 20 jointpos; feet
  are box collision geoms + 4 corner sites each (`*_foot_c0..3`).
- Action space: `JointPositionActionCfg`, `use_default_offset=True`, scale
  `NUGUS_ACTION_SCALE` ≈ 0.247 rad. clock_learned appends a 1-dim
  `PhaseDeltaAction` (`src/mjlab/envs/mdp/actions/phase_delta.py`).

## Timing

Sim dt 0.005 (200 Hz), decimation 4 → 50 Hz control; episode 20 s;
`num_steps_per_env=24` → **1 iteration = 24 control steps**;
curriculum `common_step_counter` is in control steps (iter × 24).
Flat-env solver trims: `njmax=300, ccd_iterations=50` (env_cfgs flat
overrides).

## Rewards (base weights → NUgus overrides; implementations in `src/mjlab/tasks/velocity/mdp/rewards.py`)

| Term | Weight | Notes |
|---|---|---|
| track_linear_velocity | 2.0 | exp(−(‖cmd−v‖²+v_z²)/0.05) |
| track_angular_velocity | 2.0 | std²=0.5 |
| upright | `UPRIGHT_W` (grid: 0.5) | hard stage cuts →0.25, widens std |
| pose (`variable_posture`) | 1.0 | per-joint std, speed-regime switched |
| stand_still_pose / _motion | −0.15 (`STAND_W`) / −0.003 | |
| command_progress_backslide | −0.5 (`PROGRESS_BACKSLIDE_W`) | was silently 0 through v9 (`938b771`) |
| body_ang_vel / angular_momentum | −0.05 / −0.01 | hard stage relaxes to −0.02/−0.005 |
| dof_pos_limits | −1.0 | |
| action_rate_l2 / action_acc_l2 | −0.1 / −0.1 | |
| foot_clearance | −15.0 | squared one-sided below-target × foot speed |
| **foot_swing_height** | 0.75 | swapped to `feet_swing_height_clock`: clock-tracked swing arc, target 0.08 m, period `GAIT_PERIOD`=0.7 s, swing_ratio 0.45, offsets (0,0.5) — THE phase-clock reward |
| foot_swing_height_landing / air_time | 0 → ramped | clock_anneal handoff terms |
| phase_delta_nominal | 0 (−5.0 in clock_learned) | staged anneal; tail `PHASE_DELTA_TAIL_W` |
| foot_slip / foot_flat / feet_distance | −1.0 / −0.5 / −0.1 | foot_flat: sole normal is local-X |
| gait_phase_regularity | −0.1 | swing/stance CV across feet |
| joule_heating / joint_acc / torque_rate / soft_landing / base_height | 0 → Phase-C ramp to `JOULE_W`(−3e-4 default; **use −1e-5**), −1e-4, −1e-3, −0.01, +0.3 | |
| actuation_power / cot_proxy / limb_symmetry | 0 — "Disable (debugging)" | wired but off |
| termination_penalty | −10.0 | |

## Curriculum

- Engine: `src/mjlab/envs/mdp/curriculums.py` — `reward_curriculum`
  (step-scheduled, last-stage-wins), `termination_curriculum`,
  **`staged_on_plateau` (EMA plateau-triggered — exists, tested, NOT wired
  for NUgus)**.
- Phase boundaries: `_phase_steps` — p1/p2/p3 = 0.25/`PHASE_C_FRAC`/0.85 ×
  (`PHASE_ITERATIONS`×24), decoupled from `MAX_ITERATIONS`.
- Gait curriculum per variant: `_add_gait_curriculum`.
- Phase-C penalties: `_add_phase_c_curriculum` (4-stage ramp p2→p3).
- base→hard: `_add_hard_continue_curriculum` — from `CONT_BASE_STEP`
  (48000 = iter 2000), offsets 0/250/500/1000 iters: commands →
  x±0.75/y±0.45/yaw±0.80; push ×2; upright 0.5→0.25; hip_roll pose std up;
  body_ang_vel/angular_momentum relaxed.
- Command-vel base curriculum: 3 steps (iter 0/~250/~562) in
  `velocity_env_cfg.py`.

## Domain randomization / events (NUgus events section of env_cfgs.py)

| Event | Mode | Range / op |
|---|---|---|
| push_robot | interval 3–10 s | vx(−0.2,0.4) vy±0.2 r/p±0.05; ×2 in hard stage |
| foot_friction | startup | abs (0.7,1.3), per-foot independent |
| encoder_bias | startup | add ±0.015 rad/joint |
| base_com | startup | torso COM add x/y±0.025, z±0.03 m |
| pd_gains | reset | scale kp,kd (0.9,1.1) |
| effort_limits | reset | scale (`EFFORT_LO`,`EFFORT_HI`)=(0.7,1.2) |
| joint_friction | reset | scale (0.8,1.2) of `dof_frictionloss` — **NO-OP: baseline is 0** |
| joint_armature | reset | scale (0.8,1.2) |
| effort_drift | interval 2–4 s | forcerange ×0.995 per fire |
| current_sensor | reset (CURRENT_OBS only) | gain (0.9,1.1), offset (−0.1,0.1) |

**Absent:** any mass/inertia DR. Available helpers in
`src/mjlab/envs/mdp/dr/__init__.py`: `body_mass`, `pseudo_inertia`,
`joint_damping`, `dof_frictionloss`, `body_com_offset`, etc.

## Observations

- Actor (corrupted, normalized): base_ang_vel(3), projected_gravity(3),
  joint_pos(20), joint_vel(20), actions, command(3), gait_clock(2)
  [+ actuator_current(20) if CURRENT_OBS]. **base_lin_vel and height_scan
  are popped from the actor.** Noise: measured — gyro std 0.02–0.03 rad/s,
  gravity ~4e-3, joint pos 0.01 / vel 0.05. Obs delays: ang_vel/grav lag
  0–2 steps (0–40 ms), joints 1–3 (20–60 ms). Single frame, no history.
- Critic (clean): actor terms + base_lin_vel + height_scan (iff
  `CRITIC_HEIGHT_SCAN`) + foot_height(2) + foot_air_time(2) +
  foot_contact(2) + foot_contact_forces(6). **No DR params** (fixed by C1).
- `gait_clock` (`observations.py`): [sin,cos] 2πφ; φ from time or policy;
  zeroed at standstill; optional `silence_stages` fade (SILENCE_CLOCK —
  never run).

## RL config (`rl_cfg.py`)

Actor & critic MLP (512,256,128) ELU, obs-norm on, init_std 1.0; PPO:
clip 0.2, entropy 0.01, 5 epochs, 4 minibatches, adaptive LR 1e-3
(desired_kl 0.01), γ 0.99, λ 0.95, grad-norm 1.0; 24 steps/env;
save_interval 250. Runner `MjlabOnPolicyRunner` (`src/mjlab/rl/runner.py`):
checkpoint state incl. `common_step_counter` + curriculum snapshot, ONNX
export. **No teacher-student/RMA/distillation anywhere.**

## Commands & termination

`UniformVelocityCommand` (`src/mjlab/tasks/velocity/mdp/velocity_command.py`):
resample 3–8 s (`RESAMPLE_MIN`), rel_standing 0.1, rel_heading 0.3,
rel_forward 0.2, `rel_stop_envs=0.5` with 0.75 s ramp/settle stop tails.
Terminations: time_out 20 s; fell_over = tilt > 50°; out_of_bounds.

## k8s / W&B harness

- Batch generators v3–v15: `scripts/k8s/gen-gridsearch.sh` (one `gen_vNN`
  function per batch — follow this pattern for v16+). Rendered manifests in
  `scripts/k8s/gen_v15/` (untracked).
- Template: `scripts/k8s/volcano-train-job.template.yaml` — **every new env
  var must be added here or it silently defaults** (the v9 backslide bug).
- `scripts/k8s/configmap.yaml`: `WANDB_PROJECT=mjlab`, pinned `GIT_COMMIT`
  (update per batch), `NUM_ENVS=8192`/GPU, 4 GPU/job via torchrunx.
- W&B: entity `vincenttumm-the-university-of-newcastle`, project `mjlab`,
  tags `batch-vNN`+variant+seed. API key: k8s secret `mjlab-wandb`
  (namespace `mjlab`).
- Experiment log: `docs/experiments/2026-07-nugus-gridsearch-summary.md`.

## Relevant tests (patterns to imitate)

`tests/test_policy_phase.py`, `test_nugus_observation_vector.py`,
`test_nugus_v10_pipeline.py`, `test_rewards.py`,
`test_stand_still_rewards.py`, `test_progress_backslide_reward.py`,
`test_velocity_command_curriculum.py`, `test_envs_curriculums.py`,
`test_actuator_current.py`, `test_foot_height_sensor.py`.

## Dev workflow (from CLAUDE.md)

`uv run` everything; `make check` (format+lint+type) before committing;
`make test` before PRs; changelog entries in `docs/source/changelog.rst`
for user-facing changes.
