# Walk engine evaluation

Batched, GPU-parallel evaluation of three walk controllers on two robot models.
The scripts here set up and run the comparison; **collecting the data is left to
you** — everything below has only been smoke-tested at 64 environments.

- `eval_quintic_walk.py` — the ported NUbots quintic walk engine.
- `eval_distilled_quintic_walk.py` — NUbots' distilled copy of that engine, the
  policy their `module/skill/NeuralWalk` deploys.
- `eval_rl_walk.py` — a trained RL policy, from an rsl-rl checkpoint.

All three are thin entry points over `mjlab.evaluation`, which holds the plant
construction, the harnesses, the metrics and the output format. No script
computes a metric of its own: they all hand raw simulator state to
`WalkMetrics`, so every controller's numbers are produced by the same code.

## The grid

|                    | `--plant eval`                     | `--plant training`                    |
| ------------------ | ---------------------------------- | ------------------------------------- |
| **quintic walk**   | `eval_quintic_walk.py`             | `eval_quintic_walk.py --plant training` |
| **distilled copy** | `eval_distilled_quintic_walk.py`   | `eval_distilled_quintic_walk.py --plant training` |
| **RL policy**      | `eval_rl_walk.py --checkpoint ...` | `eval_rl_walk.py --plant training --checkpoint ...` |

`eval` is the reference model: sim-to-real randomisation at nominal and the
hardware's ±π leg joint limits, and otherwise the training model untouched.
`training` is the model policies are trained against — backlash joints, actuator
latency, narrow RL joint clamps. See
`src/mjlab/asset_zoo/robots/nugus/nugus_eval_constants.py` for the argument.

The quintic and distilled scripts additionally take `--plant nubots-sim`
(NUbots' dynamics on mjlab's kinematic tree) and `--plant nubots-xml` (their
MJCF verbatim). The RL policy cannot run on those: its observations read mjlab
sensor and site names that the NUbots models do not carry. The distilled policy
can, because it reads nothing at all — see below.

## Running

```sh
# Quintic, evaluation plant, 512 robots, 20 s each, forward at 0.3 m/s.
uv run python scripts/eval/eval_quintic_walk.py --plant eval --num-envs 512 \
  --duration 20 --vx 0.3

# Quintic, training plant. Expect it to fall at about 1.72 s.
uv run python scripts/eval/eval_quintic_walk.py --plant training --num-envs 512

# The distilled copy of the engine, both plants. No checkpoint to supply: the
# policy NUbots deploys ships with mjlab.
uv run python scripts/eval/eval_distilled_quintic_walk.py --plant eval \
  --num-envs 512 --duration 20 --vx 0.3
uv run python scripts/eval/eval_distilled_quintic_walk.py --plant training

# How far the copy runs from the engine it copies, joint by joint.
uv run python scripts/eval/eval_distilled_quintic_walk.py --track-teacher True

# RL policy, both plants. The checkpoint argument is required.
uv run python scripts/eval/eval_rl_walk.py --plant eval --num-envs 512 \
  --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/5l83efo3/model_39997.pt
uv run python scripts/eval/eval_rl_walk.py --plant training --num-envs 512 \
  --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/5l83efo3/model_39997.pt

# Sweep a command axis across the batch instead of holding one command. The
# grid is tiled over the environments, so 2048 robots over 5 speeds is 409 each.
uv run python scripts/eval/eval_quintic_walk.py --num-envs 2048 \
  --sweep-vx "(0.1,0.2,0.3,0.4,0.5)"

# Sweeps compose; unswept axes fall back to --vx / --vy / --wz.
uv run python scripts/eval/eval_quintic_walk.py --num-envs 2048 \
  --sweep-vx "(0.1,0.3)" --sweep-wz "(-0.5,0.0,0.5)"
```

mjlab runs tyro with flag conversion off, so boolean flags take an explicit
value: `--balance False`, `--exact-ik True`, `--switch-when-planted True`.

### Watching a run

`--viser` streams one environment to a viser server so you can watch the gait in
a browser. It is for eyeballing, not for collecting: it renders every control
step and throttles the run to real time.

```sh
uv run python scripts/eval/eval_quintic_walk.py --num-envs 1 --duration 20 --viser True
uv run python scripts/eval/eval_distilled_quintic_walk.py --num-envs 1 --duration 20 \
  --viser True
uv run python scripts/eval/eval_rl_walk.py --num-envs 1 --duration 20 --viser True \
  --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/5l83efo3/model_39997.pt
```

The script prints the address to open:

```
viser             : http://localhost:8080 (env 0, real time)
```

`--viser-port` moves it, `--viser-env` picks which environment of the batch to
show, and `--viser-realtime False` lets the playback run as fast as it computes.

The view sits on the `on_step` callback the harnesses already take, so it cannot
influence the run: the physics for a step is integrated before the callback sees
it, and all the callback does is copy state out and sleep. A one-environment run
with `--viser` and the same run headless produce byte-identical `per_env.csv`
(verified; the paced run just takes longer on the wall clock). With the flag off
nothing here is constructed and viser is never imported.

Nothing stops you pointing it at a batch of thousands, but it will be miserable
and the numbers are the point there anyway.

### Checkpoints

`--checkpoint` takes a path; the pattern is
`logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run-id>/model_<iteration>.pt`.

**Smoke test every new checkpoint before trusting a long run.** A checkpoint
trained against an older observation layout fails in one of two ways: it either
refuses to load (a shape mismatch out of `runner.load`, which is loud and fine)
or it loads and quietly stands still, which is not. Of the checkpoints on this
machine, `5l83efo3/model_39997.pt` tracks a 0.3 m/s command to within 0.005 m/s,
`gdkfin0z/model_44998.pt` will not load, and `0lyeduth/model_44996.pt` loads and
stands. A 10 s run at 64 environments takes well under a minute and tells you
which you have: look at `achieved_vx` against the command you gave.

### The distilled policy

`eval_distilled_quintic_walk.py` needs no checkpoint. The policy NUbots deploys
is copied into the repository at
`src/mjlab/controllers/distilled_walk/data/walk_policy.onnx` (their
`module/skill/NeuralWalk/data/model/walk_policy.onnx`, with its external weight
file folded in) and is the default for `--policy`. Its weights are read out of
the ONNX and replayed with torch operators, so a whole batch infers on the GPU
at once; `test_distilled_walk_policy.py` pins that replay against onnxruntime
and against two hundred steps of NUbots' own recorded training data, which it
reproduces to 0.03°.

**The policy is blind.** Its forty-six observations are the velocity command,
the walk engine's phase clock and state, and its own three previous outputs —
no joint positions, no attitude, no contact. It is a learned trajectory
generator, not a feedback controller, and it will keep producing a gait for a
robot lying on the floor. That is also why it runs on every plant: it asks the
model for nothing. The phase clock is mjlab's port of the walk engine, the same
object `WalkDataCollector` drove to generate the training set; replaying a
recorded episode through it reproduces the recorded clock columns to 3e-8.

**It learned a different IK from the one this engine ships.** The training
targets came from NUbots' deployed solver — the analytical solution refined
numerically against `models/robot.urdf` — while `calculate_leg_joints` here is
the analytical one alone. Standing, the policy asks for 1.359 rad of knee where
this engine's idealised IK asks for 0.845 and its exact-geometry solver
(`--exact-ik`) asks for 1.723. So the policy stands in a deeper crouch than
`eval_quintic_walk.py` does, and comparing the two joint angle by joint angle
mostly measures that calibration rather than the distillation.

Two flags follow from it:

- `--history-init` chooses what the run starts from. The default, `settled`,
  iterates the policy at zero command until its output stops moving and starts
  the robot in *that* pose — standing is a fixed point of an autoregressive
  policy, and this one's fixed point matches the stance recorded in NUbots'
  training data to 1e-4 rad, so the training-time stance is recovered without
  needing the IK that produced it. `stance` starts from the walk engine's stance
  instead, which is the same start `eval_quintic_walk.py` uses; the policy
  leaves it within three control steps. `zeros` reproduces the thirty-six zeros
  `NeuralWalk.cpp` assigns on start.
- `--track-teacher` runs the walk engine alongside on the same commands and
  writes a `teacher_tracking` block into `summary.json`. Read
  `stance_relative_mean_abs_error_rad`, not `mean_abs_error_rad`: the first
  subtracts each controller's own standing pose and leaves the shape of the
  motion, the second is dominated by the IK offset above. At 0.2 m/s forward on
  the evaluation plant they come out around 0.012 rad and 0.165 rad
  respectively.

The engine it is compared against runs with balance control off, because
`WalkDataCollector` recorded the generator's own foot poses and never ran
`FootController` over them — the policy never saw a balance correction and
cannot produce one.

## Output

Each run writes a directory under `--output-dir` (default `logs/eval/`), named
`<engine>_<plant>_<timestamp>` unless you pass `--tag`:

```
logs/eval/quintic_eval_20260819_084037/
  per_env.csv    one row per environment
  summary.json   the aggregate, plus the run's configuration
```

`per_env.csv` has an `env` column followed by every field of `PerEnvMetrics`:

| column | meaning |
| --- | --- |
| `command_vx`, `command_vy`, `command_wz` | what this environment was asked for |
| `survived` | 1.0 if still upright at the end |
| `fall_time` | seconds until the torso tipped past 60° from vertical; NaN if it never did |
| `alive_time` | seconds measured, i.e. the run length or the time to the fall |
| `achieved_vx`, `achieved_vy`, `achieved_wz` | mean body-frame velocity over the alive period |
| `error_vx`, `error_vy`, `error_wz`, `tracking_error` | achieved minus commanded; `tracking_error` is the planar norm |
| `displacement_x`, `displacement_y`, `path_speed` | world-frame travel, and distance per second of alive time |
| `rms_roll`, `rms_pitch`, `min_upright` | torso attitude statistics |
| `cadence_hz` | foot swaps per second |

CSV rather than parquet: a run is one row per environment, so even a large sweep
is a few thousand rows, and CSV costs no dependency and opens anywhere.

`summary.json` carries `run` (engine, plant, checkpoint or policy, batch size,
duration, control rate, wall time), a `teacher_tracking` block if the distilled
run was given `--track-teacher`, the fall statistics, and then two blocks of the same
walking metrics: `survivors` and `all_envs`. Quote the survivor figures —
averaging a fallen robot's slide into a mean speed describes neither population
— and note that when nothing survives they are all NaN and only `all_envs` has
content.

## Adding a metric

One file, three places, all in `src/mjlab/evaluation/metrics.py`:

1. a buffer in `WalkMetrics.__init__`,
2. its update in `WalkMetrics.record` (multiply by the `weight` mask so it only
   accumulates while the robot is upright),
3. a field on `PerEnvMetrics`, and its name in `WALK_QUALITY_METRICS` if it
   belongs in the aggregate.

The CSV columns and the JSON summary are derived from `PerEnvMetrics`, so
nothing downstream needs touching. If the metric needs a quantity the harnesses
do not read yet, add it to `EvalState` and to `EvalState.from_entity` — that one
method is where both engines get their state, and keeping it that way is what
makes the comparison sound.

## Runtime and memory

Measured on an RTX 3060 (12 GiB), quintic engine, evaluation plant, warm warp
kernel cache:

| environments | wall time for 5 s each | throughput | device memory |
| --- | --- | --- | --- |
| 256 | 7.5 s | 171 robot-s/s | 0.19 GiB |
| 1024 | 8.4 s | 611 robot-s/s | 0.19 GiB |
| 2048 | 11.9 s | 861 robot-s/s | 0.46 GiB |
| 4096 | 21.4 s | 958 robot-s/s | 1.38 GiB |

Throughput saturates around 950 robot-seconds per second; batches in the low
thousands are comfortable and memory is nowhere near the limit, so the practical
ceiling is time, not VRAM. A 4096 × 30 s collection is about two minutes of wall
time. (The engine runs in double, which costs roughly a quarter of the
throughput on this card and buys a cadence that matches the C++ exactly; see
``ENGINE_DTYPE``.) Add ~30 s the first time a model is compiled on a cold kernel cache. The
RL side is faster per robot-second (50 Hz control against the engine's 100 Hz)
but pays for policy inference. The distilled policy replaces the engine's IK
with one small matrix multiply per control step and keeps the phase clock, so it
is a little cheaper than the quintic side at the same rate — until
`--track-teacher`, which runs both and costs both.

## Things to read before designing an experiment

**Backward walking falls over.** With the deployed parameters the quintic engine
does not walk backwards: at −0.1 m/s it falls after about 2.5 s on both the
evaluation plant and NUbots' own dynamics, and its mean displacement is
*forwards* under a backwards command. This is the tuning, not the plant or the
port. Sweeping `--sweep-vx` through negative values will produce a wall of falls
that says nothing new.

**`only_switch_when_planted` is off, deliberately.** `Walk.yaml` sets it true,
but `Walk.cpp` calls `set_parameters` before assigning that one field, so the
deployed binary runs with it false. `--switch-when-planted True` turns it on if
you want to measure it: forward it defers a single control tick in twenty
seconds and changes nothing, and turning in place it stalls the gait for up to
150 ms at a time and topples the robot after 2.3 s. See `NUGUS_WALK_PARAMETERS` in
`src/mjlab/controllers/quintic_walk/walk_generator.py`.

**The three controllers do not start the same way, and should not.** The walk
engine begins in the stance its own generator holds, at 100 Hz; the distilled
policy begins in the stance *it* holds at rest, also at 100 Hz, and the two
stances are not the same one; the RL policy begins at the task keyframe, at the
task's 50 Hz. Each controller gets its own home pose and its own control rate
because that is what each one is, and a robot planted in another controller's
stance spends its first control steps being dragged out of it. What *is* shared
is the plant, built from one place for all three, and the measurement.

**The distilled policy is not the engine with a different label.** It copies the
engine's trajectories through a different IK, at a different stance, with no
balance correction and no way to react to the robot at all. Where it beats the
engine — on the evaluation plant at 0.2 m/s it tracked forward speed slightly
better and halved RMS pitch in a short smoke test, and on the training plant it
fell at 1.4 s against the engine's 1.7 — the interesting question is which of
those differences it is, and the answer is usually the crouch rather than the
network. `--track-teacher` and `--exact-ik` on the quintic side are the two
handles for separating them.

**The RL script uses the task environment internally.** The policy expects
noise-shaped, delayed, clock-augmented, normalised observations, so
`eval_rl_walk.py` builds them with the task's own environment rather than
reconstructing the vector by hand. It disables the domain-randomisation events
(so the robot is nominal), zeroes the reset pose jitter (so every environment
starts identically), removes terminations (so a fallen robot stays fallen and
gets measured instead of being teleported upright), and pins the velocity
command. The environment supplies observations and actions and nothing else —
the metrics come from raw simulator state, exactly as on the quintic side.
