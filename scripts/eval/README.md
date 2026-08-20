# Walk engine evaluation

Batched, GPU-parallel evaluation of two walk controllers on two robot models.
The scripts here set up and run the comparison; **collecting the data is left to
you** — everything below has only been smoke-tested at 64 environments.

- `eval_quintic_walk.py` — the ported NUbots quintic walk engine.
- `eval_rl_walk.py` — a trained RL policy, from an rsl-rl checkpoint.

Both are thin entry points over `mjlab.evaluation`, which holds the plant
construction, the two harnesses, the metrics and the output format. Neither
script computes a metric of its own: they both hand raw simulator state to
`WalkMetrics`, so the two engines' numbers are produced by the same code.

## The 2x2

|                  | `--plant eval`                     | `--plant training`                    |
| ---------------- | ---------------------------------- | ------------------------------------- |
| **quintic walk** | `eval_quintic_walk.py`             | `eval_quintic_walk.py --plant training` |
| **RL policy**    | `eval_rl_walk.py --checkpoint ...` | `eval_rl_walk.py --plant training --checkpoint ...` |

`eval` is the reference model: sim-to-real randomisation at nominal and the
hardware's ±π leg joint limits, and otherwise the training model untouched.
`training` is the model policies are trained against — backlash joints, actuator
latency, narrow RL joint clamps. See
`src/mjlab/asset_zoo/robots/nugus/nugus_eval_constants.py` for the argument.

The quintic script additionally takes `--plant nubots-sim` (NUbots' dynamics on
mjlab's kinematic tree) and `--plant nubots-xml` (their MJCF verbatim). The
policy cannot run on those: its observations read mjlab sensor and site names
that the NUbots models do not carry.

## Running

```sh
# Quintic, evaluation plant, 512 robots, 20 s each, forward at 0.3 m/s.
uv run python scripts/eval/eval_quintic_walk.py --plant eval --num-envs 512 \
  --duration 20 --vx 0.3

# Quintic, training plant. Expect it to fall at about 1.72 s.
uv run python scripts/eval/eval_quintic_walk.py --plant training --num-envs 512

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

`summary.json` carries `run` (engine, plant, checkpoint, batch size, duration,
control rate, wall time), the fall statistics, and then two blocks of the same
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
but pays for policy inference.

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

**The two engines do not start the same way, and should not.** The walk engine
begins in the stance its own generator holds, at 100 Hz; the policy begins at
the task keyframe, at the task's 50 Hz. Each controller gets its own home pose
and its own control rate because that is what each one is; forcing either onto
the other's terms would measure the transplant. What *is* shared is the plant,
built from one place for both, and the measurement.

**The RL script uses the task environment internally.** The policy expects
noise-shaped, delayed, clock-augmented, normalised observations, so
`eval_rl_walk.py` builds them with the task's own environment rather than
reconstructing the vector by hand. It disables the domain-randomisation events
(so the robot is nominal), zeroes the reset pose jitter (so every environment
starts identically), removes terminations (so a fallen robot stays fallen and
gets measured instead of being teleported upright), and pins the velocity
command. The environment supplies observations and actions and nothing else —
the metrics come from raw simulator state, exactly as on the quintic side.
