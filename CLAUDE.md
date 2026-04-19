# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# Architecture

mjlab is a GPU-accelerated robotics RL framework built on MuJoCo Warp. It implements Isaac Lab's manager-based API with direct access to native MuJoCo data structures.

## Manager-Based Pattern

`ManagerBasedRlEnv` (`envs/manager_based_rl_env.py`) is the central environment class. It owns a set of managers, each responsible for one aspect of the RL loop:

- **ActionManager** — decodes policy outputs into per-actuator commands
- **ObservationManager** — computes observations with optional noise, history, and delay
- **RewardManager** — sums weighted reward terms each step
- **TerminationManager** — checks done conditions (terminations and timeouts)
- **CommandManager** — generates task-specific targets (e.g., velocity commands)
- **CurriculumManager** — scales task difficulty over training
- **EventManager** — applies domain randomization on reset and at intervals
- **MetricsManager** — accumulates per-step values into episode-level metrics
- **RecorderManager** — records trajectories for logging

Each manager is configured by a dict of named **terms**: small functions or classes (e.g., `reward_joint_limits`, `obs_joint_pos`) that are composed by the manager. Adding a new reward or observation means writing a term function and registering it in the env config — no changes to the manager itself.

## Simulation Stack

**Sim** (`sim/sim.py`) wraps MuJoCo Warp physics. It exposes `step()`, `forward()` (recompute derived quantities), and `sense()` (sensor updates). CUDA graph optimization is supported; first-step compilation delays are expected.

**Scene** (`scene/scene.py`) builds the MuJoCo world incrementally from Entities, Sensors, and Terrain, then compiles the model. Once compiled, only state (positions, velocities) can be mutated.

**Entity** (`entity/entity.py`) represents a robot or object. It owns an **ActuatorGroup** that converts policy commands through a configurable control law (IdealPD, DC motor, learned, XML-defined). Actuators support command delays and effort limits.

## Task Structure

Tasks live under `tasks/<domain>/` (e.g., `velocity/`, `tracking/`, `manipulation/`). Each task subdirectory contains:

```
config/   # ManagerBasedRlEnvCfg subclasses (one per robot/variant)
mdp/      # Custom reward, observation, and termination term functions
rl/       # RSL-RL hyperparameter configs
```

Tasks are registered via `register_mjlab_task()` in `tasks/registry.py` using the `mjlab.tasks` entry point group. The registry resolves task IDs at runtime; `list-envs` prints all registered tasks.

Each task has a **play config** variant (separate `ManagerBasedRlEnvCfg`) used during evaluation — typically with longer episodes and domain randomization disabled (`enable_corruption=False`).

## Environment Step Sequence

Understanding the step order matters when debugging observation staleness or reward timing:

1. `action_manager.process_action()` — parse and scale raw action
2. For each physics substep (`decimation` times): apply action → write to sim → `sim.step()` → update scene sensors
3. `termination_manager.compute()` then `reward_manager.compute()`
4. Auto-reset: curriculum → `sim.reset()` → `scene.reset()` → `scene.write_data_to_sim()`
5. `sim.forward()` — recompute derived quantities (1-substep lag on derived state)
6. `command_manager.compute()` then `event_manager.apply()`
7. `sim.sense()` then `observation_manager.compute()`
8. Return `(obs, rewards, terminated, truncated, extras)`

## Key Conventions

- **Config-as-code:** All environment configuration uses Python dataclasses loaded by `tyro`. CLI flags mirror dataclass field paths (e.g., `--env.scene.num-envs 4096`).
- **Regex name matching:** Joint, geom, and body names throughout configs are regex patterns matched against the MuJoCo model at build time.
- **Observation groups:** Configs define separate `"actor"` (policy input) and `"critic"` (value function) observation groups, each a list of named terms.
- **GPU-first:** All tensors live on GPU. Set `FORCE_CPU=1` to run on CPU (used in tests without a GPU).
- **Multi-GPU training:** `torchrunx` launches one independent environment per GPU; use `--gpu-ids 0 1 2 3` with `train`.
- **Entry scripts:** `train`, `play`, `demo`, `list-envs`, `export-scene`, `viz-nan` (installed via pyproject entry points).
