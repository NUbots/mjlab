"""System identification data for a trained velocity policy, in its own simulator.

Three subcommands:

``sequences``
  plays every scripted command sequence of :mod:`mjlab.evaluation.sysid_sequences`
  (steps, level changes, ramps, random staircases, chirps, combined-axis steps,
  plus held-out staircases) and writes one file per run with the commanded and
  achieved velocity. ``--variant nominal`` runs one robot per sequence with every
  randomisation off; ``--variant randomised`` runs ``--replicas`` robots per
  sequence drawn from the training distribution.
``envelope``
  holds each point of a (vx, vy, wz) grid spanning 120 % of the training range,
  one robot per point, and summarises the steady state.
``export-commands``
  writes the command sequences alone, for replay on the robot.

Command amplitudes are fractions of the range the policy trained on, read from
the training run's own config on W&B (``--wandb-run``) or from a saved copy of it
(``--train-config``), and written to ``training_config.json`` in the output.

Examples::

  uv run python scripts/eval/collect_sysid.py sequences --variant nominal
  uv run python scripts/eval/collect_sysid.py sequences --variant randomised \\
    --seeds 0
  uv run python scripts/eval/collect_sysid.py envelope --variant nominal
  uv run python scripts/eval/collect_sysid.py envelope --variant randomised \\
    --seeds 0 1 2
  uv run python scripts/eval/collect_sysid.py export-commands
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

from mjlab.evaluation.harness import TASK_ID
from mjlab.evaluation.sysid import (
  COLUMNS,
  SysidRig,
  TrainingReference,
  Variant,
  file_sha256,
  heading_frame,
  nan_to_none,
  run_table,
  tracking_score,
  write_run,
)
from mjlab.evaluation.sysid_sequences import (
  AXES,
  AXIS_UNITS,
  FAMILIES,
  IDENTIFICATION_SEED,
  LEAD_IN_S,
  VALIDATION_SEED,
  CommandSequence,
  Family,
  build_sequences,
  export_commands,
)
from mjlab.utils.torch import configure_torch_backends

DEFAULT_CHECKPOINT = Path(
  "logs/rsl_rl/nugus_velocity/wandb_checkpoints/qufuh82s/model_39997.pt"
)
DEFAULT_WANDB_RUN = "vincenttumm-the-university-of-newcastle/mjlab/qufuh82s"
DEFAULT_OUT = Path("logs/eval/sysid_qufuh82s")

COLUMN_DOC = {
  "t": "s since reset; row k is at k * dt",
  "cmd_vx": "m/s, command in force over [t, t + dt), as the policy observes it",
  "cmd_vy": "m/s",
  "cmd_wz": "rad/s",
  "vx": "m/s, root body linear velocity in the heading frame (forward)",
  "vy": "m/s, root body linear velocity in the heading frame (left)",
  "wz": "rad/s, root body yaw rate (world z, equal in the heading frame)",
  "x": "m, root body position, world frame, env origin not subtracted",
  "y": "m",
  "z": "m",
  "yaw": "rad, heading of the root body x axis, wrapped to (-pi, pi]",
  "fall": "1 on the row the task's fall condition first held; the run ends there",
  "push": "1 where a training push was applied right after this row's state",
}


@dataclass
class Common:
  checkpoint: Path = DEFAULT_CHECKPOINT
  """rsl-rl checkpoint of the policy."""
  task_id: str = TASK_ID
  out: Path = DEFAULT_OUT
  wandb_run: str = DEFAULT_WANDB_RUN
  """Training run whose config supplies the command ranges and push settings."""
  train_config: Path | None = None
  """Saved copy of the training env config, used instead of ``--wandb-run``.
  Defaults to ``<out>/training_config.json`` when that exists."""
  device: str = "cuda:0"


@dataclass
class Sequences(Common):
  """Play the scripted sequences and write one file per run."""

  variant: Variant = "nominal"
  seeds: tuple[int, ...] = (0,)
  """Environment seeds. Each seed is a full pass over the sequences."""
  replicas: int | None = None
  """Robots per sequence. Defaults to 1 for nominal (it is deterministic), 64
  for randomised."""
  families: tuple[Family, ...] = FAMILIES
  identification_seed: int = IDENTIFICATION_SEED
  validation_seed: int = VALIDATION_SEED
  format: Literal["mat", "csv"] = "mat"
  max_envs: int = 4096
  """Largest batch to build. A family needing more is split."""


@dataclass
class Envelope(Common):
  """Hold each point of a command grid and summarise the steady state."""

  variant: Variant = "nominal"
  seeds: tuple[int, ...] = (0,)
  levels: tuple[int, int, int] = (11, 7, 9)
  """Grid points along vx, vy, wz."""
  span: float = 1.2
  """Grid half-width as a multiple of each axis's training range."""
  hold_s: float = 8.0
  discard_s: float = 3.0
  sigma: float = 0.25
  """Scale of the tracking score ``exp(-||v - v_hat|| / sigma)``."""


@dataclass
class ExportCommands(Common):
  """Write the command sequences alone, for replay on the robot."""

  families: tuple[Family, ...] = FAMILIES
  identification_seed: int = IDENTIFICATION_SEED
  validation_seed: int = VALIDATION_SEED
  control_dt: float = 0.02
  """Policy period the sequences are built at. Must match the policy's."""


def _reference(args: Common) -> TrainingReference:
  """Training config, fetched once and then read from the output directory."""
  args.out.mkdir(parents=True, exist_ok=True)
  saved = args.out / "training_config.json"
  if args.train_config is not None:
    reference = TrainingReference.from_file(args.train_config)
  elif saved.exists():
    reference = TrainingReference.from_file(saved)
  else:
    reference = TrainingReference.from_wandb(args.wandb_run)
  if not saved.exists():
    saved.write_text(
      json.dumps(
        {"source": reference.source, "env_cfg": reference.env_cfg},
        indent=1,
        default=str,
      )
    )
  return reference


def _git_commit() -> str | None:
  try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode
    return commit + ("-dirty" if dirty else "")
  except (OSError, subprocess.CalledProcessError):
    return None


def _provenance(args: Common, reference: TrainingReference) -> dict[str, Any]:
  return {
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": file_sha256(args.checkpoint),
    "task_id": args.task_id,
    "training_config_source": reference.source,
    "git_commit": _git_commit(),
    "command_ranges": {axis: list(reference.command_ranges[axis]) for axis in AXES},
    "command_units": AXIS_UNITS,
    "amplitude_convention": (
      "positive fraction f -> f * range_high; negative f -> f * |range_low|"
    ),
  }


def _rig_metadata(rig: SysidRig) -> dict[str, Any]:
  info = rig.info
  return {
    "variant": info.variant,
    "randomisation": info.variant == "randomised",
    "seed": info.seed,
    "batch_num_envs": info.num_envs,
    "physics_dt_s": info.physics_dt,
    "decimation": info.decimation,
    "policy_dt_s": info.control_dt,
    "policy_rate_hz": 1.0 / info.control_dt,
    "physics_rate_hz": 1.0 / info.physics_dt,
    "velocity_body": info.velocity_body,
    "velocity_source": (
      "root body free joint qpos/qvel, read after each physics step; linear "
      "velocity of the body origin rotated into the heading frame (yaw only)"
    ),
    "observation_noise": info.observation_noise,
    "observation_delay_steps": info.observation_delays,
    "actuator_delay_physics_steps": info.actuator_delays,
    "events": info.events,
    "fall_condition": info.fall_condition,
    "command_source": (
      "scripted: the velocity term's resample/update hooks write the schedule "
      "row for the current policy step immediately before the observation is "
      "built; no smoothing, clamping or dead-zone compensation"
    ),
  }


def _batches(
  sequences: list[CommandSequence], replicas: int, max_envs: int
) -> list[list[CommandSequence]]:
  """Group by family (one logging rate per batch), split to fit ``max_envs``."""
  per_batch = max(1, max_envs // replicas)
  batches = []
  for family in dict.fromkeys(sequence.family for sequence in sequences):
    members = [sequence for sequence in sequences if sequence.family == family]
    for start in range(0, len(members), per_batch):
      batches.append(members[start : start + per_batch])
  return batches


def _schedule(sequences: list[CommandSequence], replicas: int) -> torch.Tensor:
  """Shape ``(T_max, len * replicas, 3)``; shorter sequences hold their last row."""
  length = max(sequence.num_steps for sequence in sequences)
  columns = []
  for sequence in sequences:
    padded = np.concatenate(
      [
        sequence.commands,
        np.repeat(sequence.commands[-1:], length - sequence.num_steps, axis=0),
      ]
    )
    columns += [padded] * replicas
  return torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)


def run_sequences(args: Sequences) -> None:
  reference = _reference(args)
  provenance = _provenance(args, reference)
  replicas = args.replicas or (1 if args.variant == "nominal" else 64)
  for seed in args.seeds:
    dataset = args.variant if args.variant == "nominal" else f"randomised_seed{seed}"
    if args.variant == "nominal" and len(args.seeds) > 1:
      dataset = f"nominal_seed{seed}"
    root = args.out / dataset
    index: list[dict[str, Any]] = []
    rig_meta: dict[str, Any] = {}
    sequences: list[CommandSequence] = []
    for batch_no, batch in enumerate(
      _batches(
        _sequences(args, reference, None),
        replicas,
        args.max_envs,
      )
    ):
      num_envs = len(batch) * replicas
      started = time.time()
      rig = SysidRig(
        args.checkpoint,
        args.variant,
        num_envs,
        seed,
        reference,
        args.device,
        args.task_id,
      )
      _check_dt(batch, rig)
      rig_meta = _rig_metadata(rig)
      physics_rate = batch[0].physics_rate
      result = rig.run(_schedule(batch, replicas), physics_rate)
      randomised = rig.randomised_parameters() if args.variant == "randomised" else None
      rig.close()
      sim_s = time.time() - started

      for seq_no, sequence in enumerate(batch):
        directory = root / sequence.split / sequence.family / sequence.name
        runs = []
        for replica in range(replicas):
          env_id = seq_no * replicas + replica
          runs.append(
            _write_sequence_run(
              args, sequence, result, env_id, replica, directory, randomised
            )
          )
        meta = {
          **provenance,
          "sequence": sequence.metadata(),
          "plant": {**rig_meta, "batch_num_envs": num_envs},
          "columns": {key: COLUMN_DOC[key] for key in COLUMNS},
          "file_format": args.format,
          "runs": runs,
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps(meta, indent=1))
        falls = sum(run["fell"] for run in runs)
        index.append(
          {
            "name": sequence.name,
            "family": sequence.family,
            "split": sequence.split,
            "directory": str(directory.relative_to(root)),
            "runs": len(runs),
            "falls": falls,
          }
        )
        sequences.append(sequence)
      print(
        f"[{dataset}] batch {batch_no}: {len(batch)} x {replicas} "
        f"({batch[0].family}), {result.policy.root.shape[0] * rig.control_dt:.0f} s "
        f"simulated in {sim_s:.0f} s, falls "
        f"{int((result.fall_step >= 0).sum())}/{num_envs}"
      )

    export_commands(sequences, reference.command_ranges, root / "commands")
    (root / "dataset.json").write_text(
      json.dumps(
        {
          **provenance,
          "plant": {
            key: value for key, value in rig_meta.items() if key != "batch_num_envs"
          },
          "replicas_per_sequence": replicas,
          "identification_seed": args.identification_seed,
          "validation_seed": args.validation_seed,
          "columns": {key: COLUMN_DOC[key] for key in COLUMNS},
          "sequences": index,
        },
        indent=1,
      )
    )
    print(f"wrote {root}")


def _sequences(
  args: Sequences | ExportCommands, reference: TrainingReference, dt: float | None
) -> list[CommandSequence]:
  return build_sequences(
    reference.command_ranges,
    dt if dt is not None else _policy_dt(args.task_id),
    args.families,
    args.identification_seed,
    args.validation_seed,
  )


def _policy_dt(task_id: str) -> float:
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task_id, play=True)
  return cfg.sim.mujoco.timestep * cfg.decimation


def _check_dt(batch: list[CommandSequence], rig: SysidRig) -> None:
  for sequence in batch:
    if abs(sequence.dt - rig.control_dt) > 1e-12:
      raise ValueError(
        f"{sequence.name} was built at {sequence.dt} s, the policy runs at "
        f"{rig.control_dt} s"
      )


def _write_sequence_run(
  args: Sequences,
  sequence: CommandSequence,
  result,
  env_id: int,
  replica: int,
  directory: Path,
  randomised: list[dict[str, Any]] | None,
) -> dict[str, Any]:
  steps = sequence.num_steps
  fall_step = int(result.fall_step[env_id])
  # The trace has rows 0..T-1; a fall on the very last step is dated to the last
  # row rather than dropped.
  fell = 0 <= fall_step and fall_step <= steps
  fall_row = min(fall_step, steps - 1) if fell else None
  num_rows = fall_row + 1 if fall_row is not None else steps
  policy = run_table(
    result.policy, env_id, num_rows, fall_row, result.push_flag, rows_per_push_step=1
  )
  physics = None
  if result.physics is not None:
    decimation = round(result.policy.dt / result.physics.dt)
    phys_fall = fall_row * decimation if fall_row is not None else None
    phys_rows = phys_fall + 1 if phys_fall is not None else steps * decimation
    physics = run_table(
      result.physics,
      env_id,
      phys_rows,
      phys_fall,
      result.push_flag,
      rows_per_push_step=decimation,
    )
  fall_time = fall_row * result.policy.dt if fall_row is not None else math.nan
  files = write_run(
    directory / f"env{replica:03d}",
    policy,
    physics,
    {"fell": float(fell), "fall_time_s": fall_time, "env_index": float(env_id)},
    args.format,
  )
  pushes = [
    {"t_s": step * result.policy.dt, "dqvel": delta}
    for push_env, step, delta in result.pushes
    if push_env == env_id and step < num_rows
  ]
  run: dict[str, Any] = {
    "replica": replica,
    "batch_env_index": env_id,
    "files": files,
    "fell": bool(fell),
    "fall_time_s": nan_to_none(fall_time),
    "policy_rows": num_rows,
    "physics_rows": None if physics is None else int(len(physics["t"])),
  }
  if randomised is not None:
    run["pushes"] = pushes
    run["randomised_parameters"] = randomised[env_id]
  return run


def _grid(reference: TrainingReference, levels: tuple[int, int, int], span: float):
  axes = []
  for axis, count in zip(AXES, levels, strict=True):
    low, high = reference.command_ranges[axis]
    axes.append(np.linspace(span * low, span * high, count))
  points = np.array([(x, y, w) for x in axes[0] for y in axes[1] for w in axes[2]])
  return axes, points


def run_envelope(args: Envelope) -> None:
  import scipy.io

  reference = _reference(args)
  provenance = _provenance(args, reference)
  axes, points = _grid(reference, args.levels, args.span)
  num_envs = len(points)
  for seed in args.seeds:
    name = args.variant if args.variant == "nominal" else f"randomised_seed{seed}"
    if args.variant == "nominal" and len(args.seeds) > 1:
      name = f"nominal_seed{seed}"
    directory = args.out / "envelope" / name
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rig = SysidRig(
      args.checkpoint,
      args.variant,
      num_envs,
      seed,
      reference,
      args.device,
      args.task_id,
    )
    dt = rig.control_dt
    lead = round(LEAD_IN_S / dt)
    hold = round(args.hold_s / dt)
    schedule = np.zeros((lead + hold, num_envs, 3))
    schedule[lead:] = points[None]
    result = rig.run(torch.tensor(schedule, dtype=torch.float32), physics_rate=False)
    rig_meta = _rig_metadata(rig)
    rig.close()

    frame = heading_frame(result.policy.root)
    achieved = (
      torch.stack([frame["vx"], frame["vy"], frame["wz"]], dim=-1).cpu().numpy()
    )
    first = lead + round(args.discard_s / dt)
    rows = []
    for env_id, command in enumerate(points):
      fall_step = int(result.fall_step[env_id])
      end = lead + hold if fall_step < 0 else min(fall_step, lead + hold)
      window = achieved[first:end, env_id]
      scores = tracking_score(command[None], window, args.sigma)
      full_window = lead + hold - first
      mean = window.mean(axis=0) if len(window) else np.full(3, math.nan)
      std = window.std(axis=0) if len(window) else np.full(3, math.nan)
      rows.append(
        {
          "cmd_vx": command[0],
          "cmd_vy": command[1],
          "cmd_wz": command[2],
          "frac_vx": _fraction(command[0], reference.command_ranges["vx"]),
          "frac_vy": _fraction(command[1], reference.command_ranges["vy"]),
          "frac_wz": _fraction(command[2], reference.command_ranges["wz"]),
          "mean_vx": mean[0],
          "mean_vy": mean[1],
          "mean_wz": mean[2],
          "std_vx": std[0],
          "std_vy": std[1],
          "std_wz": std[2],
          "tracking_score": scores.mean() if len(scores) else math.nan,
          "tracking_score_fall_as_zero": scores.sum() / full_window,
          "tracking_score_of_mean": float(tracking_score(command, mean, args.sigma)),
          "window_samples": len(window),
          "fell": int(fall_step >= 0),
          "fall_time_s": (fall_step * dt - LEAD_IN_S) if fall_step >= 0 else math.nan,
        }
      )
    keys = list(rows[0])
    np.savetxt(
      directory / "summary.csv",
      np.array([[row[key] for key in keys] for row in rows], dtype=np.float64),
      delimiter=",",
      header=",".join(keys),
      comments="",
      fmt="%.9g",
    )
    scipy.io.savemat(
      directory / "traces.mat",
      {
        "t": np.arange(lead + hold) * dt,
        "cmd": points,
        "vx": achieved[..., 0],
        "vy": achieved[..., 1],
        "wz": achieved[..., 2],
        "fall_step": result.fall_step,
      },
      do_compression=True,
    )
    meta = {
      **provenance,
      "plant": rig_meta,
      "grid": {
        "levels": dict(zip(AXES, args.levels, strict=True)),
        "span_of_training_range": args.span,
        "values": {axis: axes[i].tolist() for i, axis in enumerate(AXES)},
        "one_env_per_point": True,
      },
      "timing": {
        "lead_in_zero_s": LEAD_IN_S,
        "hold_s": args.hold_s,
        "discard_s": args.discard_s,
        "window_s": [LEAD_IN_S + args.discard_s, LEAD_IN_S + args.hold_s],
      },
      "summary_columns": {
        "mean_*/std_*": "achieved heading-frame velocity over the window, up to a fall",
        "tracking_score": f"mean of exp(-||v - v_hat|| / {args.sigma}) over the "
        "window samples before any fall; the norm mixes m/s and rad/s",
        "tracking_score_fall_as_zero": "same, with samples after a fall scored 0",
        "tracking_score_of_mean": "the score of the window-mean velocity; the per-sample score is dominated by within-stride sway",
        "fall_time_s": "s after the command was applied (negative = fell "
        "during the standing lead-in)",
      },
      "traces_mat": "policy-rate heading-frame velocities, shape (T, points); "
      "rows after fall_step are a robot on the floor",
    }
    (directory / "metadata.json").write_text(json.dumps(meta, indent=1))
    falls = sum(row["fell"] for row in rows)
    print(
      f"[envelope {name}] {num_envs} points in {time.time() - started:.0f} s, "
      f"falls {falls}/{num_envs} -> {directory}"
    )


def _fraction(value: float, axis_range: tuple[float, float]) -> float:
  low, high = axis_range
  return value / high if value >= 0 else value / abs(low)


def run_export(args: ExportCommands) -> None:
  reference = _reference(args)
  sequences = _sequences(args, reference, args.control_dt)
  out = args.out / "commands"
  export_commands(sequences, reference.command_ranges, out)
  print(f"wrote {len(sequences)} command sequences to {out}")


def main() -> None:
  args = tyro.extras.subcommand_cli_from_dict(
    {
      "sequences": Sequences,
      "envelope": Envelope,
      "export-commands": ExportCommands,
    },
    description=__doc__,
  )
  configure_torch_backends()
  if isinstance(args, Sequences):
    run_sequences(args)
  elif isinstance(args, Envelope):
    run_envelope(args)
  else:
    run_export(args)


if __name__ == "__main__":
  main()
