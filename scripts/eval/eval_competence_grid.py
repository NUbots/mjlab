"""Per-episode competence over a command x disturbance grid.

The sweeps ask how fast a controller can walk, the profile asks how well it
follows a command that moves, and the push battery asks what one shove costs.
None of them answers the question a behaviour tree actually has, which is
conditional on both at once: *how much of a hit can it absorb while still
delivering the commanded velocity*. This one crosses the two axes and reports,
per cell, a distribution over episodes rather than a mean.

Five quantities per episode, all of them the training competence tracker's
definitions with the smoothing taken off -- see
:mod:`mjlab.evaluation.competence`. Two of them, whether the robot fell and how
much of the episode it survived, only exist if episodes can end, so this run
puts the ``fell_over`` termination and the training episode length back, which
every other run in the pipeline deliberately removes.

Examples::

  # Smoke test: a coarse grid, a handful of episodes per cell.
  uv run python scripts/eval/eval_competence_grid.py --num-envs 512 \\
    --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model.pt \\
    --grid.vx "(0.0,0.5)" --grid.vy "(0.0,)" --grid.wz "()" \\
    --grid.shoves "(0.0,0.4)" --grid.episodes-per-cell 4

  # The two competence policies, on the evaluation plant. The history policy
  # only loads against the task that builds its observation layout.
  uv run python scripts/eval/eval_competence_grid.py --num-envs 4096 \\
    --tag gating --checkpoint logs/.../gating/model_34999.pt
  uv run python scripts/eval/eval_competence_grid.py --num-envs 4096 \\
    --tag history --task-id Mjlab-Velocity-Flat-Nubots-Nugus-History \\
    --checkpoint logs/.../history/model_34999.pt

  # The other half of the 2x2.
  uv run python scripts/eval/eval_competence_grid.py --plant training ...
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tyro

import mjlab
from mjlab.evaluation.competence import (
  ShoveCfg,
  build_grid,
  format_grid_summary,
  save_grid_run,
)
from mjlab.evaluation.harness import TASK_ID, EvalPlant, RlEvalHarness
from mjlab.utils.torch import configure_torch_backends


@dataclass
class GridCfg:
  """The command x disturbance grid.

  The command axis is the ``(vx, vy)`` plane at zero yaw rate, plus a yaw slice
  taken at one forward speed. A full ``(vx, vy, wz)`` cube would be three
  dimensional before the disturbance axis is added; the plane is what a
  behaviour tree asks for, and the slice keeps yaw from going unmeasured.
  """

  vx: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
  """Forward commands, in m/s. Spans the trained range."""
  vy: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
  """Lateral commands, in m/s."""
  wz: tuple[float, ...] = (-0.5, -0.25, 0.25, 0.5)
  """Yaw rates for the slice, in rad/s. Zero is omitted because the whole plane
  above is already at zero."""
  wz_at_vx: float = 0.3
  """Forward speed the yaw slice is taken at, in m/s. Turning on the spot and
  turning while walking are different asks, and the latter is the one a game
  situation produces."""

  shoves: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2)
  """Shove magnitudes, as ``|dv_xy|`` in m/s. Must include ``0.0``: the
  undisturbed row is what every other row is read against.

  Chosen to bracket the cliff rather than to span the training range. Measured
  on the competence-gated policy walking forward at 0.5 m/s on the evaluation
  plant, the fall rate goes 0% at 0.6, 44% at 0.8, 95% at 1.0 and 100% at 1.2,
  and nothing at all happens below 0.6 -- so bins under half a metre per second
  buy identical rows. Harder commands move the cliff down, which is why 0.4 is
  kept."""
  shove: ShoveCfg = field(default_factory=ShoveCfg)
  """When the shoves land. See ``--grid.shove.help``."""

  episodes_per_cell: int = 64
  """Episodes the worst-covered cell must reach before the run stops. Enough to
  report quartiles rather than a mean; the interesting cells are the
  high-variance ones and a mean cannot show that."""
  seed: int = 0
  """Seeds the shove headings, the protocol's only stochastic input."""

  def commands(self) -> tuple[tuple[float, float, float], ...]:
    plane = tuple((x, y, 0.0) for x in self.vx for y in self.vy)
    yaw = tuple((self.wz_at_vx, 0.0, w) for w in self.wz)
    return plane + yaw

  @property
  def num_cells(self) -> int:
    return len(self.commands()) * len(self.shoves)


@dataclass
class Args:
  checkpoint: Path
  """rsl-rl checkpoint to evaluate."""
  task_id: str = TASK_ID
  """Registered task supplying the policy's observation, action and command
  pipeline.

  A checkpoint only loads against the task it was trained on, so the policy
  that reads a window of past observations needs
  ``Mjlab-Velocity-Flat-Nubots-Nugus-History`` named here and the one that does
  not needs the default."""
  plant: EvalPlant = "eval"
  """Robot model: ``eval`` is the reference, ``training`` the model the policy
  was trained against."""
  num_envs: int = 4096
  """Environments. Must be at least the cell count; the grid is tiled over
  them, and more environments per cell means fewer episodes run end to end."""
  device: str = "cuda:0"
  """Torch device."""

  grid: GridCfg = field(default_factory=GridCfg)
  """The grid and the episode budget. See ``--grid.help``."""

  output_dir: Path = Path("logs/eval")
  """Runs land in ``<output_dir>/<tag>/``."""
  tag: str | None = None
  """Name for this run's output directory. Defaults to plant and time."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  if not args.checkpoint.exists():
    raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
  if 0.0 not in args.grid.shoves:
    raise ValueError(
      "--grid.shoves must include 0.0: without the undisturbed row there is "
      "nothing to read the disturbed rows against"
    )

  cfg = args.grid
  harness = RlEvalHarness(
    checkpoint=args.checkpoint,
    plant=args.plant,
    num_envs=args.num_envs,
    device=args.device,
    task_id=args.task_id,
    episodic=True,
  )
  grid = build_grid(
    commands=cfg.commands(),
    shoves=cfg.shoves,
    num_envs=args.num_envs,
    device=args.device,
  )
  episode_s = float(harness.env.max_episode_length) * harness.control_dt
  onsets = cfg.shove.onsets(harness.control_dt, int(harness.env.max_episode_length))

  print(f"\ncompetence grid on the {args.plant} plant")
  print(
    f"grid              : {len(cfg.commands())} commands x "
    f"{len(cfg.shoves)} shove bins = {cfg.num_cells} cells"
  )
  print(
    f"coverage          : {args.num_envs} envs, "
    f"{int(grid.envs_per_cell().min())} per cell, "
    f"{cfg.episodes_per_cell} episodes wanted per cell"
  )
  print(
    f"episode           : {episode_s:.0f} s at "
    f"{1.0 / harness.control_dt:.0f} Hz, {len(onsets)} shoves at "
    f"{', '.join(f'{o * harness.control_dt:.0f}' for o in onsets)} s"
  )

  started = time.time()
  last_report = [0]

  def report(step: int, completed: int) -> None:
    if step - last_report[0] < 500:
      return
    last_report[0] = step
    print(
      f"  step {step:>6}: worst-covered cell has {completed}/"
      f"{cfg.episodes_per_cell} episodes"
    )

  try:
    table = harness.run_competence_grid(
      grid,
      episodes_per_cell=cfg.episodes_per_cell,
      shove_cfg=cfg.shove,
      seed=cfg.seed,
      on_step=report,
    )
  finally:
    harness.close()
  elapsed = time.time() - started

  tag = args.tag or f"competence_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  output_dir = args.output_dir / tag
  run = {
    "engine": "rl",
    "plant": args.plant,
    "task_id": args.task_id,
    "checkpoint": str(args.checkpoint),
    "num_envs": args.num_envs,
    "episode_length_s": round(episode_s, 3),
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "grid": asdict(cfg),
    "shove_onsets_s": [round(o * harness.control_dt, 3) for o in onsets],
    "wall_time_s": round(elapsed, 1),
  }
  summary = save_grid_run(output_dir, run, grid, table)

  print()
  print(format_grid_summary(grid, summary["cells"]))
  print()
  print(f"episodes          : {table.num_episodes}")
  print(f"wall time         : {elapsed:.1f} s")
  print(f"wrote             : {output_dir}/episodes.csv, cells.json")


if __name__ == "__main__":
  main()
