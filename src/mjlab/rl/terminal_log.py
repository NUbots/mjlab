"""JSON logger that captures structured training metrics each iteration.

Wraps the rsl-rl ``Logger.log`` method to extract metrics from the
logger's internal buffers and write them to ``training_log.json`` inside
the run's log directory.

The JSON format is metric-keyed: each top-level key is a metric name and
its value is an array of per-iteration values::

    {
    "iteration": [0, 1, 2],
    "mean_reward": [null, 1.23, 4.56],
    "loss/surrogate": [0.04, 0.03, 0.02]
    }

Metrics that are unavailable on a given iteration (e.g. ``mean_reward``
before any episode completes) are stored as ``null`` so every array stays
the same length.

The file is rewritten atomically each iteration so it is always valid
JSON, and is truncated at the start of each training run.
"""

import json
import statistics
from pathlib import Path
from typing import Any

import torch
from rsl_rl.utils.logger import Logger


def _collect_metrics(
  logger: Logger,
  ep_extras: list[dict[str, Any]],
  it: int,
  start_it: int,
  total_it: int,
  collect_time: float,
  learn_time: float,
  loss_dict: dict[str, float],
  learning_rate: float,
  action_std: torch.Tensor,
  rnd_weight: float | None,
) -> dict[str, Any]:
  """Collect all metrics for a single iteration as a flat dict.

  Every metric printed to the terminal or logged to the writer by the
  rsl-rl ``Logger.log`` method is included here.  Values that are not
  available on this iteration are set to ``None``.
  """
  run_name = logger.cfg.get("run_name")

  m: dict[str, Any] = {
    "iteration": it,
    "total_iterations": total_it,
  }
  if run_name:
    m["run_name"] = run_name

  m["learning_rate"] = round(learning_rate, 8)

  # Losses.
  for key, value in loss_dict.items():
    m[f"loss/{key}"] = round(value, 6)

  # Rewards and episode length.
  has_episodes = len(logger.rewbuffer) > 0
  if has_episodes and logger.cfg["algorithm"]["rnd_cfg"]:
    m["mean_extrinsic_reward"] = round(statistics.mean(logger.erewbuffer), 4)
    m["mean_intrinsic_reward"] = round(statistics.mean(logger.irewbuffer), 4)
    m["rnd_weight"] = round(rnd_weight, 6) if rnd_weight is not None else None
  m["mean_reward"] = (
    round(statistics.mean(logger.rewbuffer), 4) if has_episodes else None
  )
  m["mean_ep_len"] = (
    round(statistics.mean(logger.lenbuffer), 4) if has_episodes else None
  )

  # Policy.
  m["mean_action_std"] = round(action_std.mean().item(), 4)

  # Episode extras (reward components, custom metrics).
  if ep_extras:
    for key in ep_extras[0]:
      values = [ep[key] for ep in ep_extras if key in ep]
      if not values:
        continue
      tensors = []
      for v in values:
        if not isinstance(v, torch.Tensor):
          v = torch.tensor([v])
        if v.dim() == 0:
          v = v.unsqueeze(0)
        tensors.append(v)
      mean_val = torch.cat(tensors).float().mean().item()
      out_key = key if "/" in key else f"Episode/{key}"
      m[out_key] = round(mean_val, 6)

  return m


def _dump_history(history: dict[str, list[Any]], path: Path) -> None:
  """Write *history* as JSON with one line per metric key."""
  lines = ["{"]
  keys = list(history.keys())
  for i, key in enumerate(keys):
    comma = "," if i < len(keys) - 1 else ""
    encoded_values = json.dumps(history[key], separators=(",", ":"))
    lines.append(f"{json.dumps(key)}:{encoded_values}{comma}")
  lines.append("}\n")
  tmp = path.with_suffix(".json.tmp")
  tmp.write_text("\n".join(lines), encoding="utf-8")
  tmp.replace(path)


def enable_terminal_log(logger: Logger) -> None:
  """Patch *logger* so each ``log()`` call also writes a JSON record.

  The JSON file is written to ``<log_dir>/training_log.json`` and is
  truncated when this function is called (i.e. at the start of each run).
  """
  if logger.log_dir is None:
    return

  json_path = Path(logger.log_dir) / "training_log.json"

  # Metric-keyed accumulator: {"metric_name": [val_it0, val_it1, ...]}.
  history: dict[str, list[Any]] = {}
  # Track how many iterations have been recorded so we can pad new keys.
  state = {"n": 0}

  original_log = logger.log

  def _patched_log(
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
    learning_rate: float,
    action_std: torch.Tensor,
    rnd_weight: float | None,
    print_minimal: bool = False,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    # Copy ep_extras before it is cleared by original_log
    saved_ep_extras = list(logger.ep_extras)

    # Call the original log (prints to console, writes to TB/wandb).
    original_log(
      it=it,
      start_it=start_it,
      total_it=total_it,
      collect_time=collect_time,
      learn_time=learn_time,
      loss_dict=loss_dict,
      learning_rate=learning_rate,
      action_std=action_std,
      rnd_weight=rnd_weight,
      print_minimal=print_minimal,
      width=width,
      pad=pad,
    )

    if logger.writer is None:
      return

    metrics = _collect_metrics(
      logger,
      saved_ep_extras,
      it,
      start_it,
      total_it,
      collect_time,
      learn_time,
      loss_dict,
      learning_rate,
      action_std,
      rnd_weight,
    )

    n = state["n"]

    # Append values, back-filling new keys with null so arrays stay
    # aligned.
    for key, value in metrics.items():
      if key not in history:
        history[key] = [None] * n
      history[key].append(value)

    # Keys present in history but absent this iteration get null.
    for key in history:
      if key not in metrics:
        history[key].append(None)

    state["n"] = n + 1

    _dump_history(history, json_path)

  logger.log = _patched_log  # type: ignore[assignment]
