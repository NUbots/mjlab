"""Tests for the terminal log JSON writer."""

import json
from collections import deque
from unittest.mock import MagicMock

import torch

from mjlab.rl.terminal_log import _collect_metrics, _dump_history, enable_terminal_log


def _make_logger(
  rewbuffer=None,
  lenbuffer=None,
  ep_extras=None,
  run_name=None,
):
  """Create a minimal mock of rsl-rl's Logger with the fields we read."""
  logger = MagicMock()
  cfg: dict = {
    "algorithm": {"rnd_cfg": None},
  }
  if run_name is not None:
    cfg["run_name"] = run_name
  logger.cfg = cfg
  logger.rewbuffer = deque(rewbuffer or [])
  logger.lenbuffer = deque(lenbuffer or [])
  logger.ep_extras = ep_extras or []
  return logger


def test_collect_metrics_all_terminal_fields():
  logger = _make_logger(
    rewbuffer=[42.0, 43.0],
    lenbuffer=[300.0, 310.0],
    run_name="nugus-flat",
  )
  m = _collect_metrics(
    logger,
    logger.ep_extras,
    it=10,
    start_it=0,
    total_it=5000,
    collect_time=0.843,
    learn_time=0.378,
    loss_dict={"surrogate": 0.0234, "value_function": 1.4567},
    learning_rate=3e-4,
    action_std=torch.tensor([0.71, 0.72]),
    rnd_weight=None,
  )

  assert m["iteration"] == 10
  assert m["total_iterations"] == 5000
  assert m["run_name"] == "nugus-flat"
  assert m["learning_rate"] == 0.0003
  assert m["loss/surrogate"] == 0.0234
  assert m["loss/value_function"] == 1.4567
  assert m["mean_reward"] == 42.5
  assert m["mean_ep_len"] == 305.0
  assert m["mean_action_std"] == 0.715


def test_collect_metrics_null_rewards_when_empty():
  """When rewbuffer is empty, mean_reward/mean_ep_len should be None."""
  logger = _make_logger()
  m = _collect_metrics(
    logger,
    logger.ep_extras,
    it=0,
    start_it=0,
    total_it=100,
    collect_time=1.0,
    learn_time=0.5,
    loss_dict={},
    learning_rate=3e-4,
    action_std=torch.tensor([0.5]),
    rnd_weight=None,
  )
  assert m["mean_reward"] is None
  assert m["mean_ep_len"] is None


def test_collect_metrics_episode_extras():
  extras = [
    {"rew_tracking": torch.tensor([1.0, 2.0])},
    {"rew_tracking": torch.tensor([3.0, 4.0])},
  ]
  logger = _make_logger(ep_extras=extras)
  m = _collect_metrics(
    logger,
    logger.ep_extras,
    it=5,
    start_it=0,
    total_it=100,
    collect_time=1.0,
    learn_time=0.5,
    loss_dict={},
    learning_rate=3e-4,
    action_std=torch.tensor([0.5]),
    rnd_weight=None,
  )
  assert m["Episode/rew_tracking"] == 2.5


def test_dump_history_one_line_per_key(tmp_path):
  history = {
    "iteration": [0, 1, 2],
    "mean_reward": [None, 1.23, 4.56],
  }
  path = tmp_path / "test.json"
  _dump_history(history, path)
  text = path.read_text(encoding="utf-8")

  # Should have one line per key plus braces.
  lines = text.strip().splitlines()
  assert lines[0] == "{"
  assert lines[-1] == "}"
  assert len(lines) == 4  # { + 2 keys + }

  data = json.loads(text)
  assert data == history


def test_enable_terminal_log_metric_keyed_format(tmp_path):
  """End-to-end: patched logger writes metric-keyed JSON with arrays."""
  logger = _make_logger(rewbuffer=[10.0], lenbuffer=[200.0])
  logger.log_dir = str(tmp_path)
  logger.writer = MagicMock()

  enable_terminal_log(logger)

  action_std = torch.tensor([0.5])
  for it in range(3):
    logger.log(
      it=it,
      start_it=0,
      total_it=10,
      collect_time=0.5,
      learn_time=0.3,
      loss_dict={"surrogate": 0.01},
      learning_rate=3e-4,
      action_std=action_std,
      rnd_weight=None,
    )

  json_path = tmp_path / "training_log.json"
  data = json.loads(json_path.read_text(encoding="utf-8"))

  assert data["iteration"] == [0, 1, 2]
  assert data["total_iterations"] == [10, 10, 10]
  assert len(data["mean_reward"]) == 3
  assert len(data["loss/surrogate"]) == 3
  assert len(data["learning_rate"]) == 3


def test_enable_terminal_log_null_padding(tmp_path):
  """Keys that appear mid-run are back-filled with null."""
  logger = _make_logger()  # Empty rewbuffer.
  logger.log_dir = str(tmp_path)
  logger.writer = MagicMock()

  enable_terminal_log(logger)

  action_std = torch.tensor([0.5])
  # Iteration 0: no episodes completed yet.
  logger.log(
    it=0,
    start_it=0,
    total_it=10,
    collect_time=0.5,
    learn_time=0.3,
    loss_dict={"surrogate": 0.01},
    learning_rate=3e-4,
    action_std=action_std,
    rnd_weight=None,
  )

  # Simulate episodes completing.
  logger.rewbuffer = deque([5.0])
  logger.lenbuffer = deque([100.0])
  logger.ep_extras = [{"custom_metric": torch.tensor([1.0])}]

  # Iteration 1: now has reward + extras.
  logger.log(
    it=1,
    start_it=0,
    total_it=10,
    collect_time=0.5,
    learn_time=0.3,
    loss_dict={"surrogate": 0.02},
    learning_rate=3e-4,
    action_std=action_std,
    rnd_weight=None,
  )

  json_path = tmp_path / "training_log.json"
  data = json.loads(json_path.read_text(encoding="utf-8"))

  # mean_reward: null for iter 0, value for iter 1.
  assert data["mean_reward"] == [None, 5.0]
  # episode/custom_metric didn't exist at iter 0.
  assert data["Episode/custom_metric"] == [None, 1.0]
  # All arrays same length.
  lengths = {len(v) for v in data.values()}
  assert lengths == {2}
