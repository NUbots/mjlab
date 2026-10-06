"""Tests for the system identification command sequences and frame transform."""

import math

import numpy as np
import pytest
import torch

from mjlab.evaluation.sysid import (
  COLUMNS,
  RAW_STATE_COLUMNS,
  Trace,
  heading_frame,
  run_table,
  write_run,
)
from mjlab.evaluation.sysid_sequences import (
  AXES,
  LEAD_IN_S,
  MULTILEVEL_MAX_FRACTION,
  build_sequences,
  scale,
)

RANGES = {"vx": (-0.9, 0.9), "vy": (-0.3, 0.3), "wz": (-0.5, 0.5)}
DT = 0.02


@pytest.fixture(scope="module")
def sequences():
  return {sequence.name: sequence for sequence in build_sequences(RANGES, DT)}


def test_every_sequence_starts_standing(sequences):
  lead = round(LEAD_IN_S / DT)
  for sequence in sequences.values():
    assert np.all(sequence.commands[:lead] == 0.0), sequence.name
    assert np.all(sequence.commands[-1] == 0.0), sequence.name


def test_step_levels_are_fractions_of_range(sequences):
  step = sequences["step_vy_-075"]
  lead = round(LEAD_IN_S / DT)
  assert step.commands[lead, AXES.index("vy")] == pytest.approx(-0.225)
  assert np.all(step.commands[:, [0, 2]] == 0.0)
  assert step.duration == pytest.approx(10.0)


def test_asymmetric_range_scales_each_sign_separately():
  ranges = {"vx": (-0.5, 1.0), "vy": (-0.3, 0.3), "wz": (-0.5, 0.5)}
  assert scale(0.5, "vx", ranges) == pytest.approx(0.5)
  assert scale(-0.5, "vx", ranges) == pytest.approx(-0.25)


def test_multilevel_staircases(sequences):
  identification = [s for s in sequences.values() if s.family == "multilevel"]
  assert (
    sorted(s.split for s in identification)
    == ["identification"] * 3 + ["validation"] * 3
  )
  seeds = {s.params["seed"] for s in identification}
  assert len(seeds) == 6
  for sequence in identification:
    axis = AXES.index(sequence.params["axis"])
    bound = MULTILEVEL_MAX_FRACTION * RANGES[sequence.params["axis"]][1]
    assert np.abs(sequence.commands[:, axis]).max() <= bound + 1e-12
    holds = [level["hold_s"] for level in sequence.params["levels"]]
    assert sum(holds) == pytest.approx(120.0)
    assert all(hold <= 3.0 + 1e-9 for hold in holds)
    assert all(hold >= 0.5 - DT for hold in holds[:-1])


def test_multilevel_is_reproducible():
  first = build_sequences(RANGES, DT, families=("multilevel",))
  second = build_sequences(RANGES, DT, families=("multilevel",))
  for a, b in zip(first, second, strict=True):
    assert np.array_equal(a.commands, b.commands)


def test_chirp_amplitude_and_ends(sequences):
  chirp = sequences["chirp_wz"]
  values = chirp.commands[:, AXES.index("wz")]
  assert np.abs(values).max() == pytest.approx(0.25, rel=1e-3)
  assert chirp.duration == pytest.approx(LEAD_IN_S + 60.0 + 3.0)


def test_combined_holds_other_axis(sequences):
  run = sequences["combined_vy_+100_hold_vx_-050"]
  onset = round(run.params["step_onset_s"] / DT)
  assert run.commands[onset].tolist() == pytest.approx([-0.45, 0.3, 0.0])
  assert run.commands[onset - 1].tolist() == pytest.approx([-0.45, 0.0, 0.0])


def test_heading_frame_removes_yaw_only():
  yaw, pitch = 0.7, 0.2
  # Quaternion for yaw about z followed by pitch about the body y axis.
  qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  qy = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
  w1, x1, y1, z1 = qz
  w2, x2, y2, z2 = qy
  quat = [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ]
  speed = 0.4
  lin_w = [speed * math.cos(yaw), speed * math.sin(yaw), 0.0]
  # A pure world-z rotation, expressed in the pitched body frame.
  rate = 0.3
  ang_b = [-rate * math.sin(pitch), 0.0, rate * math.cos(pitch)]
  root = torch.tensor([[1.0, 2.0, 0.5, *quat, *lin_w, *ang_b]], dtype=torch.float64)
  frame = heading_frame(root)
  assert float(frame["yaw"]) == pytest.approx(yaw)
  assert float(frame["vx"]) == pytest.approx(speed)
  assert float(frame["vy"]) == pytest.approx(0.0, abs=1e-12)
  assert float(frame["wz"]) == pytest.approx(rate)


def test_run_table_keeps_raw_state(tmp_path):
  rows, num_envs = 8, 2
  root = torch.randn(rows, num_envs, 13)
  root[..., 3:7] /= root[..., 3:7].norm(dim=-1, keepdim=True)
  trace = Trace(root, torch.zeros(rows, num_envs, 3), dt=0.005)
  push_flag = np.zeros((2, num_envs), dtype=np.int8)
  push_flag[1, 1] = 1
  table = run_table(trace, 1, 5, 4, push_flag, rows_per_push_step=4)
  assert list(table) == list(COLUMNS)
  for key, column in RAW_STATE_COLUMNS:
    assert np.allclose(table[key], root[:5, 1, column].numpy()), key
  assert table["fall"].tolist() == [0, 0, 0, 0, 1]
  assert table["push"].tolist() == [0, 0, 0, 0, 1]
  (written,) = write_run(tmp_path / "env000", table, {"fell": 1.0}, "csv")
  loaded = np.loadtxt(tmp_path / written, delimiter=",", skiprows=1)
  assert loaded.shape == (5, len(COLUMNS))
