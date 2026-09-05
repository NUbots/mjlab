"""Tests for the competence-grid plotter's two pieces of real logic.

Layout is not tested -- it is looked at. What is tested is the arithmetic the
figures rest on: the robust half-width a diverging scale is drawn with, and the
promise that narrowing a comparison does not repaint the runs that survive.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from plot_competence_grid import (  # noqa: E402
  cells_at,
  diverging_span,
  load_runs,
  min_episodes,
)


def write_run(directory: Path, cells: list[dict]) -> None:
  directory.mkdir(parents=True, exist_ok=True)
  with (directory / "cells.json").open("w") as handle:
    json.dump({"run": {"plant": "eval"}, "cells": cells}, handle)


def cell(vx: float, shove: float, episodes: int = 8) -> dict:
  return {
    "vx": vx,
    "vy": 0.0,
    "wz": 0.0,
    "shove": shove,
    "episodes": episodes,
    "fell_rate": 0.0,
    "fell_ci_low": 0.0,
    "fell_ci_high": 0.2,
    "attain": {"n": episodes, "median": 1.0, "q25": 0.9, "q75": 1.1, "iqr": 0.2},
  }


def test_diverging_span_is_not_set_by_a_single_outlier():
  """One collapsed cell must not push every other one into the neutral third."""
  values = [np.full(50, 1.05), np.array([0.02])]
  span = diverging_span(values, centre=1.0)

  assert span < 0.2
  assert span == pytest.approx(0.05, abs=0.02)


def test_diverging_span_survives_an_all_nan_grid():
  span = diverging_span([np.full(4, np.nan)], centre=0.0)
  assert span > 0.0


def test_narrowing_a_comparison_keeps_each_run_its_colour(tmp_path: Path):
  """Colour follows the run, never its rank in the drawn set."""
  for name in ("alpha", "beta", "gamma"):
    write_run(tmp_path / name, [cell(0.5, 0.0)])

  everything = {run.name: run.colour for run in load_runs(tmp_path, None)}
  narrowed = load_runs(tmp_path, "gamma,alpha")

  assert [run.name for run in narrowed] == ["gamma", "alpha"]
  assert [run.colour for run in narrowed] == [everything["gamma"], everything["alpha"]]


def test_a_missing_run_is_named(tmp_path: Path):
  write_run(tmp_path / "alpha", [cell(0.5, 0.0)])
  with pytest.raises(KeyError, match="delta"):
    load_runs(tmp_path, "delta")


def test_an_empty_directory_says_what_it_expected(tmp_path: Path):
  with pytest.raises(FileNotFoundError, match="cells.json"):
    load_runs(tmp_path, None)


def test_cells_at_orders_by_shove_and_ignores_other_commands(tmp_path: Path):
  write_run(
    tmp_path / "alpha",
    [cell(0.5, 0.8), cell(0.5, 0.0), cell(0.25, 0.4)],
  )
  run = load_runs(tmp_path, None)[0]

  cells = cells_at(run, (0.5, 0.0, 0.0))
  assert [c["shove"] for c in cells] == [0.0, 0.8]
  assert cells_at(run, (0.9, 0.0, 0.0)) == []


def test_min_episodes_reports_the_worst_covered_cell():
  assert min_episodes([cell(0.5, 0.0, 40), cell(0.5, 0.8, 9)]) == 9
  assert min_episodes([]) == 0
