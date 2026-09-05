"""Tests for the per-episode competence metrics, the grid and the shove driver.

No simulator: the grid is arithmetic, the collector is a recorder fed
hand-built states, and the driver writes a velocity onto a stub. What is worth
checking is that the definitions match the training tracker's -- the same tilt
bound, the same meaningful-command filter, the same per-axis energy weighting
-- and that nothing carries across a reset.
"""

import math

import pytest
import torch

from mjlab.evaluation.competence import (
  FELL_OVER_UPRIGHT,
  MIN_AXIS_COMMAND,
  MIN_COMMAND_NORM,
  WOBBLE_GRAVITY_XY,
  EpisodeCompetence,
  GridCell,
  ShoveCfg,
  ShoveDriver,
  build_grid,
  episode_end,
  projected_gravity_xy_norm,
  summarise_cells,
  write_episodes_csv,
)
from mjlab.evaluation.metrics import FALL_UPRIGHT_THRESHOLD, EvalState

DT = 0.02
MAX_STEPS = 50


def state(
  quaternion: torch.Tensor | None = None,
  lin_vel_b: torch.Tensor | None = None,
  num_envs: int = 1,
) -> EvalState:
  """An :class:`EvalState` carrying only what the competence metrics read."""
  if quaternion is None:
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
  if lin_vel_b is None:
    lin_vel_b = torch.zeros(num_envs, 3)
  return EvalState(
    position_w=torch.zeros(num_envs, 3),
    quaternion_w=quaternion,
    lin_vel_b=lin_vel_b,
    ang_vel_b=torch.zeros(num_envs, 3),
  )


def pitched(angle: float, num_envs: int = 1) -> torch.Tensor:
  """A quaternion pitched ``angle`` radians about the body y axis."""
  half = angle / 2.0
  return torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]]).repeat(num_envs, 1)


def collector(
  commands: tuple[tuple[float, float, float], ...],
  shoves: tuple[float, ...] = (0.0,),
) -> EpisodeCompetence:
  grid = build_grid(commands, shoves, num_envs=len(commands) * len(shoves))
  return EpisodeCompetence(grid, max_episode_steps=MAX_STEPS, step_dt=DT, device="cpu")


def run(
  competence: EpisodeCompetence,
  steps: int,
  *,
  quaternion: torch.Tensor | None = None,
  lin_vel_b: torch.Tensor | None = None,
  fell: bool = False,
) -> None:
  """Record ``steps`` identical steps, the last of them ending the episode."""
  n = competence.grid.num_envs
  for step in range(steps):
    done = torch.full((n,), step == steps - 1)
    competence.record(
      state(quaternion=quaternion, lin_vel_b=lin_vel_b, num_envs=n),
      done,
      done & fell,
    )


# --------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------


def test_wobble_bound_is_the_trackers_twenty_five_degrees():
  """The literal the training tracker compares against is sin(25 degrees)."""
  assert WOBBLE_GRAVITY_XY == pytest.approx(math.sin(math.radians(25.0)), abs=5e-5)


def test_projected_gravity_from_quaternion_is_the_sine_of_the_tilt():
  angles = torch.tensor([0.0, 0.1, 0.4363, 1.0])
  tilt = projected_gravity_xy_norm(pitched(0.0))
  assert float(tilt) == pytest.approx(0.0, abs=1e-6)
  for angle in angles:
    tilt = projected_gravity_xy_norm(pitched(float(angle)))
    assert float(tilt) == pytest.approx(math.sin(float(angle)), abs=1e-6)


def test_attain_is_delivered_speed_projected_on_the_command():
  """Half the commanded speed, delivered along the command, reads 0.5."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, lin_vel_b=torch.tensor([[0.2, 0.0, 0.0]]))

  table = competence.table()
  assert float(table.attain[0]) == pytest.approx(0.5)


def test_attain_ignores_velocity_orthogonal_to_the_command():
  """Lateral sway is orthogonal to a forward command, so it drops out."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, lin_vel_b=torch.tensor([[0.4, 0.9, 0.0]]))

  assert float(competence.table().attain[0]) == pytest.approx(1.0)


def test_attain_is_undefined_below_the_meaningful_command_filter():
  """A command under |c| >= 0.15 takes no attainment sample at all.

  Not zero: zero is the reading for a robot that was asked to move and did not.
  A cell the instrument was switched off in has to say so, or it will be read
  as the worst sandbagging on the grid.
  """
  below = MIN_COMMAND_NORM - 0.01
  competence = collector(((below, 0.0, 0.0), (0.0, 0.0, 0.4)))
  run(competence, 10, lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]]))

  table = competence.table()
  assert bool(table.attain.isnan().all())
  # The quantities that do not depend on a command are still measured.
  assert bool(table.wobble.eq(0.0).all())
  assert bool(table.fell.eq(0.0).all())


def test_attain_axes_are_signed_and_separate():
  """Asked for lateral, delivered forward: the scalar hides it, the axes do not."""
  competence = collector(((0.0, 0.4, 0.0),))
  run(competence, 10, lin_vel_b=torch.tensor([[0.5, -0.2, 0.0]]))

  table = competence.table()
  # The command is all y, so the x axis takes no sample and y carries it all.
  assert bool(table.attain_x.isnan().all())
  assert float(table.attain_y[0]) == pytest.approx(-0.5)
  # Backpedalling against the command reads negative in the scalar too.
  assert float(table.attain[0]) == pytest.approx(-0.5)


def test_attain_axis_below_its_own_floor_takes_no_sample():
  below = MIN_AXIS_COMMAND - 0.01
  competence = collector(((0.5, below, 0.0),))
  run(competence, 10, lin_vel_b=torch.tensor([[0.5, 0.3, 0.0]]))

  table = competence.table()
  assert float(table.attain_x[0]) == pytest.approx(1.0)
  assert bool(table.attain_y.isnan().all())


def test_wobble_is_the_fraction_of_steps_past_the_bound():
  """Six tilted steps in ten, with the tenth ending the episode."""
  competence = collector(((0.4, 0.0, 0.0),))
  upright = pitched(0.0)
  tilted = pitched(math.radians(30.0))
  for step in range(10):
    done = torch.tensor([step == 9])
    competence.record(state(quaternion=tilted if step < 6 else upright), done, done)

  # The step the episode ends on is excluded from the averages, so the
  # denominator is nine.
  assert float(competence.table().wobble[0]) == pytest.approx(6.0 / 9.0)


def test_a_tilt_just_under_the_bound_does_not_count():
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, quaternion=pitched(math.radians(24.0)))
  assert float(competence.table().wobble[0]) == pytest.approx(0.0)

  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, quaternion=pitched(math.radians(26.0)))
  assert float(competence.table().wobble[0]) == pytest.approx(1.0)


def test_ep_len_frac_counts_the_terminal_step():
  """Survival is the episode's length, including the step it ended on."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 20, fell=True)

  table = competence.table()
  assert float(table.ep_len_frac[0]) == pytest.approx(20.0 / MAX_STEPS)
  assert float(table.fell[0]) == 1.0


def test_a_timed_out_episode_reads_full_survival_and_no_fall():
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, MAX_STEPS, fell=False)

  table = competence.table()
  assert float(table.ep_len_frac[0]) == pytest.approx(1.0)
  assert float(table.fell[0]) == 0.0


# --------------------------------------------------------------------------
# Episode independence
# --------------------------------------------------------------------------


def test_nothing_carries_across_a_reset():
  """A wobbling episode must not contaminate the calm one after it."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, quaternion=pitched(math.radians(40.0)), fell=True)
  run(competence, 10, quaternion=pitched(0.0), fell=False)

  table = competence.table()
  assert table.num_episodes == 2
  assert float(table.wobble[0]) == pytest.approx(1.0)
  assert float(table.wobble[1]) == pytest.approx(0.0)
  assert float(table.fell[0]) == 1.0
  assert float(table.fell[1]) == 0.0
  assert float(table.ep_len_frac[0]) == pytest.approx(10.0 / MAX_STEPS)
  assert float(table.ep_len_frac[1]) == pytest.approx(10.0 / MAX_STEPS)


def test_episodes_still_in_flight_are_dropped():
  """An episode truncated by the end of the run has a censored length."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10, fell=False)
  n = competence.grid.num_envs
  for _ in range(5):
    competence.record(
      state(num_envs=n),
      torch.zeros(n, dtype=torch.bool),
      torch.zeros(n, dtype=torch.bool),
    )

  assert competence.table().num_episodes == 1


def test_environments_end_episodes_independently():
  """Two cells, one falling every five steps, the other running clean."""
  competence = collector(((0.4, 0.0, 0.0), (0.5, 0.0, 0.0)))
  for step in range(10):
    done = torch.tensor([(step + 1) % 5 == 0, step == 9])
    competence.record(state(num_envs=2), done, done & torch.tensor([True, False]))

  table = competence.table()
  assert table.num_episodes == 3
  assert sorted(table.cell.tolist()) == [0, 0, 1]
  assert competence.completed_per_cell.tolist() == [2, 1]
  assert competence.min_completed == 1


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


def test_grid_crosses_commands_with_shoves():
  grid = build_grid(((0.3, 0.0, 0.0), (0.0, 0.3, 0.0)), (0.0, 0.4), num_envs=8)

  assert len(grid.cells) == 4
  assert grid.envs_per_cell().tolist() == [2, 2, 2, 2]
  assert grid.cells[0] == GridCell(vx=0.3, vy=0.0, wz=0.0, shove=0.0)
  assert grid.cells[3] == GridCell(vx=0.0, vy=0.3, wz=0.0, shove=0.4)


def test_grid_is_tiled_so_a_short_batch_loses_slots_not_cells():
  grid = build_grid(((0.3, 0.0, 0.0), (0.0, 0.3, 0.0)), (0.0, 0.4), num_envs=5)

  counts = grid.envs_per_cell().tolist()
  assert counts == [2, 1, 1, 1]
  assert min(counts) >= 1


def test_a_batch_smaller_than_the_grid_is_refused():
  with pytest.raises(ValueError, match="cannot cover"):
    build_grid(((0.3, 0.0, 0.0),), (0.0, 0.2, 0.4, 0.6), num_envs=3)


def test_a_cell_knows_whether_attainment_is_defined_in_it():
  assert GridCell(0.3, 0.0, 0.0, 0.0).attain_defined
  assert not GridCell(0.0, 0.0, 0.5, 0.0).attain_defined
  assert GridCell(0.1, 0.12, 0.0, 0.0).attain_defined  # the norm, not the axes


# --------------------------------------------------------------------------
# The shove driver
# --------------------------------------------------------------------------


class StubRobot:
  """Records what was written, and reports zero velocity."""

  def __init__(self, num_envs: int) -> None:
    self.data = self
    self.root_link_vel_w = torch.zeros(num_envs, 6)
    self.writes: list[tuple[torch.Tensor, torch.Tensor]] = []

  def write_root_link_velocity_to_sim(self, velocity, env_ids):
    self.writes.append((velocity.clone(), env_ids.clone()))


def test_onsets_settle_first_and_leave_a_tail():
  cfg = ShoveCfg(settle=3.0, period=4.0, tail=2.0)
  onsets = cfg.onsets(dt=0.02, max_episode_steps=1000)

  assert onsets == (150, 350, 550, 750)
  assert onsets[0] * 0.02 == pytest.approx(3.0)
  assert onsets[-1] * 0.02 <= 1000 * 0.02 - 2.0


def test_shove_magnitude_is_the_cell_and_only_the_heading_is_drawn():
  magnitude = torch.tensor([0.0, 0.3, 0.6])
  robot = StubRobot(3)
  driver = ShoveDriver(magnitude, robot, DT, MAX_STEPS, ShoveCfg(0.1, 0.2, 0.1))

  first = driver.onsets[0]
  ids = driver.apply(torch.full((3,), first, dtype=torch.long))

  # The zero bin is never written to: it is the undisturbed row.
  assert ids.tolist() == [1, 2]
  velocity, written = robot.writes[0]
  assert written.tolist() == [1, 2]
  delivered = velocity[:, :2].norm(dim=-1)
  assert delivered.tolist() == pytest.approx([0.3, 0.6], abs=1e-5)
  # Planar only: including the roll and pitch kicks the training event applies
  # would mean |dv_xy| no longer described the whole disturbance.
  assert bool(velocity[:, 2:].eq(0.0).all())


def test_shoves_land_only_on_the_onset_step():
  robot = StubRobot(1)
  driver = ShoveDriver(
    torch.tensor([0.4]), robot, DT, MAX_STEPS, ShoveCfg(0.1, 0.2, 0.1)
  )

  for step in range(MAX_STEPS):
    driver.apply(torch.tensor([step]))

  assert driver.delivered == len(driver.onsets)
  assert len(robot.writes) == len(driver.onsets)


def test_an_episode_too_short_for_a_shove_is_refused():
  with pytest.raises(ValueError, match="no shove fits"):
    ShoveDriver(torch.tensor([0.4]), StubRobot(1), DT, 10, ShoveCfg(3.0, 4.0, 2.0))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_cell_summary_reports_spread_and_marks_undefined_attainment():
  competence = collector(((0.4, 0.0, 0.0), (0.0, 0.0, 0.4)))
  speeds = (0.1, 0.2, 0.3, 0.4)
  for speed in speeds:
    for step in range(5):
      done = torch.tensor([step == 4, step == 4])
      competence.record(
        state(lin_vel_b=torch.tensor([[speed, 0.0, 0.0], [0.0, 0.0, 0.0]]), num_envs=2),
        done,
        done & torch.tensor([False, True]),
      )

  records = summarise_cells(competence.grid, competence.table())
  walking, turning = records

  assert walking["attain_defined"]
  assert walking["episodes"] == len(speeds)
  assert walking["attain"]["median"] == pytest.approx(0.625)
  assert walking["attain"]["iqr"] > 0.0
  assert walking["fell_rate"] == 0.0

  assert not turning["attain_defined"]
  assert turning["attain"]["n"] == 0
  assert math.isnan(turning["attain"]["median"])
  # A cell with no attainment evidence is still read through falls and wobble.
  assert turning["fell_rate"] == 1.0
  assert turning["wobble"]["n"] == len(speeds)


def test_fall_rate_carries_a_binomial_interval():
  """Zero falls in four episodes and zero in forty are different claims."""
  competence = collector(((0.4, 0.0, 0.0),))
  for _ in range(4):
    run(competence, 5, fell=False)
  narrow = summarise_cells(competence.grid, competence.table())[0]

  competence = collector(((0.4, 0.0, 0.0),))
  for _ in range(40):
    run(competence, 5, fell=False)
  wide = summarise_cells(competence.grid, competence.table())[0]

  assert narrow["fell_rate"] == wide["fell_rate"] == 0.0
  assert narrow["fell_ci_high"] > wide["fell_ci_high"]


# --------------------------------------------------------------------------
# The termination rule, for harnesses with no termination manager
# --------------------------------------------------------------------------


def test_fall_bound_is_the_tasks_fifty_degrees_not_the_metrics_sixty():
  """The two bounds answer different questions and must not be conflated.

  ``FALL_UPRIGHT_THRESHOLD`` dates the moment a robot stopped walking, and is
  deliberately generous. ``fell`` has to mean the episode the policy would have
  been terminated for, or a scripted engine gets to tip ten degrees further
  than a policy before it counts.
  """
  assert FELL_OVER_UPRIGHT == pytest.approx(math.cos(math.radians(50.0)))
  assert FELL_OVER_UPRIGHT > FALL_UPRIGHT_THRESHOLD


def test_a_tilt_between_the_two_bounds_is_a_fall_here():
  """55 degrees: past the task's termination, short of the metrics' bound."""
  step = torch.tensor([1])
  done, fell = episode_end(state(quaternion=pitched(math.radians(55.0))), step, 100)

  assert bool(fell.all())
  assert bool(done.all())


def test_a_tilt_inside_the_bound_is_not_a_fall():
  step = torch.tensor([1])
  done, fell = episode_end(state(quaternion=pitched(math.radians(45.0))), step, 100)

  assert not bool(fell.any())
  assert not bool(done.any())


def test_reaching_the_episode_length_ends_it_without_a_fall():
  done, fell = episode_end(state(), torch.tensor([100]), 100)

  assert bool(done.all())
  assert not bool(fell.any())


def test_tipping_on_the_last_step_is_recorded_as_a_fall():
  """Ordering matters: the termination manager resolves failure before timeout."""
  done, fell = episode_end(
    state(quaternion=pitched(math.radians(60.0))), torch.tensor([100]), 100
  )

  assert bool(done.all())
  assert bool(fell.all())


def test_environments_end_independently():
  quats = torch.cat([pitched(math.radians(60.0)), pitched(0.0), pitched(0.0)])
  done, fell = episode_end(
    state(quaternion=quats, num_envs=3), torch.tensor([5, 100, 5]), 100
  )

  assert done.tolist() == [True, True, False]
  assert fell.tolist() == [True, False, False]


# --------------------------------------------------------------------------
# The raw step counts and which steps wobbled
# --------------------------------------------------------------------------


def test_steps_is_the_denominator_the_wobble_fraction_used():
  """``num_wobble_steps / steps`` has to reproduce ``wobble`` exactly.

  Otherwise the raw counts and the fraction are two different measurements
  wearing one name.
  """
  competence = collector(((0.4, 0.0, 0.0),))
  tilted = pitched(math.radians(30.0))
  for step in range(10):
    done = torch.tensor([step == 9])
    competence.record(
      state(quaternion=tilted if step < 6 else pitched(0.0)), done, done
    )

  table = competence.table()
  # Nine sampled steps: the tenth ended the episode and is excluded.
  assert float(table.steps[0]) == 9.0
  assert float(table.num_wobble_steps[0]) == 6.0
  assert float(table.num_wobble_steps[0] / table.steps[0]) == pytest.approx(
    float(table.wobble[0])
  )


def test_ep_len_is_seconds_and_counts_the_terminal_step():
  """Steps are rate-dependent; seconds are what two controllers share."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 20, fell=True)

  table = competence.table()
  assert float(table.ep_len[0]) == pytest.approx(20 * DT)
  # The episode length counts the terminal step, the averages do not.
  assert float(table.steps[0]) == 19.0
  assert float(table.ep_len_frac[0]) == pytest.approx(20.0 / MAX_STEPS)


def test_wobble_indices_name_the_steps_that_wobbled():
  """0-based within the episode, in order."""
  competence = collector(((0.4, 0.0, 0.0),))
  wobbly = {2, 3, 7}
  for step in range(10):
    done = torch.tensor([step == 9])
    quat = pitched(math.radians(30.0)) if step in wobbly else pitched(0.0)
    competence.record(state(quaternion=quat), done, done)

  table = competence.table()
  assert len(table.wobble_steps_index) == 1
  assert table.wobble_steps_index[0].tolist() == sorted(wobbly)
  assert float(table.num_wobble_steps[0]) == len(wobbly)


def test_an_episode_that_never_wobbled_has_an_empty_index():
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 10)

  table = competence.table()
  assert table.wobble_steps_index[0].numel() == 0
  assert float(table.num_wobble_steps[0]) == 0.0


def test_wobble_indices_do_not_carry_across_a_reset():
  """The bitmap is per episode, like every other accumulator."""
  competence = collector(((0.4, 0.0, 0.0),))
  run(competence, 4, quaternion=pitched(math.radians(40.0)), fell=True)
  run(competence, 6, quaternion=pitched(0.0))

  table = competence.table()
  assert table.num_episodes == 2
  # Three sampled steps in the first episode, all of them tilted.
  assert table.wobble_steps_index[0].tolist() == [0, 1, 2]
  assert table.wobble_steps_index[1].numel() == 0


def test_indices_stay_with_their_own_episode_when_several_end_at_once():
  """Two environments closing on one step must not swap index lists."""
  competence = collector(((0.4, 0.0, 0.0), (0.5, 0.0, 0.0)))
  calm, tilted = pitched(0.0), pitched(math.radians(30.0))
  for step in range(6):
    done = torch.tensor([step == 5, step == 5])
    # Only the second environment wobbles, and only on steps 1 and 3.
    quats = torch.cat([calm, tilted if step in (1, 3) else calm])
    competence.record(state(quaternion=quats, num_envs=2), done, done)

  table = competence.table()
  by_cell = {
    int(c): idx for c, idx in zip(table.cell, table.wobble_steps_index, strict=True)
  }
  assert by_cell[0].tolist() == []
  assert by_cell[1].tolist() == [1, 3]


def test_the_ragged_column_renders_as_space_separated_integers(tmp_path):
  """One row per episode in the CSV, and no quoting needed."""
  competence = collector(((0.4, 0.0, 0.0),))
  for step in range(6):
    done = torch.tensor([step == 5])
    quat = pitched(math.radians(30.0)) if step in (0, 4) else pitched(0.0)
    competence.record(state(quaternion=quat), done, done)

  path = tmp_path / "episodes.csv"
  write_episodes_csv(path, competence.table())
  header, row = path.read_text().splitlines()

  columns = dict(zip(header.split(","), row.split(","), strict=True))
  assert columns["wobble_steps_index"] == "0 4"
  assert columns["num_wobble_steps"] == "2.0"
  assert "," not in columns["wobble_steps_index"]
