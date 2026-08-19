"""Tests for the optional live view on an evaluation run.

No viser server is started here. Standing one up in a test means binding a port
and serving assets, which is slow and flaky on shared machines, and it would not
test anything this module owns: the parts worth pinning are that the off path
costs nothing, that the pacing arithmetic is right, and that the callback does
not disturb the run. The scene sync itself belongs to mjviser.

The end-to-end check -- server reachable, browser page served, metrics identical
to a headless run -- is done by hand; see ``scripts/eval/README.md``.
"""

from dataclasses import dataclass

import pytest

from mjlab.evaluation.live_view import (
  DEFAULT_VISER_PORT,
  LiveView,
  LiveViewCfg,
  RealTimePacer,
  open_live_view,
)


@dataclass
class FakeHarness:
  """Enough of the harness protocol to construct a view against."""

  num_envs: int = 1
  control_dt: float = 0.01

  @property
  def sim(self):
    raise AssertionError("the live view must not touch the sim before stepping")


class FakeClock:
  """A clock that only moves when told to, and a sleep that records."""

  def __init__(self) -> None:
    self.now = 0.0
    self.slept: list[float] = []

  def tick(self, seconds: float) -> None:
    self.now += seconds

  def clock(self) -> float:
    return self.now

  def sleep(self, seconds: float) -> None:
    self.slept.append(seconds)
    self.now += seconds


def test_pacer_waits_out_the_lead():
  """Stepping faster than real time sleeps off the difference."""
  clock = FakeClock()
  pacer = RealTimePacer(0.01, clock=clock.clock, sleep=clock.sleep)

  # A step that took no time at all owes the whole control period.
  assert pacer.wait() == pytest.approx(0.01)
  # The next one is measured from the same start, so a step that took 4 ms of
  # real work owes the remaining 6 ms rather than another full period.
  clock.tick(0.004)
  assert pacer.wait() == pytest.approx(0.006)
  assert clock.slept == pytest.approx([0.01, 0.006])
  assert clock.now == pytest.approx(0.02), "wall clock tracks simulated time"


def test_pacer_does_not_wait_when_already_behind():
  """A slow simulator is never slowed further."""
  clock = FakeClock()
  pacer = RealTimePacer(0.01, clock=clock.clock, sleep=clock.sleep)

  clock.tick(0.5)
  assert pacer.wait() == 0.0
  clock.tick(0.5)
  assert pacer.wait() == 0.0
  assert clock.slept == []


def test_pacer_recovers_rather_than_accumulating_drift():
  """One slow step does not put every later step permanently in debt."""
  clock = FakeClock()
  pacer = RealTimePacer(0.01, clock=clock.clock, sleep=clock.sleep)

  clock.tick(0.1)  # one very slow step
  assert pacer.wait() == 0.0
  # Ten more instant steps: simulated time is 0.11 s, wall time is 0.1 s, so
  # only the last one owes anything.
  for _ in range(10):
    pacer.wait()
  assert sum(clock.slept) == pytest.approx(0.01, abs=1e-9)


def test_disabled_config_starts_nothing():
  """The default path never constructs a view, so it never imports viser."""
  assert LiveViewCfg().viser is False
  assert open_live_view(FakeHarness(), LiveViewCfg()) is None


def test_config_defaults_are_the_documented_ones():
  cfg = LiveViewCfg()
  assert cfg.viser_port == DEFAULT_VISER_PORT
  assert cfg.viser_env == 0
  assert cfg.viser_realtime is True


def test_out_of_range_env_index_is_rejected_before_any_server_starts():
  """The check runs before viser is imported, so it is cheap and safe."""
  with pytest.raises(ValueError, match="outside the batch"):
    LiveView(FakeHarness(num_envs=1), env_index=3)
  with pytest.raises(ValueError, match="outside the batch"):
    LiveView(FakeHarness(num_envs=4), env_index=-1)
