"""Bus-voltage model tests (doc 17 power network)."""

from unittest.mock import MagicMock

import torch

from mjlab.tasks.velocity.mdp.bus_voltage import (
  bus_voltage_step,
  servo_voltage,
)

NU = 4


def _env(n: int = 2) -> MagicMock:
  env = MagicMock()
  env.num_envs = n
  env.device = "cpu"
  env.step_dt = 0.02
  env.episode_length_buf = torch.zeros(n, dtype=torch.long)
  asset = MagicMock()
  asset.data.actuator_force = torch.zeros(n, NU)
  asset.actuator_names = tuple(f"m{i}" for i in range(NU))
  asset.indexing.ctrl_ids = torch.arange(NU)
  env.scene = {"robot": asset}
  env.sim.model.actuator_forcerange = torch.tensor([[[-10.0, 10.0]] * NU] * n)
  return env


KT = {"default": 2.68}


def test_voltage_sags_with_fleet_load_and_chain_position() -> None:
  env = _env()
  bus_voltage_step(env, None, KT)  # init/resample at zero load
  env.episode_length_buf[:] = 1
  env.scene["robot"].data.actuator_force = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0], [8.0, 8.0, 8.0, 8.0]]
  )
  bus_voltage_step(env, None, KT)
  v = servo_voltage(env)
  # Loaded env sags below the idle env on every servo (shared term).
  assert (v[1] < v[0]).all()
  # And torque authority shrinks with it.
  fr = env.sim.model.actuator_forcerange
  assert (fr[1, :, 1] < fr[0, :, 1]).all()


def test_no_ratchet_across_steps_and_episodes() -> None:
  """R29 regression: NOTHING restores actuator_forcerange at reset in the
  real config (effort_limits DR goes through actuator.set_effort_limit),
  so the live field a fresh env carries into its next episode is our own
  scaled write. 1000 steps with frequent falls (2 s episodes, worst case)
  must leave forcerange = original_base * scale, never compounding. The
  original version of this test restored the field on reset - modeling
  the assumption instead of reality - and green-lit the ratchet that
  melted v48's first launch to zero authority."""
  env = _env(1)
  env.scene["robot"].data.actuator_force = torch.full((1, NU), 5.0)
  bus_voltage_step(env, None, KT)
  for step in range(1000):
    # Short episodes (fall every 100 steps); reset does NOT touch the
    # forcerange field - exactly like the real NUgus event stack.
    env.episode_length_buf[:] = (step + 1) % 100
    bus_voltage_step(env, None, KT)
  fr = env.sim.model.actuator_forcerange[0, 0, 1]
  # Worst case: 10 * clamp floor 0.3; never below, never spiraling.
  assert 3.0 <= fr <= 11.5


def test_base_cached_once_never_rereads_live() -> None:
  """The base must be frozen at the first-call (pre-write) values; later
  steps, loads, and resets must never fold the scaled live field back in."""
  from mjlab.tasks.velocity.mdp.bus_voltage import _STATE

  env = _env(1)
  bus_voltage_step(env, None, KT)
  base0 = _STATE[env].base_forcerange.clone()
  assert torch.equal(base0, torch.tensor([[[-10.0, 10.0]] * NU]))
  env.scene["robot"].data.actuator_force = torch.full((1, NU), 8.0)
  for k in [1, 2, 3, 0, 1, 2]:  # includes a reset (wrap to 0)
    env.episode_length_buf[:] = k
    bus_voltage_step(env, None, KT)
  assert torch.equal(_STATE[env].base_forcerange, base0)


def test_discharge_drains_over_episode() -> None:
  env = _env(1)
  bus_voltage_step(env, None, KT)
  v0 = servo_voltage(env).clone()
  for k in range(1, 500):
    env.episode_length_buf[:] = k
    bus_voltage_step(env, None, KT)
  v1 = servo_voltage(env)
  assert (v1 < v0).all()  # ~10 s at 40-90 mV/min: strictly lower
