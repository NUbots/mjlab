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
  """R29 regression: the scale multiplies a cached per-episode base;
  1000 steps and several resets must leave forcerange = base * scale,
  never compounding."""
  env = _env(1)
  env.scene["robot"].data.actuator_force = torch.full((1, NU), 5.0)
  bus_voltage_step(env, None, KT)
  for step in range(1000):
    env.episode_length_buf[:] = (step % 200) + 1
    if step % 200 == 199:
      # Episode reset: DR restore writes a clean value.
      env.episode_length_buf[:] = 0
      env.sim.model.actuator_forcerange[:] = torch.tensor([[[-10.0, 10.0]] * NU])
    bus_voltage_step(env, None, KT)
  fr = env.sim.model.actuator_forcerange[0, 0, 1]
  # Worst case: 10 * clamp floor 0.3; never below, never spiraling.
  assert 3.0 <= fr <= 11.5


def test_base_refresh_reads_clean_reset_values_once() -> None:
  env = _env(1)
  bus_voltage_step(env, None, KT)
  env.episode_length_buf[:] = 1
  bus_voltage_step(env, None, KT)
  base1 = None
  from mjlab.tasks.velocity.mdp.bus_voltage import _STATE

  base1 = _STATE[env].base_forcerange.clone()
  # Steps 2..5 with load: base must not move (no re-read of scaled live).
  env.scene["robot"].data.actuator_force = torch.full((1, NU), 8.0)
  for k in range(2, 6):
    env.episode_length_buf[:] = k
    bus_voltage_step(env, None, KT)
  assert torch.equal(_STATE[env].base_forcerange, base1)


def test_discharge_drains_over_episode() -> None:
  env = _env(1)
  bus_voltage_step(env, None, KT)
  v0 = servo_voltage(env).clone()
  for k in range(1, 500):
    env.episode_length_buf[:] = k
    bus_voltage_step(env, None, KT)
  v1 = servo_voltage(env)
  assert (v1 < v0).all()  # ~10 s at 40-90 mV/min: strictly lower
