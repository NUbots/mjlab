"""Shared-bus voltage model: per-servo supply sag under fleet load.

Measured on NUgus hardware (doc 17 power-network section): supply
voltage is NOT constant - a shared battery/harness term couples all
servos (V_oc ~15.1 V sagging 69 mOhm x I_total, ~22 A / 311 W fleet
peaks costing everyone ~1.5 V of authority at once), a slow discharge
drains ~65 mV/min, and daisy-chain position adds per-servo drops (up to
~0.5 Ohm at chain ends; ankles chronically ~0.5-0.8 V below the arms).
The rigid 14.8 V sim never produces any of this, so the policy neither
experiences the shared authority budget nor can learn to see sag
coming.

Model, applied per control step (event mode="step"):

  I_i   = |actuator_force_i| / Kt_i          (previous-step torque)
  V_i   = V_oc(t) - R_src * sum(I) - R_local_i * I_i
  scale = clamp(V_i / V_nom, 0.3, 1.15)
  actuator_forcerange_i = episode_base_i * scale

R29 lesson applies: the scale multiplies a BASE cached ONCE from the
first-ever call, before this event has written anything - never the
live field. Nothing in the NUgus config rewrites
model.actuator_forcerange at reset (the effort_limits DR event goes
through actuator.set_effort_limit, the very mechanism split R29
exposed), so re-reading the live field "after reset" captures our own
previous scale and compounds it geometrically per episode. That exact
ratchet melted v48's first launch to zero authority within ~450 iters
(simstate ratio 0.0000, every env falling on spawn) - caught by the
R28 telemetry. If per-episode forcerange DR is ever added, the cache
must take that event's output explicitly, not a live re-read.

Per-episode DR (measured ranges, doc 17): V_oc U(12.5, 16.5) V,
discharge U(40, 90) mV/min, R_src U(0.04, 0.15) Ohm, R_local
U(0, 0.35) Ohm per servo. The ``servo_voltage`` observation exposes
V_i (what hardware presentVoltage reports), quantized to the 0.1 V
Dynamixel register LSB by its noise config.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["bus_voltage_step", "servo_voltage", "V_NOMINAL"]

_STATE: WeakKeyDictionary = WeakKeyDictionary()
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

V_NOMINAL = 14.8


class _BusState:
  def __init__(self, env: ManagerBasedRlEnv, kt: dict[str, float]) -> None:
    asset = env.scene["robot"]
    n, device = env.num_envs, env.device
    nu = asset.data.actuator_force.shape[1]
    names = asset.actuator_names if hasattr(asset, "actuator_names") else []
    kt_vec = torch.full((nu,), kt.get("default", 2.68), device=device)
    for i, nm in enumerate(names):
      for pattern, val in kt.items():
        if pattern != "default" and re.search(pattern, nm):
          kt_vec[i] = val
          break
    self.kt = kt_vec
    self.v_oc = torch.full((n,), V_NOMINAL, device=device)
    self.sag_per_s = torch.zeros(n, device=device)
    self.r_src = torch.full((n,), 0.069, device=device)
    self.r_local = torch.zeros((n, nu), device=device)
    self.age_s = torch.zeros(n, device=device)
    self.base_forcerange: torch.Tensor | None = None
    self.voltage = torch.full((n, nu), V_NOMINAL, device=device)
    # Reset detector (episode_length_buf wrap): triggers per-episode DR
    # resampling only. The forcerange base is never re-read after the
    # first call - see the module docstring for the ratchet this avoids.
    self.prev_ep_len = torch.full((n,), torch.iinfo(torch.long).max, device=device)

  def resample(self, env_ids: torch.Tensor, device: str) -> None:
    k = len(env_ids)

    def u(a: float, b: float, shape: tuple) -> torch.Tensor:
      return sample_uniform(
        torch.tensor(a, device=device), torch.tensor(b, device=device), shape, device
      )

    self.v_oc[env_ids] = u(12.5, 16.5, (k,))
    self.sag_per_s[env_ids] = u(0.040, 0.090, (k,)) / 60.0
    self.r_src[env_ids] = u(0.04, 0.15, (k,))
    self.r_local[env_ids] = u(0.0, 0.35, (k, self.r_local.shape[1]))
    self.age_s[env_ids] = 0.0
    self.voltage[env_ids] = self.v_oc[env_ids].unsqueeze(-1)


def _get_state(env: ManagerBasedRlEnv, kt: dict[str, float]) -> _BusState:
  state = _STATE.get(env)
  if state is None:
    state = _BusState(env, kt)
    _STATE[env] = state
  return state


def bus_voltage_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kt: dict[str, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Step event: update per-servo voltage and scale torque authority."""
  del env_ids
  state = _get_state(env, kt)
  asset = env.scene[asset_cfg.name]
  device = env.device

  # Base cached ONCE, from the clean pre-write field on the first call.
  # Never refreshed from the live field afterwards: nothing restores
  # actuator_forcerange at reset in this config, so the live rows of a
  # freshly reset env still hold OUR previous scale - re-adopting them
  # compounds ~0.85x per episode into zero authority (the R29 ratchet;
  # it killed v48's first launch). The scale below always multiplies
  # this immutable base.
  ctrl_ids = asset.indexing.ctrl_ids
  if state.base_forcerange is None:
    state.base_forcerange = env.sim.model.actuator_forcerange[:, ctrl_ids].clone()
  fresh = env.episode_length_buf < state.prev_ep_len
  state.prev_ep_len = env.episode_length_buf.clone()
  if fresh.any():
    ids = torch.arange(env.num_envs, device=device)[fresh]
    state.resample(ids, device)

  step_dt = float(env.step_dt)
  state.age_s += step_dt
  current = asset.data.actuator_force.abs() / state.kt
  i_total = current.sum(dim=1, keepdim=True)
  v_oc = (state.v_oc - state.sag_per_s * state.age_s).unsqueeze(-1)
  volts = v_oc - state.r_src.unsqueeze(-1) * i_total - state.r_local * current
  state.voltage = volts
  scale = (volts / V_NOMINAL).clamp(0.3, 1.15)
  env.sim.model.actuator_forcerange[:, ctrl_ids] = (
    state.base_forcerange * scale.unsqueeze(-1)
  )


def servo_voltage(
  env: ManagerBasedRlEnv,
  kt: dict[str, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-servo supply voltage observation (hardware: presentVoltage)."""
  del asset_cfg
  state = _STATE.get(env)
  if state is None:
    state = _get_state(env, kt or {"default": 2.68})
  return state.voltage
