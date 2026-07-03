"""NUgus-specific on-policy runner hooks (optimizer hygiene)."""

from __future__ import annotations

import os
from typing import Callable

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def _env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw in (None, ""):
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw in (None, "") else float(raw)


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  return default if raw in (None, "") else int(raw)


def _optimizer_hygiene_enabled() -> bool:
  return _env_bool("ENTROPY_DECAY", default=False) or os.environ.get("LR_CAP") not in (
    None,
    "",
  )


def _apply_entropy_decay(runner: VelocityOnPolicyRunner, iteration: int) -> None:
  if not _env_bool("ENTROPY_DECAY", default=False):
    return
  start = _env_float("ENTROPY_START", 0.01)
  end = _env_float("ENTROPY_END", 0.001)
  max_iters = runner.cfg["max_iterations"]
  if max_iters <= 1:
    coef = end
  else:
    frac = min(max(iteration / (max_iters - 1), 0.0), 1.0)
    coef = start + (end - start) * frac
  runner.alg.entropy_coef = coef


def _apply_lr_cap(runner: VelocityOnPolicyRunner, iteration: int) -> None:
  if os.environ.get("LR_CAP") in (None, ""):
    return
  start_iter = _env_int("LR_CAP_START_ITER", 2500)
  if iteration < start_iter:
    return
  cap = _env_float("LR_CAP", 3e-4)
  if runner.alg.learning_rate <= cap:
    return
  runner.alg.learning_rate = cap
  for param_group in runner.alg.optimizer.param_groups:
    param_group["lr"] = cap


class NugusOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner with optional NUgus optimizer hygiene hooks."""

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if not _optimizer_hygiene_enabled():
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    original_update: Callable[[], dict[str, float]] = self.alg.update

    def update_with_hygiene() -> dict[str, float]:
      _apply_entropy_decay(self, self.current_learning_iteration)
      loss_dict = original_update()
      _apply_lr_cap(self, self.current_learning_iteration)
      return loss_dict

    self.alg.update = update_with_hygiene  # type: ignore[method-assign]
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.alg.update = original_update  # type: ignore[method-assign]
