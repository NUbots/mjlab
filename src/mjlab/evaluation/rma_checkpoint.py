"""Load an add-phase-clock RMA checkpoint into this branch's history actor.

The v-generation runs on the ``add-phase-clock`` branch train an RMA model
that carries two latent producers:

* ``encoder`` -- the privileged teacher, an MLP over the ``dr`` observation
  group (169 dims of domain-randomization truth) producing ``z``;
* ``estimator`` -- the student, a TCN over the 25-frame observation window
  producing ``z_hat``, which is what a robot can actually compute.

This branch has only the student. That is not a gap: those runs set
``RMA_E2E=1``, and the end-to-end forward path is estimator-only --

.. code-block:: python

   if self._e2e:
     z = self.estimator(self.history_normalizer(obs[_HISTORY_GROUP]))
     return torch.cat([x, z], dim=-1)

-- so the teacher is never evaluated, and ``zhat_mix`` (the blend that would
bring it back) is 0.0 in the checkpoint. Every tensor the inference path
touches exists here at exactly the same shape; only the *name* differs,
because a model with one latent producer calls it ``encoder`` while a model
with two calls the student ``estimator``.

:func:`translate_rma_actor` does that rename and drops the teacher. It is
deliberately strict: it refuses a checkpoint whose ``zhat_mix`` is non-zero,
because such a policy's actions depend on privileged inputs this branch
cannot supply, and evaluating it here would silently measure a different
policy.
"""

from __future__ import annotations

from typing import Any

import torch

TEACHER_PREFIXES: tuple[str, ...] = ("encoder.", "dr_normalizer.")
"""Teacher-only tensors, unused whenever ``zhat_mix`` is 0."""

STUDENT_PREFIX = "estimator."
"""Name the two-latent model gives the TCN this branch calls ``encoder``."""


def is_rma_actor(actor_state: dict[str, Any]) -> bool:
  """Whether ``actor_state`` came from the two-latent RMA model."""
  return "zhat_mix" in actor_state


def translate_rma_actor(actor_state: dict[str, Any]) -> dict[str, Any]:
  """Return ``actor_state`` rewritten for this branch's history actor.

  Raises:
    ValueError: if the checkpoint's policy reads the privileged teacher at
      inference (``zhat_mix > 0``), which this branch cannot reproduce.
  """
  mix = actor_state.get("zhat_mix")
  if mix is None:
    return dict(actor_state)
  mix_value = float(mix)
  if mix_value != 0.0:
    raise ValueError(
      f"This checkpoint blends the privileged RMA teacher into the policy "
      f"latent (zhat_mix={mix_value}), so its actions depend on the "
      f"domain-randomization observation group. This branch builds the "
      f"student TCN only and cannot reproduce those actions."
    )

  translated: dict[str, Any] = {}
  for key, value in actor_state.items():
    if key == "zhat_mix" or key.startswith(TEACHER_PREFIXES):
      continue
    if key.startswith(STUDENT_PREFIX):
      key = "encoder." + key[len(STUDENT_PREFIX) :]
    translated[key] = value
  return translated


def translate_checkpoint(checkpoint: Any, destination: Any) -> Any:
  """Write ``checkpoint`` to ``destination`` with its actor translated.

  The critic is carried across untouched. An evaluation loads the actor only,
  so the critic's privileged input layer is never built and never has to
  match; keeping it means the written file still describes the run it came
  from.

  Returns:
    ``destination``, for chaining into a loader.
  """
  loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
  loaded["actor_state_dict"] = translate_rma_actor(loaded["actor_state_dict"])
  torch.save(loaded, destination)
  return destination
