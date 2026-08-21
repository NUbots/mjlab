"""NUbots' distilled walk policy, run batched inside mjlab.

The policy is a supervised copy of the quintic walk engine: ``WalkDataCollector``
logged the engine's IK output over ten thousand episodes of random velocity
commands, and ``tools/walk_distillation`` fit a small MLP to reproduce it from
the command, the engine's phase clock and its own three previous outputs.
NUbots deploys the result as ``module/skill/NeuralWalk``.

Reproduced here so that the copy can be measured against the engine it copies,
on the same robot models and by the same metrics as a reinforcement-learned
policy. See ``scripts/eval/eval_distilled_quintic_walk.py``.
"""

from mjlab.controllers.distilled_walk.controller import (
  HISTORY_FRAMES,
  DistilledWalkController,
  HistoryInit,
)
from mjlab.controllers.distilled_walk.policy import (
  DEFAULT_POLICY_PATH,
  OBS_DIM,
  TARGET_DIM,
  DistilledWalkPolicy,
)

__all__ = (
  "DEFAULT_POLICY_PATH",
  "DistilledWalkController",
  "DistilledWalkPolicy",
  "HISTORY_FRAMES",
  "HistoryInit",
  "OBS_DIM",
  "TARGET_DIM",
)
