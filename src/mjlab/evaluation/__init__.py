"""Standalone evaluation pipeline for comparing walk controllers.

Two entry points in ``scripts/eval`` drive the two engines; everything they
share -- the plant, the batched harnesses, the metrics and the output format --
lives here. See ``scripts/eval/README.md``.
"""

from mjlab.evaluation.metrics import (
  FALL_UPRIGHT_THRESHOLD,
  EvalState,
  PerEnvMetrics,
  VelocityTrace,
  WalkMetrics,
  format_summary,
  save_run,
  summarise,
  write_trace_csv,
)
from mjlab.evaluation.profile import (
  Lane,
  Profile,
  ProfileCfg,
  Segment,
  omnidirectional_profile,
)

__all__ = (
  "EvalState",
  "FALL_UPRIGHT_THRESHOLD",
  "Lane",
  "PerEnvMetrics",
  "Profile",
  "ProfileCfg",
  "Segment",
  "VelocityTrace",
  "WalkMetrics",
  "format_summary",
  "omnidirectional_profile",
  "save_run",
  "summarise",
  "write_trace_csv",
)
