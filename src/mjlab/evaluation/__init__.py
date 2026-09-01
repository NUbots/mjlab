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
from mjlab.evaluation.push import (
  PerEnvPushMetrics,
  PushCfg,
  PushDriver,
  PushMetrics,
  PushPlan,
  concat_push_metrics,
  format_push_summary,
  push_battery,
  push_envelope,
  run_push_battery,
  summarise_push,
)

__all__ = (
  "EvalState",
  "FALL_UPRIGHT_THRESHOLD",
  "Lane",
  "PerEnvMetrics",
  "PerEnvPushMetrics",
  "Profile",
  "ProfileCfg",
  "PushCfg",
  "PushDriver",
  "PushMetrics",
  "PushPlan",
  "Segment",
  "VelocityTrace",
  "WalkMetrics",
  "concat_push_metrics",
  "format_push_summary",
  "format_summary",
  "omnidirectional_profile",
  "push_battery",
  "push_envelope",
  "run_push_battery",
  "save_run",
  "summarise",
  "summarise_push",
  "write_trace_csv",
)
