"""Orchestrator for the RL parameter tuning system.

Plain Python loop — no LLM reasoning at this level. The orchestrator owns
state, sequencing, and process management. It calls three agents (Planner,
Editor, Analyst) which DO use the Claude Code SDK.

State layout (all on disk under WORKSPACE):
    manifest.json                    : parameter manifest
    state.json                       : current best config, run status
    experiment_history.jsonl         : append-only log of all experiments
    interpretation_rules.md          : rules read by Planner & Analyst
    scoring_function.py              : deterministic score
    metrics_compression.py           : produces summary from full log
    runs/<timestamp>/
        proposal.json                : Planner output for this run
        diff.txt                     : Editor's record of file changes
        training_log.json            : copy of the parser output
        compressed.json              : output of metrics_compression
        score.json                   : output of scoring_function
        analysis.json                : Analyst output
        feedback.json                : human feedback (if requested)

Subprocess management:
    Training is launched via the configured TRAIN_CMD. The orchestrator
    polls the run's training_log.json and sends SIGTERM when the target
    iteration count is reached.

Each agent call is a fresh one-shot subprocess via the Claude Code SDK —
no context carries between calls. All shared state is on disk.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import claude_agent_sdk as sdk

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent
RUNS_DIR = WORKSPACE / "runs"
FEEDBACK_DIR = WORKSPACE / "feedback_queue"
HISTORY_FILE = WORKSPACE / "experiment_history.jsonl"
STATE_FILE = WORKSPACE / "state.json"
MANIFEST_FILE = WORKSPACE / "parameter_manifest.json"

# Training launch — fill these in for your environment.
TRAIN_CMD = [
  "uv",
  "run",
  "train",
  "Mjlab-Velocity-Flat-Nubots-Nugus",
  "--env.scene.num-envs",
  "4096",
  "--log_terminal",
  "True",
]
# Where the training process writes its output dir. The orchestrator looks for
# the most recent timestamped folder here after launch to find this run's log.
TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "nugus_velocity"

# Probe length per experiment.
PROBE_ITERATIONS = 3000

# Adaptive extension. When a probe finishes, the orchestrator inspects the
# current run's score + trends and either extends by another PROBE_ITERATIONS
# or terminates. MAX_EXTENSIONS bounds the worst case at
# PROBE_ITERATIONS * (1 + MAX_EXTENSIONS) iterations.
MAX_EXTENSIONS = 4

# Polling intervals.
LOG_POLL_SECONDS = 10.0
FEEDBACK_POLL_SECONDS = 30.0

# When the Claude session usage limit is hit and we cannot parse a reset
# time from the error message, sleep this long before retrying the agent
# call. The orchestrator never crashes on a usage limit — it just waits.
USAGE_LIMIT_FALLBACK_SLEEP_SECONDS = 1800  # 30 min
USAGE_LIMIT_HEARTBEAT_SECONDS = 300  # log progress every 5 min while waiting

# Human feedback cadence.
EXPERIMENTS_PER_FEEDBACK = 5

# Early-failure gate: if the most recent fell_over rate exceeds this past
# EARLY_KILL_AFTER_ITER iterations, kill the run regardless of promise. This
# is the "obviously broken" tripwire — adaptive extension handles the
# subtler "is this worth more compute" question.
EARLY_KILL_FELL_OVER = 1.5
EARLY_KILL_AFTER_ITER = 750

# Stability gate threshold (matches scoring_function.STABILITY_GATE_FELL_OVER).
# Used by the promise check to estimate when a run might pass the gate.
STABILITY_GATE_FELL_OVER = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class State:
  best_score: float
  best_run_id: str | None
  experiments_since_feedback: int
  total_experiments: int

  @classmethod
  def load(cls) -> "State":
    if not STATE_FILE.exists():
      return cls(
        best_score=-1.0,
        best_run_id=None,
        experiments_since_feedback=0,
        total_experiments=0,
      )
    with open(STATE_FILE) as f:
      return cls(**json.load(f))

  def save(self) -> None:
    with open(STATE_FILE, "w") as f:
      json.dump(self.__dict__, f, indent=2)


def append_history(record: dict) -> None:
  with open(HISTORY_FILE, "a") as f:
    f.write(json.dumps(record) + "\n")


def load_history() -> list[dict]:
  if not HISTORY_FILE.exists():
    return []
  with open(HISTORY_FILE) as f:
    return [json.loads(line) for line in f if line.strip()]


def new_run_dir() -> Path:
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir = RUNS_DIR / timestamp
  run_dir.mkdir(parents=True, exist_ok=True)
  return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# SDK helper
# ─────────────────────────────────────────────────────────────────────────────


class UsageLimitError(RuntimeError):
  """Raised when the Claude session usage limit has been hit.

  ``reset_at`` is the parsed reset time (timezone-aware) when one could be
  extracted from the error message, otherwise None.
  """

  def __init__(self, message: str, reset_at: datetime | None) -> None:
    super().__init__(message)
    self.reset_at = reset_at


# Matches messages like:
#   "You've hit your limit · resets 1:40pm (Australia/Sydney)"
#   "...resets 13:40 (UTC)"
_USAGE_LIMIT_RESET_RE = re.compile(
  r"resets\s+(\d{1,2}):(\d{2})\s*(am|pm)?\s*\(([^)]+)\)",
  re.IGNORECASE,
)


def _parse_usage_limit_reset(message: str) -> datetime | None:
  m = _USAGE_LIMIT_RESET_RE.search(message)
  if not m:
    return None
  hour_s, minute_s, ampm, tz_name = m.groups()
  hour = int(hour_s)
  minute = int(minute_s)
  if ampm:
    ampm_l = ampm.lower()
    if ampm_l == "pm" and hour != 12:
      hour += 12
    elif ampm_l == "am" and hour == 12:
      hour = 0
  try:
    tz = ZoneInfo(tz_name.strip())
  except ZoneInfoNotFoundError:
    return None
  now = datetime.now(tz)
  candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
  # If the parsed wall-clock time has already passed today, the reset is
  # tomorrow — these limits roll over within ~24h.
  if candidate <= now:
    candidate += timedelta(days=1)
  return candidate


def _is_usage_limit_message(text: str) -> bool:
  t = text.lower()
  return "hit your limit" in t or "usage limit" in t


async def _query_agent(
  prompt: str,
  system_prompt: str,
  output_format: dict[str, Any] | None = None,
  max_turns: int = 5,
  permission_mode: sdk.PermissionMode = "default",
  tools: list[str] | None = None,
) -> Any:
  """Run a one-shot Claude agent query and return the result."""
  # The SDK expects output_format wrapped as {"type": "json_schema", "schema": ...}
  # so the CLI receives the --json-schema flag.
  sdk_output_format = None
  if output_format is not None:
    sdk_output_format = {"type": "json_schema", "schema": output_format}
  options = sdk.ClaudeAgentOptions(
    system_prompt=system_prompt,
    output_format=sdk_output_format,
    max_turns=max_turns,
    permission_mode=permission_mode,
    cwd=str(REPO_ROOT),
    model="claude-sonnet-4-6",
    tools=tools if tools is not None else None,
  )
  async for message in sdk.query(prompt=prompt, options=options):
    if isinstance(message, sdk.ResultMessage):
      if message.is_error:
        details = message.result or (
          "; ".join(message.errors) if message.errors else "unknown"
        )
        if _is_usage_limit_message(details):
          raise UsageLimitError(details, _parse_usage_limit_reset(details))
        raise RuntimeError(f"Agent error: {details}")
      if output_format:
        if message.structured_output is not None:
          return message.structured_output
        # Fallback: parse result as JSON if structured_output is missing.
        if message.result:
          return json.loads(message.result)
      return message.result or ""
  raise RuntimeError("Agent returned no result")


def _get_event_loop() -> asyncio.AbstractEventLoop:
  """Get or create a persistent event loop for agent queries."""
  global _event_loop
  if _event_loop is None or _event_loop.is_closed():
    _event_loop = asyncio.new_event_loop()
  return _event_loop


_event_loop: asyncio.AbstractEventLoop | None = None


def _sleep_with_heartbeat(seconds: float, what: str) -> None:
  """Sleep, logging remaining time periodically so the user sees progress."""
  end = time.monotonic() + seconds
  while True:
    remaining = end - time.monotonic()
    if remaining <= 0:
      return
    chunk = min(remaining, USAGE_LIMIT_HEARTBEAT_SECONDS)
    time.sleep(chunk)
    remaining_after = end - time.monotonic()
    if remaining_after > 0:
      mins = int(remaining_after // 60)
      log.info(f"  still waiting for {what}... ~{mins} min remaining")


def _run_agent(prompt: str, system_prompt: str, **kwargs: Any) -> Any:
  """Sync wrapper around the async SDK query.

  If the Claude session usage limit is hit, this sleeps until the reset
  time (parsed from the error) and retries — it never raises out. This
  lets the orchestrator survive limit windows without losing in-flight
  training progress: training has already been recorded to disk by the
  time the planner/editor (next experiment) or analyst (this experiment)
  is called.
  """
  loop = _get_event_loop()
  while True:
    try:
      return loop.run_until_complete(_query_agent(prompt, system_prompt, **kwargs))
    except UsageLimitError as e:
      if e.reset_at is not None:
        # Add a small buffer so we don't retry right at the boundary.
        wait_s = (e.reset_at - datetime.now(e.reset_at.tzinfo)).total_seconds() + 30
        wait_s = max(60.0, wait_s)
        log.warning(
          f"Usage limit hit; resets at {e.reset_at.isoformat()}. "
          f"Sleeping ~{int(wait_s // 60)} min then retrying."
        )
      else:
        wait_s = float(USAGE_LIMIT_FALLBACK_SLEEP_SECONDS)
        log.warning(
          f"Usage limit hit; reset time not parseable ({e}). "
          f"Sleeping {int(wait_s // 60)} min then retrying."
        )
      _sleep_with_heartbeat(wait_s, "usage limit reset")
      log.info("Retrying agent call after usage limit wait")


# ─────────────────────────────────────────────────────────────────────────────
# File snapshots for revert
# ─────────────────────────────────────────────────────────────────────────────


def snapshot_files(run_dir: Path, proposal: dict) -> None:
  """Copy source files that will be edited into run_dir/snapshots/."""
  manifest = json.loads(MANIFEST_FILE.read_text())
  snap_dir = run_dir / "snapshots"
  snap_dir.mkdir(exist_ok=True)
  snapped: set[str] = set()
  for param_name in proposal["changes"]:
    entry = manifest.get(param_name)
    if entry is None:
      continue
    rel_path = entry["file"]
    if rel_path in snapped:
      continue
    src = REPO_ROOT / rel_path
    if src.exists():
      # Encode path separators so snapshots are flat files.
      dest = snap_dir / rel_path.replace("/", "__")
      shutil.copy2(src, dest)
      snapped.add(rel_path)
  log.info(f"  Snapshotted {len(snapped)} file(s)")


def revert_changes(run_dir: Path) -> None:
  """Restore source files from run_dir/snapshots/."""
  snap_dir = run_dir / "snapshots"
  if not snap_dir.exists():
    log.warning("  No snapshots to revert from")
    return
  for snap_file in snap_dir.iterdir():
    rel_path = snap_file.name.replace("__", "/")
    dest = REPO_ROOT / rel_path
    shutil.copy2(snap_file, dest)
    log.info(f"  Reverted {rel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent implementations
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a parameter tuning planner for a humanoid robot (NUbots Nugus) \
locomotion RL policy. Your job: propose the next experiment — a small set \
of parameter changes (at most 2-3) to improve gait quality while \
maintaining stability.

## Domain Rules (MUST follow)
{rules}

## Edit Rules
- Change AT MOST 2-3 parameters per experiment for interpretability.
- Stay within the 'range' specified in the manifest for every parameter.
- If two parameters are semantically coupled (e.g. both foot height \
targets), count them as one change and adjust both.
- Use the parameter names from the manifest as keys in your "changes" dict.
"""

PLANNER_PROMPT = """\
## Current State
{state}

## Parameter Manifest
{manifest}

## Experiment History (most recent last)
{history}

{feedback_section}

Propose the next experiment. Consider:
1. What has been tried and what worked / didn't work.
2. Which parameters are most likely to improve gait quality now.
3. Interactions between parameters (see rules).

Use {probe_iterations} for probe_iterations unless you have a specific \
reason to change it.
"""

PROPOSAL_SCHEMA: dict[str, Any] = {
  "type": "object",
  "properties": {
    "hypothesis": {
      "type": "string",
      "description": "What you expect this change to achieve",
    },
    "changes": {
      "type": "object",
      "description": "Map of parameter_manifest key -> new numeric value",
      "additionalProperties": {"type": "number"},
    },
    "expected_effects": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Predicted metric changes to check",
    },
    "probe_iterations": {"type": "integer"},
  },
  "required": ["hypothesis", "changes", "expected_effects", "probe_iterations"],
}


def call_planner(run_dir: Path) -> dict:
  """Produce next experiment proposal via the Planner agent."""
  manifest_text = MANIFEST_FILE.read_text()
  rules = (WORKSPACE / "interpretation_rules.md").read_text()
  history_text = HISTORY_FILE.read_text() if HISTORY_FILE.exists() else "(none)"
  state_text = STATE_FILE.read_text() if STATE_FILE.exists() else "{}"

  # Find most recent human feedback across all runs.
  feedback_section = ""
  if RUNS_DIR.exists():
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
      fb = d / "feedback.json"
      if fb.exists():
        feedback_section = f"## Latest Human Feedback\n{fb.read_text()}"
        break

  result = _run_agent(
    prompt=PLANNER_PROMPT.format(
      state=state_text,
      manifest=manifest_text,
      history=history_text,
      feedback_section=feedback_section,
      probe_iterations=PROBE_ITERATIONS,
    ),
    system_prompt=PLANNER_SYSTEM.format(rules=rules),
    output_format=PROPOSAL_SCHEMA,
    max_turns=3,
  )

  if not isinstance(result, dict):
    raise TypeError(f"Planner returned {type(result).__name__}, expected dict")
  if len(result.get("changes", {})) > 3:
    raise ValueError(f"Planner proposed {len(result['changes'])} changes (max 3)")

  with open(run_dir / "proposal.json", "w") as f:
    json.dump(result, f, indent=2)
  return result


# ── Editor ──────────────────────────────────────────────────────────────────

EDITOR_SYSTEM = """\
You are a precise code editor for RL parameter tuning. You will be given \
a set of parameter changes with the current source file contents. For each \
change, output the minimal old_string / new_string replacement needed.

## How to locate values
- key_path like `rewards['name'].weight` → find `weight=<value>` inside \
the RewardTermCfg block for that reward.
- key_path like `cfg.rewards['name'].weight` → find the assignment \
`cfg.rewards["name"].weight = <value>`.
- key_path like `cfg.rewards['name'].params['key']` → find the assignment \
`cfg.rewards["name"].params["key"] = <value>`.
- key_path with regex like `[r'.*pattern.*']` → find that regex as a dict \
key and replace its value.

## Rules
- Replace ONLY the numeric value. Preserve all surrounding formatting.
- old_string must be an exact substring of the file content.
- Each old_string must be unique within its file (include enough context).
"""

EDITOR_PROMPT = """\
Apply these changes. The source files are shown below.

## Changes
{changes_json}

## Parameter info (file paths and key_paths)
{param_info_json}

## Source files
{file_contents}

Return the edits needed.
"""

EDITOR_SCHEMA: dict[str, Any] = {
  "type": "object",
  "properties": {
    "edits": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "parameter_name": {"type": "string"},
          "file": {"type": "string"},
          "old_string": {"type": "string"},
          "new_string": {"type": "string"},
        },
        "required": ["parameter_name", "file", "old_string", "new_string"],
      },
    },
  },
  "required": ["edits"],
}


def call_editor(run_dir: Path) -> None:
  """Apply proposal changes via the Editor agent."""
  proposal = json.loads((run_dir / "proposal.json").read_text())
  manifest = json.loads(MANIFEST_FILE.read_text())

  # Collect manifest entries and source file contents for changed params.
  param_info: dict[str, Any] = {}
  files_to_read: dict[str, str] = {}  # rel_path -> content
  for param_name in proposal["changes"]:
    entry = manifest.get(param_name)
    if entry is None:
      continue
    param_info[param_name] = {
      "file": entry["file"],
      "key_path": entry["key_path"],
      "new_value": proposal["changes"][param_name],
    }
    rel = entry["file"]
    if rel not in files_to_read:
      files_to_read[rel] = (REPO_ROOT / rel).read_text()

  file_contents_str = ""
  for rel, content in files_to_read.items():
    file_contents_str += f"\n### {rel}\n```python\n{content}\n```\n"

  result = _run_agent(
    prompt=EDITOR_PROMPT.format(
      changes_json=json.dumps(proposal["changes"], indent=2),
      param_info_json=json.dumps(param_info, indent=2),
      file_contents=file_contents_str,
    ),
    system_prompt=EDITOR_SYSTEM,
    output_format=EDITOR_SCHEMA,
    max_turns=8,
    tools=[],
  )

  if not isinstance(result, dict):
    raise TypeError(f"Editor returned {type(result).__name__}, expected dict")

  # Apply edits deterministically and record the diff.
  diff_lines: list[str] = []
  for edit in result["edits"]:
    rel = edit["file"]
    path = REPO_ROOT / rel
    content = path.read_text()
    old = edit["old_string"]
    new = edit["new_string"]
    if old not in content:
      raise ValueError(
        f"old_string not found in {rel} for {edit['parameter_name']}:\n  {old!r}"
      )
    if content.count(old) > 1:
      raise ValueError(
        f"old_string is ambiguous in {rel} for {edit['parameter_name']} "
        f"({content.count(old)} occurrences)"
      )
    content = content.replace(old, new, 1)
    path.write_text(content)
    diff_lines.append(
      f"parameter: {edit['parameter_name']}\n"
      f"file: {rel}\n"
      f"before: {old.strip()}\n"
      f"after:  {new.strip()}\n"
    )
    log.info(f"  Edited {edit['parameter_name']} in {rel}")

  (run_dir / "diff.txt").write_text("\n".join(diff_lines))


# ── Analyst ─────────────────────────────────────────────────────────────────

ANALYST_SYSTEM = """\
You are an analyst for humanoid robot locomotion RL experiments. Your job: \
evaluate a completed experiment and provide actionable insights for the \
next one.

## Domain Rules (MUST follow)
{rules}
"""

ANALYST_PROMPT = """\
Analyse this completed experiment.

## Proposal (what was tried)
{proposal}

## Compressed Metrics Summary
{compressed}

## Score
{score}
{feedback_section}

Provide your analysis:
1. Did the hypothesis hold? Why or why not?
2. What are the key observations from the metrics?
3. What should the next experiment focus on?

Consider reward competition effects, curriculum stage, and the domain rules.
"""

ANALYSIS_SCHEMA: dict[str, Any] = {
  "type": "object",
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["promising", "neutral", "regression"],
    },
    "key_observations": {
      "type": "array",
      "items": {"type": "string"},
    },
    "next_suggestion": {"type": "string"},
  },
  "required": ["verdict", "key_observations", "next_suggestion"],
}


def call_analyst(run_dir: Path) -> dict:
  """Analyse a completed run via the Analyst agent."""
  compressed = (run_dir / "compressed.json").read_text()
  score_text = (run_dir / "score.json").read_text()
  proposal = (run_dir / "proposal.json").read_text()
  rules = (WORKSPACE / "interpretation_rules.md").read_text()

  feedback_section = ""
  fb_path = run_dir / "feedback.json"
  if fb_path.exists():
    feedback_section = f"\n## Human Feedback\n{fb_path.read_text()}"

  result = _run_agent(
    prompt=ANALYST_PROMPT.format(
      proposal=proposal,
      compressed=compressed,
      score=score_text,
      feedback_section=feedback_section,
    ),
    system_prompt=ANALYST_SYSTEM.format(rules=rules),
    output_format=ANALYSIS_SCHEMA,
    max_turns=3,
  )

  if not isinstance(result, dict):
    raise TypeError(f"Analyst returned {type(result).__name__}, expected dict")

  with open(run_dir / "analysis.json", "w") as f:
    json.dump(result, f, indent=2)
  return result


# ─────────────────────────────────────────────────────────────────────────────
# Training subprocess management
# ─────────────────────────────────────────────────────────────────────────────


def find_latest_log_dir(after_time: float) -> Path | None:
  """Find the most recent training output dir created after `after_time`."""
  if not TRAINING_LOG_ROOT.exists():
    return None
  candidates = [
    d
    for d in TRAINING_LOG_ROOT.iterdir()
    if d.is_dir() and d.stat().st_mtime > after_time
  ]
  if not candidates:
    return None
  return max(candidates, key=lambda d: d.stat().st_mtime)


@dataclass
class ExtensionDecision:
  """Result of asking 'should this run keep going for another probe?'."""

  extend: bool
  reason: str


def assess_run_promise(
  run_dir: Path,
  log_path: Path,
  current_iter: int,
  extensions_so_far: int,
) -> ExtensionDecision:
  """Decide whether the in-flight run is promising enough to extend.

  Pure Python — no LLM call. Reads a snapshot of the partial log, runs the
  same compression + scoring used for the final eval, and compares the
  result to the experiment history. The decision and the partial scoring
  snapshot are written to ``run_dir/checkpoints/`` for later debugging.
  """
  compress_log = _import_local("metrics_compression").compress_log
  score_run = _import_local("scoring_function").score_run

  # Snapshot the log first — the training process may still be writing.
  ckpt_dir = run_dir / "checkpoints"
  ckpt_dir.mkdir(exist_ok=True)
  snapshot_path = ckpt_dir / f"training_log_iter_{current_iter}.json"
  shutil.copy(log_path, snapshot_path)

  try:
    summary = compress_log(snapshot_path)
    score = score_run(summary)
  except Exception as e:
    return ExtensionDecision(False, f"could not evaluate partial log: {e}")

  with open(ckpt_dir / f"score_iter_{current_iter}.json", "w") as f:
    json.dump(
      {
        "iteration": current_iter,
        "extensions_so_far": extensions_so_far,
        "score": score,
        "health_flags": summary.get("health_flags", []),
      },
      f,
      indent=2,
    )

  if extensions_so_far >= MAX_EXTENSIONS:
    return ExtensionDecision(False, f"reached MAX_EXTENSIONS={MAX_EXTENSIONS}")

  current_score = float(score["score"])
  fell_over_rate = float(summary["final_window"].get("fell_over_rate", 0.0))
  fell_over_slope = float(score.get("fell_over_slope", 0.0))
  gates_failed = score.get("gates_failed", []) or []

  if not gates_failed:
    # Already passing hard gates. Extend unless we've slipped well below
    # the best run we have seen — that's diminishing returns.
    history = load_history()
    past_scores = [
      float(h["score"].get("score", 0.0))
      for h in history
      if isinstance(h.get("score"), dict)
    ]
    best = max(past_scores, default=0.0)
    if best > 0.0 and current_score < 0.7 * best:
      return ExtensionDecision(
        False,
        f"passing but score {current_score:.3f} below 70% of best {best:.3f}",
      )
    return ExtensionDecision(
      True, f"passing gates with score {current_score:.3f} — extend"
    )

  # Gate-failing run: only extend if recovery is plausible. Project the
  # slope forward — how many iterations until fell_over crosses the gate?
  if fell_over_slope >= 0:
    return ExtensionDecision(
      False,
      f"failing ({'/'.join(gates_failed)}, fell_over={fell_over_rate:.2f}) "
      f"with non-negative slope {fell_over_slope:.5f}",
    )
  iters_to_pass = (fell_over_rate - STABILITY_GATE_FELL_OVER) / -fell_over_slope
  if iters_to_pass > 2 * PROBE_ITERATIONS:
    return ExtensionDecision(
      False,
      f"failing ({'/'.join(gates_failed)}); recovery would need ~"
      f"{int(iters_to_pass)} iters at slope {fell_over_slope:.5f}",
    )
  return ExtensionDecision(
    True,
    f"failing ({'/'.join(gates_failed)}) but recovering — "
    f"~{int(iters_to_pass)} iters from gate at slope {fell_over_slope:.5f}",
  )


def _wait_for_training_log(
  proc: subprocess.Popen, launch_time: float
) -> tuple[Path, Path]:
  """Wait for the training subprocess to produce its log file."""
  for _ in range(60):  # up to 10 minutes
    time.sleep(LOG_POLL_SECONDS)
    log_dir = find_latest_log_dir(launch_time)
    if log_dir and (log_dir / "training_log.json").exists():
      log.info(f"  found log: {log_dir}")
      return log_dir, log_dir / "training_log.json"
  proc.send_signal(signal.SIGTERM)
  raise RuntimeError("Training did not produce a log within 10 minutes")


def launch_training(run_dir: Path, initial_target: int) -> int:
  """Launch training and adaptively extend until the run stops being promising.

  Polls the training subprocess's log. When the iteration count reaches the
  current target, evaluates partial results and either extends by another
  ``PROBE_ITERATIONS`` (up to ``MAX_EXTENSIONS`` times) or terminates. The
  early-kill tripwire still applies for obvious stability disasters.

  Returns the final iteration count reached.
  """
  log.info(f"Launching training (initial target: {initial_target} iters)")
  launch_time = time.time()
  proc = subprocess.Popen(
    TRAIN_CMD,
    cwd=str(REPO_ROOT),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  log.info(f"  pid: {proc.pid}")

  log_dir, log_path = _wait_for_training_log(proc, launch_time)

  # Persist the training log dir so the human-feedback step (and anyone
  # later running ``play``) can find the policy this run produced.
  with open(run_dir / "run_meta.json", "w") as f:
    json.dump(
      {
        "training_log_dir": str(log_dir),
        "started_at": datetime.fromtimestamp(launch_time).isoformat(),
        "initial_target_iterations": initial_target,
      },
      f,
      indent=2,
    )

  target = initial_target
  extensions = 0
  current = 0

  while True:
    time.sleep(LOG_POLL_SECONDS)
    if proc.poll() is not None:
      log.warning("Training subprocess exited unexpectedly")
      break
    try:
      with open(log_path) as f:
        training_log = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
      continue  # log is being written, retry next tick

    iters = training_log.get("iteration", [])
    if not iters:
      continue
    current = iters[-1]

    # Early-kill: obvious stability disaster, regardless of promise check.
    if current >= EARLY_KILL_AFTER_ITER:
      recent_falls = training_log.get("Episode_Termination/fell_over", [])
      if recent_falls and recent_falls[-1] > EARLY_KILL_FELL_OVER:
        log.warning(f"Early kill: fell_over={recent_falls[-1]:.2f} at iter {current}")
        break

    if current < target:
      continue

    decision = assess_run_promise(
      run_dir=run_dir,
      log_path=log_path,
      current_iter=current,
      extensions_so_far=extensions,
    )
    log.info(
      f"  Iter {current} promise check: extend={decision.extend} — {decision.reason}"
    )
    if not decision.extend:
      break
    extensions += 1
    target = current + PROBE_ITERATIONS
    log.info(
      f"  Extending to {target} iters (extension #{extensions}/{MAX_EXTENSIONS})"
    )

  proc.send_signal(signal.SIGTERM)
  try:
    proc.wait(timeout=30)
  except subprocess.TimeoutExpired:
    log.warning("SIGTERM ignored, sending SIGKILL")
    proc.kill()
    proc.wait()

  shutil.copy(log_path, run_dir / "training_log.json")

  # Update run_meta with the actual final iteration count and extension tally.
  meta_path = run_dir / "run_meta.json"
  meta = json.loads(meta_path.read_text())
  meta.update({"final_iterations": current, "extensions_used": extensions})
  with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

  return current


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + compression (pure Python, no agents)
# ─────────────────────────────────────────────────────────────────────────────


def _import_local(name: str) -> Any:
  """Import a module from WORKSPACE by file path."""
  spec = importlib.util.spec_from_file_location(name, WORKSPACE / f"{name}.py")
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)  # type: ignore[union-attr]
  return mod


def evaluate_run(run_dir: Path) -> dict:
  """Compress + score the training log. Returns the score record."""
  compress_log = _import_local("metrics_compression").compress_log
  score_run = _import_local("scoring_function").score_run

  summary = compress_log(run_dir / "training_log.json")
  with open(run_dir / "compressed.json", "w") as f:
    json.dump(summary, f, indent=2)

  # Pass the full summary so the score can reflect n_iterations and the
  # fell_over slope, not just final-window means.
  score = score_run(summary)
  with open(run_dir / "score.json", "w") as f:
    json.dump(score, f, indent=2)

  gates_failed = score.get("gates_failed") or []
  gate_note = f"  (gates failed: {', '.join(gates_failed)})" if gates_failed else ""
  log.info(f"Score: {score['score']:.3f}{gate_note}")
  return score


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback handoff
# ─────────────────────────────────────────────────────────────────────────────


def request_human_feedback(state: State) -> None:
  """Pause until the human writes feedback on the current best run.

  The previous version of this function recorded both ``run_dir`` (the
  latest experiment) and ``best_run`` in pending.json, which left the
  reviewer unsure which policy to play. Latest-run feedback was also
  ambiguous because losing experiments are reverted — the source code on
  disk no longer reflects the policy under review.

  Now we always solicit feedback on the *current best* run, since that is
  the policy embedded in the source files and the baseline future
  experiments will build on. Pending.json carries a single
  ``run_to_evaluate`` field plus the path to the trained-policy checkpoints
  so the human knows exactly which run to play.
  """
  pending_path = FEEDBACK_DIR / "pending.json"

  if state.best_run_id is None:
    log.info("No best run yet — skipping feedback request")
    return

  best_run_dir = RUNS_DIR / state.best_run_id
  meta_path = best_run_dir / "run_meta.json"
  policy_log_dir: str | None = None
  if meta_path.exists():
    try:
      meta = json.loads(meta_path.read_text())
      policy_log_dir = meta.get("training_log_dir")
    except json.JSONDecodeError:
      log.warning(f"Could not parse {meta_path}; play path will be missing")

  score_path = best_run_dir / "score.json"
  score_summary = json.loads(score_path.read_text()) if score_path.exists() else None

  play_hint = (
    f"`uv run play <task> --load_run {policy_log_dir}`"
    if policy_log_dir
    else "(policy log dir not recorded for this run; check logs/rsl_rl/...)"
  )

  request = {
    "run_to_evaluate": state.best_run_id,
    "run_dir": str(best_run_dir),
    "policy_log_dir": policy_log_dir,
    "current_best_score": state.best_score,
    "score": score_summary,
    "questions": [
      "leg_coordination_1to5",
      "foot_contact_quality_1to5",
      "naturalness_1to5",
      "specific_issues_freetext",
    ],
    "instructions": (
      f"Watch the policy from run '{state.best_run_id}' (the system's "
      f"current best). Play it with: {play_hint}. "
      "Then fill in the answer fields below and save. The orchestrator "
      "will detect the change."
    ),
    "answers": {
      "leg_coordination_1to5": None,
      "foot_contact_quality_1to5": None,
      "naturalness_1to5": None,
      "specific_issues_freetext": None,
    },
  }
  with open(pending_path, "w") as f:
    json.dump(request, f, indent=2)

  log.info(
    f"⏸  Awaiting human feedback on best run '{state.best_run_id}' at {pending_path}"
  )
  while True:
    time.sleep(FEEDBACK_POLL_SECONDS)
    with open(pending_path) as f:
      current = json.load(f)
    if all(v is not None for v in current["answers"].values()):
      break

  # Save feedback against the run it actually describes, not the latest run.
  feedback_path = best_run_dir / "feedback.json"
  shutil.copy(pending_path, feedback_path)
  pending_path.unlink()
  log.info("▶  Feedback received, resuming")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────


def run_one_experiment(state: State) -> None:
  run_dir = new_run_dir()
  log.info(f"=== Experiment #{state.total_experiments + 1} | {run_dir.name} ===")

  # 1. Plan
  proposal = call_planner(run_dir)
  log.info(f"Hypothesis: {proposal.get('hypothesis', '?')}")

  # 2. Snapshot + edit configs
  snapshot_files(run_dir, proposal)
  call_editor(run_dir)

  # 3. Train
  try:
    launch_training(run_dir, PROBE_ITERATIONS)
  except Exception:
    log.exception("Training failed — reverting changes")
    revert_changes(run_dir)
    return

  # 4. Score
  score = evaluate_run(run_dir)

  # 5. Analyse
  analysis = call_analyst(run_dir)

  # 6. Update state
  if score["score"] > state.best_score:
    log.info(f"New best: {score['score']:.3f} (was {state.best_score:.3f})")
    state.best_score = score["score"]
    state.best_run_id = run_dir.name
  else:
    # Did not improve — revert so the next experiment starts from current best.
    log.info(f"Did not beat current best ({state.best_score:.3f}); reverting")
    revert_changes(run_dir)

  # 7. Append to history
  append_history(
    {
      "run_id": run_dir.name,
      "proposal": proposal,
      "score": score,
      "analysis": analysis,
    }
  )
  state.total_experiments += 1
  state.experiments_since_feedback += 1
  state.save()

  # 8. Maybe ask for human feedback
  if state.experiments_since_feedback >= EXPERIMENTS_PER_FEEDBACK:
    request_human_feedback(state)
    state.experiments_since_feedback = 0
    state.save()


def main() -> None:
  RUNS_DIR.mkdir(exist_ok=True)
  FEEDBACK_DIR.mkdir(exist_ok=True)
  state = State.load()
  log.info(f"Starting orchestrator. Best so far: {state.best_score:.3f}")

  while True:
    try:
      run_one_experiment(state)
    except KeyboardInterrupt:
      log.info("Stopped by user")
      return
    except Exception:
      # An experiment can fail for many reasons (training crash, parser
      # error, etc). Log and move on rather than killing the whole loop —
      # the next experiment starts from current best.
      log.exception("Experiment failed; continuing to next iteration")
      _sleep_with_heartbeat(60.0, "next experiment")


if __name__ == "__main__":
  main()
