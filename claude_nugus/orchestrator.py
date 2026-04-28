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
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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

# Polling intervals.
LOG_POLL_SECONDS = 10.0
FEEDBACK_POLL_SECONDS = 30.0

# Human feedback cadence.
EXPERIMENTS_PER_FEEDBACK = 5

# Early-failure gate: if late-iteration fell_over rate exceeds this past
# this iteration count, kill the run early and score it 0.
EARLY_KILL_FELL_OVER = 1.5
EARLY_KILL_AFTER_ITER = 750


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


async def _query_agent(
  prompt: str,
  system_prompt: str,
  output_format: dict[str, Any] | None = None,
  max_turns: int = 5,
  permission_mode: sdk.PermissionMode = "default",
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
  )
  async for message in sdk.query(prompt=prompt, options=options):
    if isinstance(message, sdk.ResultMessage):
      if message.is_error:
        details = message.result or (
          "; ".join(message.errors) if message.errors else "unknown"
        )
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


def _run_agent(prompt: str, system_prompt: str, **kwargs: Any) -> Any:
  """Sync wrapper around the async SDK query."""
  loop = _get_event_loop()
  return loop.run_until_complete(_query_agent(prompt, system_prompt, **kwargs))


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
    max_turns=3,
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


def launch_training(run_dir: Path, target_iterations: int) -> Path:
  """Launch training, poll its log, and kill it at the target iteration.

  Returns: path to the training_log.json that was produced, copied into
  run_dir/training_log.json.
  """
  log.info(f"Launching training (target: {target_iterations} iters)")
  launch_time = time.time()
  proc = subprocess.Popen(
    TRAIN_CMD,
    cwd=str(REPO_ROOT),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  log.info(f"  pid: {proc.pid}")

  # Find the training log directory the subprocess created.
  log_dir: Path | None = None
  for _ in range(60):  # wait up to 10 minutes for first log file
    time.sleep(LOG_POLL_SECONDS)
    log_dir = find_latest_log_dir(launch_time)
    if log_dir and (log_dir / "training_log.json").exists():
      log.info(f"  found log: {log_dir}")
      break
  else:
    proc.send_signal(signal.SIGTERM)
    raise RuntimeError("Training did not produce a log within 10 minutes")

  assert log_dir is not None
  log_path = log_dir / "training_log.json"

  # Poll the log until the target iteration is reached.
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

    # Early-kill check: stability gate.
    if current >= EARLY_KILL_AFTER_ITER:
      recent_falls = training_log.get("Episode_Termination/fell_over", [])
      if recent_falls and recent_falls[-1] > EARLY_KILL_FELL_OVER:
        log.warning(f"Early kill: fell_over={recent_falls[-1]:.2f} at iter {current}")
        break

    if current >= target_iterations:
      log.info(f"Reached target iteration {target_iterations}, killing training")
      break

  proc.send_signal(signal.SIGTERM)
  try:
    proc.wait(timeout=30)
  except subprocess.TimeoutExpired:
    log.warning("SIGTERM ignored, sending SIGKILL")
    proc.kill()
    proc.wait()

  # Copy the log into our run dir for permanence.
  dest = run_dir / "training_log.json"
  shutil.copy(log_path, dest)
  return dest


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

  score = score_run(summary["final_window"])  # type: ignore[arg-type]
  with open(run_dir / "score.json", "w") as f:
    json.dump(score, f, indent=2)

  log.info(
    f"Score: {score['score']:.3f}"
    + (f"  ({score['gate_failed']})" if score["gate_failed"] else "")
  )
  return score


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback handoff
# ─────────────────────────────────────────────────────────────────────────────


def request_human_feedback(run_dir: Path, state: State) -> None:
  """Pause until the human writes their feedback into pending.json."""
  pending_path = FEEDBACK_DIR / "pending.json"
  request = {
    "run_dir": str(run_dir),
    "best_run": state.best_run_id,
    "questions": [
      "leg_coordination_1to5",
      "foot_contact_quality_1to5",
      "naturalness_1to5",
      "specific_issues_freetext",
    ],
    "instructions": (
      "Watch the policy from this run, then fill in the answer fields "
      "below and save. The orchestrator will detect the change."
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

  log.info(f"⏸  Awaiting human feedback at {pending_path}")
  while True:
    time.sleep(FEEDBACK_POLL_SECONDS)
    with open(pending_path) as f:
      current = json.load(f)
    if all(v is not None for v in current["answers"].values()):
      break

  feedback_path = run_dir / "feedback.json"
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
    request_human_feedback(run_dir, state)
    state.experiments_since_feedback = 0
    state.save()


def main() -> None:
  RUNS_DIR.mkdir(exist_ok=True)
  FEEDBACK_DIR.mkdir(exist_ok=True)
  state = State.load()
  log.info(f"Starting orchestrator. Best so far: {state.best_score:.3f}")

  try:
    while True:
      run_one_experiment(state)
  except KeyboardInterrupt:
    log.info("Stopped by user")


if __name__ == "__main__":
  main()
