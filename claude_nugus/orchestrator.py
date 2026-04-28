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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
RUNS_DIR = WORKSPACE / "runs"
FEEDBACK_DIR = WORKSPACE / "feedback_queue"
HISTORY_FILE = WORKSPACE / "experiment_history.jsonl"
STATE_FILE = WORKSPACE / "state.json"
MANIFEST_FILE = WORKSPACE / "manifest.json"

# Training launch — fill these in for your environment.
TRAIN_CMD = [
    "uv", "run", "train",
    "Mjlab-Velocity-Flat-Nubots-Nugus",
    "--env.scene.num-envs", "4096",
    "--log_terminal", "True",
]
# Where the training process writes its output dir. The orchestrator looks for
# the most recent timestamped folder here after launch to find this run's log.
TRAINING_LOG_ROOT = Path("../logs/rsl_rl/nugus_velocity/") 

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
            return cls(best_score=-1.0, best_run_id=None,
                       experiments_since_feedback=0, total_experiments=0)
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
# Agent stubs — wired up to the Claude Code SDK in the next step
# ─────────────────────────────────────────────────────────────────────────────

def call_planner(run_dir: Path) -> dict:
    """Produce next experiment proposal.

    Reads: manifest, history, interpretation rules, current best config
    Writes: run_dir/proposal.json with structure:
        {
            "hypothesis": "...",
            "changes": {param_name: new_value, ...},
            "expected_effects": ["metric X should increase", ...],
            "probe_iterations": 3000
        }
    """
    raise NotImplementedError("Planner agent — TODO: wire to Claude Code SDK")


def call_editor(run_dir: Path) -> None:
    """Apply the proposal's changes to the actual config files.

    Reads: run_dir/proposal.json, manifest
    Writes: run_dir/diff.txt summarising the file edits made
    Side effect: edits source files in-place
    """
    raise NotImplementedError("Editor agent — TODO: wire to Claude Code SDK")


def call_analyst(run_dir: Path) -> dict:
    """Analyse a completed run and produce a structured summary.

    Reads: run_dir/compressed.json, run_dir/score.json, run_dir/proposal.json,
           interpretation rules
    Writes: run_dir/analysis.json with structure:
        {
            "verdict": "promising" | "neutral" | "regression",
            "key_observations": ["...", ...],
            "next_suggestion": "..."
        }
    """
    raise NotImplementedError("Analyst agent — TODO: wire to Claude Code SDK")


def revert_changes(run_dir: Path) -> None:
    """Undo the file edits this run made.

    Reads run_dir/diff.txt (or a snapshot of the original values from the
    proposal + manifest) and reverses them. Called when an experiment fails
    or scores worse than the current best.
    """
    raise NotImplementedError("Revert logic — TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Training subprocess management
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_log_dir(after_time: float) -> Path | None:
    """Find the most recent training output dir created after `after_time`."""
    if not TRAINING_LOG_ROOT.exists():
        return None
    candidates = [
        d for d in TRAINING_LOG_ROOT.iterdir()
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
                log.warning(
                    f"Early kill: fell_over={recent_falls[-1]:.2f} at iter {current}"
                )
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

def evaluate_run(run_dir: Path) -> dict:
    """Compress + score the training log. Returns the score record."""
    from metrics_compression import compress_log
    from scoring_function import score_run

    summary = compress_log(run_dir / "training_log.json")
    with open(run_dir / "compressed.json", "w") as f:
        json.dump(summary, f, indent=2)

    score = score_run(summary["final_window"])  # type: ignore[arg-type]
    with open(run_dir / "score.json", "w") as f:
        json.dump(score, f, indent=2)

    log.info(f"Score: {score['score']:.3f}"
             + (f"  ({score['gate_failed']})" if score['gate_failed'] else ""))
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

    # 2. Edit configs
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
    append_history({
        "run_id": run_dir.name,
        "proposal": proposal,
        "score": score,
        "analysis": analysis,
    })
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