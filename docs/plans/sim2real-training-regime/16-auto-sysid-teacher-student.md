# Doc 16 — Automatic sysid: privileged teacher → history student

Plan for implicit system identification via teacher–student distillation
(user direction, 2026-07-07). The referenced claude.ai chat
(48ca4858-…) is auth-gated and could not be read; this plan is built
from the stated design ("small network on the hidden DR variables as a
teacher, student trained off it using the latent timing variables"),
the literature it describes, and what already exists in this repo.
Written while v45 (open-road) runs; execution starts after its verdict.

## 0. Goal

A hardware-deployable policy that adapts online to the real robot's
dynamics (masses, frictions, gains, effort limits, contact properties)
without explicit parameter estimation: the "sysid" is implicit — a
network infers what it needs about the plant from how the robot has
been responding, and conditions the gait on it. This is the standard
RMA / privileged-distillation paradigm (Kumar et al. RMA; Lee et al.
2020), adapted to what mjlab already has.

## 1. What already exists (audit, 2026-07-07)

- **The privileged vector is already built**: `dr_observations.dr_ratios`
  (critic-only today) — 106 dims: per-actuator kp/kd/effort/armature/
  frictionloss ratios (5×20) + torso mass ratio + torso CoM offset (3)
  + foot friction (2). This is the "hidden DR variables" input, done.
  Gaps to add: payload mass ratio, push-cohort membership is NOT
  included (correct — pushes are transient events, not plant params).
- **rsl-rl ships the machinery**: `algorithms/distillation.py`
  (`Distillation`: student mimics teacher with `loss_type=mse`,
  `gradient_length` for BPTT) + `runners/distillation_runner.py`
  (rolls out the STUDENT, supervises against the loaded teacher —
  on-policy distillation, the DAgger property we want, for free) +
  `modules/rnn.py` and `rnn_type` model config (our runner cfg already
  strips `rnn_type: None`, i.e. recurrent models are supported).
- **Timing latents**: the actor observation already carries the gait
  clock (sin/cos phase) and the policy-owned phase delta — the "latent
  timing variables". A recurrent student consumes these plus
  proprioception history natively; no extra plumbing.
- **Training regime**: post-R29 the sim is honest and the frontier
  curriculum (R19–R30) produces healthy indefinitely-long runs — the
  precondition this whole plan was waiting on.

## 2. Architecture choice

Three candidate routes, one recommendation:

- **Route A — action distillation (RECOMMENDED)**: teacher = policy
  trained by PPO with `dr_ratios` appended to its actor observations
  (privileged actor). Student = recurrent (GRU) policy on deployable
  observations only, trained by `DistillationRunner` to match teacher
  actions on student-generated rollouts. No explicit latent z: the
  GRU hidden state IS the sysid latent. Least custom code (rsl-rl
  native), one supervised phase, and empirically matches explicit-z
  RMA on locomotion. The user's "small network on the hidden DR
  variables" = the teacher's first MLP layers over dr_ratios; the
  "latent" the student learns is implicit in its recurrent state.
- **Route B — explicit-z RMA**: factor encoder μ(dr_ratios)→z (dim
  ~8–16) trained inside PPO; student adaptation module φ(history)→ẑ
  regressed against z. More custom code (encoder in the model,
  latent-regression trainer), buys interpretability of z (nice for
  debugging which plant factors matter) and a smaller student head.
  Keep as fallback / phase-2 refinement if Route A's student–teacher
  gap exceeds the bar, or if we want z-space diagnostics.
- **Route C — concurrent estimation** (single phase: auxiliary head
  predicts dr_ratios from history during PPO itself): cheapest, no
  second phase, but couples estimator quality to PPO batch noise and
  historically underperforms distillation at DR corners. Not chosen.

## 3. Phase T — the privileged teacher (≈1 day of work, one 8-GPU run)

1. Config: new obs group wiring — append `dr_ratios` to the ACTOR
   group for the teacher variant (knob `PRIVILEGED_ACTOR=1` in
   env_cfgs; the term exists, this is plumbing). Normalizer: dr_ratios
   are already ratios ≈ O(1) — include in the frozen-at-500 empirical
   normalization like everything else (R15).
2. Widen DR ranges modestly at first (current ranges: effort/kp/kd/
   armature/damping ±20–50%): sysid is only worth learning if the
   plant actually varies. Use existing knobs; the frontier/valve
   machinery (R19–R30) absorbs the difficulty. A later DR-rungs pass
   (doc 15 backlog) widens further.
3. Run: v46-teacher = v45 stack + PRIVILEGED_ACTOR, 8 GPUs, ~6000
   iters (v44/v45 show training compounds; land on plateau).
4. Go/no-go: teacher ≥ v45-champion metrics (attain, frontier, falls)
   AND visibly exploits privilege — check: teacher return on wide-DR
   eval beats a blind v45 checkpoint by a measurable margin. If the
   privileged actor is NOT better under wide DR, sysid has nothing to
   identify at current ranges → widen DR and rerun before investing
   in the student.

## 4. Phase S — the history student (≈1–2 days of work, one run)

1. mjlab integration: `MjlabDistillationRunner` wrapping rsl-rl's
   `DistillationRunner` the same way `MjlabOnPolicyRunner` wraps
   `OnPolicyRunner` (checkpoint env-state, ONNX export, W&B). Runner
   selection by a `RUNNER=distill` knob in train.py. Reuse the k8s
   harness (gen cell v47-student) — distillation is cheap (supervised;
   expect ~1–2 h on 4 GPUs).
2. Student model: same trunk sizes as teacher, `rnn_type: gru` (one
   layer, 128–256 hidden), observations = deployable set only (current
   actor obs: proprioception, projected gravity, commands, clock +
   phase delta, prev actions). `gradient_length` 15–24 (≈0.3–0.5 s of
   BPTT; RMA uses ~0.35 s of history).
3. Teacher loading: rsl-rl requires the teacher checkpoint loaded into
   the Distillation alg (`teacher_loaded`); config points at the
   v46-teacher artifact on the logs PVC.
4. Curriculum during distillation: freeze difficulty at the teacher's
   final frontier (env knobs already allow pinning: set envelope and
   d-related gates static) — the student should imitate across the
   full operating envelope, not re-run the curriculum. Sample commands
   across the whole ellipsoid; keep the push cohort active (recovery
   behavior must distill too).
5. Go/no-go: student ≥95% of teacher return under full training DR,
   AND the adaptation probe (below) shows recovery, not just average
   competence.

## 5. Phase V — validation: does it actually identify?

The metric that distinguishes sysid from robustness: **mid-episode
plant switches**. Extend the eval script with an adaptation probe:
at t=5 s, silently change payload mass / foot friction / effort scale
to a held-out value; measure tracking error over the next 3 s vs a
static-plant episode.

- Teacher (sees dr_ratios) = upper bound; blind v45 champion = lower
  bound (pure robustness); the student should land near the teacher.
  If student ≈ blind baseline, the GRU learned robustness, not
  identification → Route B (explicit z + regression loss) is the
  escalation path.
- Held-out DR grid: evaluate all three on parameter values outside
  the training ranges (±10–20% beyond) — sysid should degrade more
  gracefully than robustness.
- Sim2sim: the Webots gate (doc 15 backlog) runs the STUDENT — it is
  the deployable artifact. Note: ONNX export of a recurrent policy
  needs hidden-state I/O in the export path (mjlab's exporter handles
  MLPs today; check `as_onnx` for RNN support — flagged risk).

## 6. Work items (file-level)

| item | where | size |
|---|---|---|
| `PRIVILEGED_ACTOR` obs wiring | `config/nugus/env_cfgs.py` (append dr_ratios to actor group) | S |
| payload ratio into dr_ratios | `config/nugus/dr_observations.py` | S |
| `MjlabDistillationRunner` | `src/mjlab/rl/runner.py` (mirror existing wrapper) | M |
| runner selection + teacher-path cfg | `scripts/train.py`, `rl/config.py` | S |
| student cfg (gru, deployable obs) | `config/nugus/rl_cfg.py` | S |
| curriculum pinning knobs for distill | env_cfgs (mostly exists) | S |
| gen cells v46-teacher / v47-student | `scripts/k8s/gen-gridsearch.sh` | S |
| adaptation probe | `scripts/nugus_eval.py` extension | M |
| RNN ONNX export check | `rl/runner.py` `export_policy_to_onnx` | M (risk) |

## 7. Risks

- **Nothing to identify**: if DR ranges are too narrow, privilege is
  worthless and the student trivially matches — caught by the Phase T
  go/no-go before any student work.
- **Robustness masquerading as sysid**: caught by the adaptation
  probe; escalation = Route B explicit latent.
- **RNN state across resets**: rsl-rl's rollout storage handles dones
  for recurrent models; verify hidden-state zeroing on env reset in
  the distillation path (test it — same class of bug as R24's
  censoring subtleties).
- **RNN on hardware/Webots**: hidden-state carry in the NUClear/
  Webots controller loop; ONNX export risk above. Fallback: stacked
  observation history (K=25–50 frames) MLP student instead of GRU —
  supported by the same distillation runner, trivially exportable,
  slightly worse ceiling.
- **Normalizer mismatch teacher↔student**: each model owns its
  normalizer (rsl-rl); student trains its own on deployable obs —
  fine, but freeze at the same sample budget (R15) for consistency.

## 8. Sequencing

1. After v45 verdict (open-road ceiling + champion refresh).
2. Phase T build + v46-teacher run → go/no-go.
3. Phase S build + v47-student distillation → go/no-go.
4. Phase V probes; Webots gate on the student.
Total: roughly 3–4 working days of build interleaved with ~3 cluster
runs, all on the now-trusted training stack.
