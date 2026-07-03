# Gait-timing strategy — variable step timing without the anneal

## The design goal (and why it's legitimate)

The clock anneal (and the clock_learned variant) existed to serve a real
requirement: the policy should NOT be locked to one gait cycle — longer,
slower steps when moving slowly; shorter, faster steps when moving quickly;
irregular capture steps under pushes. Published RL walkers demonstrably do
this (DeepMind's OP3 soccer policies take capture steps and vary cadence;
Cassie gait-family policies transition between gaits). The goal stays. What
failed is the *mechanism*: annealing the shaping reward to zero mid-run
(v9 baseline collapse on old physics, v16 collapse on new physics — see
`06-v16-collapse-analysis.md`), and letting the policy own the phase with a
penalty tying it to nominal (clock_learned: phase usage decays to zero in
every hard run — finding F3; the penalty makes deviation costly while
nothing ever makes it profitable).

## Recommended path (staged; each stage is separately shippable)

### Stage 1 — Fixed-period clock, permanent shaping (v16b, now)

`clock_persist` at GAIT_PERIOD 1.0. Boring, stable, matches every public
recipe. This is the base everything below builds on.

### Stage 2 — Gait period as a COMMAND (the main recommendation)

Instead of removing the clock, make its period part of the command space:

- Sample `period ~ U(0.8, 1.4) s` per command resample (alongside velocity).
  Optionally couple to speed: sample period from a band that narrows with
  `‖v_cmd‖` (fast command → short period band) so the policy trains
  densely on the combinations it will use.
- The clock observation already encodes phase as sin/cos; ADD the current
  period (or phase rate) as one more observation channel so the policy
  knows which cadence is commanded.
- `feet_swing_height_clock` and the gait obs must share the sampled period —
  they already share `GAIT_PERIOD`; the change is making that value
  per-env instead of global (reward params and `_gait_base_phase` both need
  the per-env tensor; this is the main implementation cost).
- Swing target height can scale mildly with period (longer step → slightly
  higher clearance) — start fixed at 0.05 and only add this if trips
  correlate with long-period samples.

This is the periodic-reward-composition approach proven on Cassie
(Siekmann et al. — see `references/reward-curriculum-symmetry.md`): the
policy learns a *family* of gaits indexed by commanded timing parameters.
At deployment you get "longer slower steps / shorter faster steps" by
scheduling the period from commanded speed (a 5-line deterministic
controller-side rule), and the whole family is sim-validated. Contrast with
clock_learned: here cadence variety is *demanded by the training
distribution*, not left to emerge against a penalty — which is why it
doesn't collapse to a single cadence.

Expected cost: medium (per-env period plumbing + obs channel + command
sampler). Run as v19 on the v16b base once stable: fixed eval must hold
falls_per_min across the period band, not just at 1.0 s.

### Stage 3 — Capture steps come from push training, not free phase

Irregular recovery steps are primarily learned from the push curriculum
(they exist within a clocked gait too — a capture step is a
phase-consistent but spatially large step in most published clocked
policies). Before concluding the clock blocks recovery: measure it — fixed
eval with scripted mid-stance pushes, count falls. If (and only if) clocked
policies demonstrably fail recovery that unclocked ones manage, add a
**perturbation phase window**: suppress the swing-height clock reward for
N steps (~0.5 s) after a detected push so the optimizer may break cadence
exactly when it pays, keeping shaping everywhere else. That is the
targeted version of what clock_learned tried to do globally.

### Stage 4 (optional endgame) — Unclocked policy (`self_paced`)

The existence proof on the closest platform: DeepMind's OP3 soccer policy
has NO gait clock at all — task + regularization rewards only, gait
emerges, cadence fully free (see
`references/small-humanoid-precedents.md`). The repo's `self_paced` variant
(no swing clock; landing + air-time terms only) approximates this and has
never been run on the modern stack. Worth ONE exploratory run after Stage 2
works, with expectations set: emergent gaits are less regular, harder to
debug, and typically need more training; if it walks at all on the new
physics it validates the endgame without betting the schedule on it.

## What NOT to do (evidence in this repo)

- Anneal shaping to zero on a step schedule (v9, v16 collapses).
- Free phase + always-on nominal-cadence penalty (clock_learned, F3: the
  penalty guarantees deviation never pays; usage ratio → 0 in every run).
- Judge any of this on `Train/mean_reward` across stages — fixed eval only.

## Sequencing note

Stage 2 changes obs dimension (period channel) → no checkpoint resume from
v16b; plan it as a fresh batch. Stage 3's push-window needs the event
system to expose "steps since last push" to the reward — small hook in the
push event. Stages are independent of Track A (actuator) work but Stage 2
conclusions are only trustworthy after the corrected velocity limits land
(06 §V5), since period feasibility depends on them.
