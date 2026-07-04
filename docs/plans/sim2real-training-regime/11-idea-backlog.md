# Idea backlog — curricula, reward terms, DR (ranked)

Candidate experiments beyond the overnight plan (doc 10). Each entry: what,
why, cost, and when to reach for it. Overnight backfill slots (doc 10 R17/
R20) may draw from the ⭐ items if their prerequisites are met. Do not run
un-starred items overnight — they need design attention.

## Curricula

1. ⭐ **Adaptive command curriculum (grow-with-competence).** Per-env command
   ranges widen when that env's tracking error is low and shrink when high —
   the `terrain_levels` pattern applied to commands. Replaces time-scheduled
   command widening (the v11–v15 lesson: time-based staging destabilizes;
   competence-based staging is self-correcting). Cost: medium (one
   curriculum term mirroring `terrain_levels_vel`). When: after the doc 10
   winner is stable; this is the healthy path back toward hard-task command
   ranges.
2. ⭐ **Adaptive push/DR curriculum (ADR-lite).** Push magnitude (later: DR
   widths) scales up when fell_over EMA < threshold, down when above —
   keeps difficulty at the edge of competence automatically (OpenAI ADR
   idea, simplified to one scalar). Cost: medium. When: same trigger as #1;
   candidate replacement for the entire hand-tuned hard_continue.
3. **Speed-coupled gait period** (doc 07 Stage 2). Period as part of the
   command, sampled coupled to |v_cmd|. Cost: medium-high (per-env period
   tensor through reward + obs — the known desync trap). When: after the
   doc 10 cadence grid (R6/R7) confirms stride length binds tracking.
4. **Stop/transition polish curriculum.** Raise `rel_stop_envs`/standing
   fraction late in training to sharpen walk↔stand transitions (RoboCup
   games are stop-heavy). Cost: trivial (env knob). When: once gait quality
   is acceptable; before first hardware demo.
5. **Terrain curriculum** (rough task registered, never run). When: after a
   flat winner passes eval; start with the doc 10 R20 taste run.

## Reward terms / weights

6. ⭐ **Single-support (biped air-time) reward.** Reward single-stance time /
   penalize prolonged double-support (the biped air-time variant from
   Isaac-family configs). Directly attacks shuffle without touching swing
   height. Cost: low (term exists in mjlab base or is a small variant).
   When: if air_time still lags after doc 10 wave 1.
7. ⭐ **Stride-length reward.** Reward per-step foot displacement ∝ commanded
   speed (measure foot travel between touchdowns vs `v_cmd × period/2`).
   Attacks the stride ceiling directly rather than hoping tracking reward
   finds long steps. Cost: low-medium (touchdown bookkeeping exists via
   air-time internals). When: if R6/R7 show cadence alone doesn't fix
   tracking at 0.5 m/s.
8. **base_height std widening / weight drop during walk.** Constant-height
   target (0.47 m, std √0.05) suppresses the natural 1–2 cm vertical gait
   oscillation and encourages crouch-shuffle. Try std ×2 or weight 0.3→0.1
   while walking (keep for standing). Cost: trivial. When: wave-1 backfill
   if the gait looks crouched in the viewer.
8b. ⭐ **Loosen the upright orientation gradient.** Current
   `0.5·exp(−sin²θ/0.2)` has a small-tilt gradient equivalent to L2 weight
   −2.5 — ~2.5× the in-repo G1 reference (−1.0) — pinning the torso at
   ~8–10° and taxing hip-strategy balance and forward lean. Cells:
   (a) `UPRIGHT_STD2` 0.2→0.4 at weight 0.5 (matches G1-scale gradient);
   (b) plus weight 0.5→0.35. Keep/strengthen `body_ang_vel` as the actual
   torso-STABILITY term (orientation ≠ stability). Judge on fixed-eval
   falls at wide commands + tracking error. Notes: 50° termination is the
   backstop; upright is also ~10% of the shuffle-income cocktail so this
   may help tracking too; the v9 "upright cut destabilizes" precedent is
   unreliable (pre-entropy-fix, and hard_continue's schedule tightened std
   while cutting weight — self-cancelling). Cost: trivial (one knob).
9. **Torso pitch offset command.** Small commanded forward lean while
   walking (kid-size robots walk better with 3–8° lean; most RoboCup teams
   do this). Implement as a pose-target offset on hip_pitch/torso, or as a
   lean-target in the upright term (`exp(−‖g_xy − g_lean‖²)`) — the natural
   follow-on to 8b. Cost: low. When: after 8b; visual assessment first.
10. **Mechanical-power energy term.** Replace `joule_heating` (τ²) with the
    existing disabled `actuation_power` (|τ·qd|) once gait is stable —
    better battery-life correlate for hardware. Keep weight ~1e-5-scale.
    Cost: trivial (term exists, disabled). When: pre-hardware tuning, not
    during gait search.
11. **Landing-attitude window term** (heel-toe refinement round 2): if the
    doc 10 one-sided foot_flat isn't enough, replace with a clock-windowed
    pre-touchdown attitude target (heel-first: slight toe-up at touchdown).
    Cost: low (clock gating machinery exists). When: after R5 results.

## Domain randomization

12. ⭐ **IMU mounting-orientation DR.** Small fixed per-episode rotation
    (±2–3°) applied to gyro + projected-gravity obs — real IMU mounting is
    never perfect and the OP3 soccer work randomized exactly this. Cost:
    low (obs-term rotation, startup mode). When: any batch after the doc 10
    winner; cheap sim2real insurance.
13. ⭐ **Ground-incline DR.** Tilt the world/gravity by ±2–3° per episode —
    RoboCup fields and labs are not level; cheaper than rough terrain and
    targets a guaranteed real-world condition. Cost: low (gravity vector
    rotation at reset). When: with #12.
14. **Low-friction floor cell.** foot_friction lower bound 0.4 (waxed
    RoboCup turf/tiles). Cost: zero (range change). When: robustness wave
    after the winner; watch slip metrics.
15. **Restore full-width mass/payload DR** — already doc 10 R14.

## Hygiene

15b. ⭐ **Randomize initial episode clocks.** All envs start at
    `episode_length_buf = 0` simultaneously and timeout preserves cohort
    phase, so envs stay synchronized (frozen at whatever distribution
    existed when the policy stopped falling) — this is why
    `Episode_Termination/time_out` oscillates as a ~41.7-iteration
    (1000/24) sine instead of sitting flat at num_envs/1000 ≈ 8.2/step.
    Fix: on the FIRST reset only, initialize `episode_length_buf` uniformly
    over [0, max_episode_length). Benefits: flat/legible timeout metric,
    and decorrelates PPO batches by episode phase (post-reset transients
    and truncation bootstraps stop arriving in synchronized bursts). Cost:
    ~one line + a test that steady-state timeout counts are ~uniform.
    Caveat: partial-length first episodes slightly perturb the first few
    iterations' stats — irrelevant beyond iter ~50.

## Learning / architecture (log only — not overnight material)

16. **num_steps_per_env 24 → 48.** Longer rollout horizon for gait credit
    assignment; halves update frequency at same batch. Cost: config-only,
    but changes optimizer dynamics — needs its own A/B.
17. **AMP-style motion prior** from retargeted walk clips (the existing
    classical walk engine can generate reference trajectories!). Replaces
    hand-tuned style terms wholesale. Cost: high (discriminator training).
    When: only if reward-shaping iteration stalls — but note the reference
    source is uniquely cheap here (the robot already walks classically).
18. **Obs history 5 frames** — C4 step 1 (doc 04); gated on hardware
    evidence per plan.

## Explicitly rejected for now (with reasons; see docs 00/06/08)

- Anneal-away of gait shaping (v9/v16 collapses), clock_learned free phase
  (F3 collapse), effort limits below stall (v16b), joint_acc ≥ −1e-4
  (v16c tracking kill), time-scheduled hard_continue as-is (v11–v15).
