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
11b. ⭐ **Arm-flail damping** (user observation, 2026-07-10: arms flail
    hard in every walk, persisting to late training). Why joule_heating
    doesn't catch it: the term is Σ τ² over ALL joints, and arm links are
    ~0.3 kg on MX-64s while the legs are XH540s carrying the whole 7.5 kg
    robot — leg torques dominate the sum by orders of magnitude, so the
    energy gradient is nearly blind to arm motion (τ² is doubly blind to
    the high-velocity/low-torque regime flailing lives in; see item 10).
    Also suspect the flailing is partly BOUGHT by the angular_momentum
    penalty: arms are the cheapest way to cancel root angmom
    (reaction-wheel function), so it may be functional, not waste.
    FIRST: verify with an eval rollout (correlate arm joint velocity with
    root angmom cancellation; log arm |qd| mean and arm share of Σ|τ·qd|)
    before suppressing — if functional, prefer smoothing over pose-holds
    or we may destabilize the walk. Cells, in preference order:
    (a) per-servo-class energy: replace τ² with (τ/Kt)² (true I²R joule
    heating; MX-64's smaller Kt then upweights arm heating for free, no
    magic constants);
    (b) arm-scoped joint_acc/action_rate cost (damps flail, keeps slow
    functional swing);
    (c) arm pose-hold to nominal while walking (strongest suppression,
    highest risk of removing balance function);
    (d) arm action-scale reduction (crude, changes the action space).
    Cost: low ((a),(b),(d) are knobs/small terms). When: next reward-
    shaping batch after v51 settles the clock-grounding question; (a) is
    also pre-hardware hygiene independent of flailing.
11c. **Foot-attitude blind spots** (user observation on the v51-s1 gait,
    2026-07-10: lateral "heel-toe" rolling outside-edge to inside-edge,
    feet yawed ~45 degrees). Two reward gaps make this free:
    (i) `foot_flat` penalizes only SWING feet (deliberately, to allow
    terrain conformance in stance), so edge-rolling DURING stance is
    unpenalized — and it composes badly with the v51 contact windows,
    which demand the foot stay planted through its stance window
    (edge-rolling keeps "contact" while the body walks over the foot);
    (ii) nothing constrains foot YAW anywhere in the stack (`foot_flat`
    projects gravity into the foot frame; yaw about vertical is
    invisible to it). The 45-degree stance plausibly load-shares sagittal
    work across pitch+roll servos at ankle and hip simultaneously
    (halves per-servo velocity/torque; relevant under velocity limits
    and the bus-voltage torque tax) — geometric motor load-balancing.
    Candidate fixes: stance-gated roll-attitude cost (roll only, keep
    pitch free for heel-toe), and a foot-heading cost tying foot yaw to
    body heading (walking-gated). Cost: low-medium. When: after the
    v51-s2 seed replicate says whether these gaits are systematic or
    seed lottery; do not stack with the v52 duty-factor change.

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

## Infrastructure

15c. **Multi-node 8-GPU single-run training (munin+hugin).** Feasible and
    algorithmically safe: iteration time is ~88% sim collection
    (collection 2.0–2.5 s vs learning 0.13–0.34 s), gradients are ~6 MB —
    NCCL over plain Ethernet is fine, and rsl-rl's data-parallel path is
    already proven single-node. Plumbing (~1 day): 2-pod Volcano job with
    per-node affinity, env-rendezvous `torchrun` instead of torchrunx SSH
    (Volcano svc plugin for MASTER_ADDR), NCCL_SOCKET_IFNAME, rank-0-only
    checkpoint/W&B verification. Honest speedup: 1.4–1.6× per run (8×4096
    sits below the 8192/GPU 4090 throughput sweet spot), NOT 2×, and it
    serializes A/B pairs (comparison arrives ~1.3× LATER than two parallel
    4-GPU runs). Use only when a single run gates everything: final 4k–8k
    consolidation, pre-hardware hard-cell training. Cheaper first: smoke
    test 4×16384 envs (only 20480 is known to OOM; 16384 untested).

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
