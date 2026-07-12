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
    UPDATE (user, 2026-07-10): duck walk predates the bus-voltage model
    (present since the first successful 10k run, pre current-obs), so
    the driver is the VELOCITY limits, which exist in every era: two
    orthogonal hip servos at w_max swinging the leg along the 45-degree
    diagonal compose to sqrt(2)*w_max (+41% foot speed), and 45 degrees
    is the equal-sharing optimum — matching the observed angle. The
    frontier curriculum pushes the speed ceiling, so it selects for
    this geometry every run. User has NOT decided the duck walk is bad:
    it may be the honest optimal gait for this morphology (same status
    as the froude cadence), and it likely also speeds recovery
    stepping. Treat as a decision to PRICE, not a defect to fix:
    (1) measure: vel_sat telemetry — do hip servos saturate in swing at
    frontier speeds; eval sweep — foot yaw vs commanded speed (theory
    predicts duck angle grows with speed, ~vanishes at 0.3 m/s);
    (2) if hardware/RoboCup constraints (kick alignment, bracket edge
    loading, downstream odometry assumptions) demand straight feet, run
    a FOOT_HEADING_W ablation pair and read off the exact frontier and
    recovery cost before deciding. The stance-gated roll-attitude cost
    (i) stands on its own for bracket-loading reasons regardless.
    UPDATE 2 (user, 2026-07-10): duck walk appears EARLY and SLOW too
    (v51-s2 by iter 1600, pre-saturation), so velocity-limit sharing is
    not the sole driver. Three-mechanism model, speed-layered:
    (1) toe-out as lateral ballast at slow speed (long single support;
    the froude slow-cadence tether amplifies this) — real physics, the
    same reason toddlers toe-out; (2) quadratic energy penalties pay
    half price for torque split across two servos (tau^2 vs
    2*(tau/sqrt(2))^2 = tau^2/2) — the 45-degree stance lets every hip
    and ankle torque be split; operates from iter 0 at all speeds, and
    is also true I2R physics; (3) feet_distance measures
    center-to-center body-Y only, so yawing widens effective support
    for free. Mechanisms 1-2 apply to the real robot identically,
    strengthening "honest optimal gait". Discriminating ablations if
    ever wanted: wider feet_distance nominal or stiffer ankle-roll
    authority (kills 1); linear-|tau| or per-servo thermal energy term
    (removes 2's discount without banning the geometry).
    UPDATE 3 (user, 2026-07-10): possible fourth mechanism — the
    command sampler moved from a box to radius-direction (ellipsoid)
    sampling with the competence curriculum, making axis-pure extremes
    (fast pure-forward AND fast pure-lateral) common where box corners
    made diagonals the extremes. 45 degrees is the minimax servo
    orientation serving both axis-pure extremes with the same geometry.
    Discriminators: duck angle vs command direction within a rollout
    (per-command reorientation vs fixed compromise angle), and timeline
    (did duck predate the rho sampler? then ellipsoid amplified, not
    caused).
    When: after the v51-s2 seed replicate; do not stack with the v52
    duty-factor change.
11d. ⭐ **Take the head away from the policy** (user observation,
    2026-07-10: head flails at a frequency unrelated to gait cadence).
    Mechanism (CORRECTED after measurement, 2026-07-10): my first guess
    was "entropy parks extra std on the reward-flat head dims". MEASURED
    the v48 actor per-dim log_std (std_type=log, genuinely per-dim, 21
    dims) directly from the checkpoint — it is UNIFORM ~0.13 across all
    joints (legs, arms, head, phase all 0.1297-0.1300). So the head is
    NOT allocated extra exploration noise; the prediction is falsified.
    Real mechanism: exploration noise is uniform, but the SAME noise
    produces a large visible excursion on the head because it is light
    and carries no balance load (nothing resists it), while the reward
    surface there is too flat for the policy to bother cancelling the
    resulting motion with its mean output. Same noise everywhere, only
    the head has no restoring force. Still white-at-policy-rate, hence
    the frequency mismatch. Implication for fixes: a per-dim entropy or
    std tweak will NOT help (std is already uniform) — the lever is
    either removing the noise source (take head out of action space) or
    giving the reward surface a reason to hold it (head-motion damping
    cost). Same logic likely applies to the arms (light, low-load);
    re-check the arm dims when the v51 checkpoints download.
    Fix (also a deployment-fidelity correction): on hardware the head
    is the VISION system's actuator, not the walk policy's. Remove
    neck_yaw/head_pitch from the action space and drive them with
    scripted scan-pattern / ball-tracking-like trajectories as a
    training disturbance the policy must tolerate. Strictly more
    realistic than both current flail and a rigid head. Cost: medium
    (action/obs dim change, from-scratch only — which is standard
    anyway). When: bundle with the next from-scratch reward-shaping
    batch (candidates: with 11b arm work).

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

15c. ⭐ **Perturbed sim2sim gate: adaptation evidence before hardware**
    (from Trent's observation, 2026-07-12: the vanilla backend is "just
    another realization" — DR + the RMA student IS the sysid, so
    cross-engine transfer should work if the gap is in-envelope, and the
    v53 student passing the fixed D1 gate with zero falls confirmed it).
    Extend `sim2sim_eval` to sweep a few servo/physics perturbations in
    vanilla MuJoCo — ±10% kp/kd (scale `NugusServoParams`), sagged
    effort limits (bus-brownout proxy), heavier torso, low friction —
    and compare student tracking/falls across cells, ideally against a
    frozen-`z_hat` ablation (feed the zero-history prior instead of the
    estimator output). Student degrading gracefully where the ablation
    degrades hard = direct evidence the ADAPTATION transfers, not just
    the mean policy. Cost: low (loop over `build_servo_params`
    perturbations + one ONNX ablation flag). When: before first hardware
    deployment of an RMA student; strongest pre-hardware evidence
    available.

15d. ⭐ **v56 design: gated latent with a LEARNED safe prior** (Trent,
    2026-07-12, extending the quiescence diagnosis after v54's verdict).
    v54 falsified the distribution-shift theory of student falls (training
    on z_hat made falls WORSE, 0.028 -> 0.045/min): the fall mechanism is
    the estimator's cold start itself, and the current "prior" is an
    accident — TCN(backfilled window), an arbitrary pose-dependent point.
    Design: ``z_eff = (1-g) * z0 + g * z_signal`` where
    - ``z0`` is a learned 16-dim parameter, trained through the POLICY
      path (PPO, undetached) in the low-evidence regime — so it converges
      to the SAFE hedged operating point (fall-cost asymmetry priced by
      reward), not the prior mean (which is what regression would give);
    - ``g`` is a learned gate emitted by the TCN alongside z_hat
      (Kalman-gain analogue, optionally per-dim), trained with the
      regression loss against ``sg(z)``;
    - ``z_signal`` = encoder-z in training, TCN z_hat at deployment: the
      anneal lesson applied — train-on-prior exactly where deployment
      runs on the prior (no information: identical by construction), and
      train-on-truth where evidence exists. Keeps the supervised anchor
      (v54 showed the self-referential regime without one degrades).
    Staging, forced by PPO re-forwarding stored obs in minibatches (no
    cross-step state on the policy path):
    - Stage 1 (training): gate from the CURRENT window only — stateless,
      minibatch-safe, kills the measured early-episode fall cluster.
    - Stage 2 (deployment only): robot carries ``(z_state, G)`` across
      ticks with an EMA hold — identify while walking, hold through
      standing. 17 floats of state; benign train/deploy gap (deployment
      holds a BETTER estimate than the training window could).
    EXTENDED (Trent, 2026-07-12, after v55's early track=2.5): the module
    becomes a THREE-HEAD TCN — ``z_hat`` (params, anchored to sg(z)),
    ``v_hat`` (base-frame linear velocity, supervised on sg(base_lin_vel);
    ground truth free in sim), ``g`` (gate). Motivation: the actor has no
    velocity sensor (base_lin_vel is critic-only) yet is rewarded on
    velocity tracking — 0.5 s of history is a leg odometer, and v55's
    tracking jump suggests that is the window's most valuable content
    (the v53/v54 anchor CONSTRAINED the channel to params-only and
    discarded it; linear-probe of v55 vs v53 trunk features for
    base_lin_vel is the confirmation test). ``v_hat`` also EXPORTS as a
    second ONNX output ("velocity", body frame, policy rate):
    walk-coupled learned odometry for the localization stack — sees
    contact timing/slip/IMU/currents AND the commanded gait, so it learns
    the gait's systematic biases (duck drift, slip) that kinematic
    odometry cannot; trained across the DR envelope so it is robust to
    realization; sim gives its RMSE spec for free (add to eval battery).
    Optional later head: foot-contact probabilities. Fast path: if the
    v55 probe confirms, train a post-hoc frozen-trunk readout on rollout
    data and splice into the existing ONNX — odometry without retraining.
    Cost: moderate (model change + tests; no env change). When: after the
    v55 e2e verdict — v55 decides whether the anchor stays, 15d decides
    how the prior/gate/heads work; they compose.

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
