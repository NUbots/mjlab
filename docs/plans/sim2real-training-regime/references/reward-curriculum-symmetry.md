# Reward composition, delays, symmetry, torque injection — supporting sources

Grab-bag of the remaining sources the plan leans on, with the specific fact
each contributes.

## Periodic / clock-based reward composition (validates our reward family)

- **Siekmann et al. 2021**, "Sim-to-Real Learning of All Common Bipedal
  Gaits via Periodic Reward Composition" —
  https://arxiv.org/pdf/2011.01387 . Origin of the clock-indicator reward
  family (Cassie). Our `feet_swing_height_clock` + swing/stance shaping is
  this lineage.
- **2025 successor:** "Periodic Bipedal Gait Learning Using Reward
  Composition Based on a Novel Gait Planner for Humanoid Robots" —
  https://arxiv.org/abs/2506.08416 . Clock rewards remain current practice;
  three-term compositions with a gait planner. Take-away for us: the reward
  *family* is not the weak point of the current setup.
- **Humanoid-Gym** — https://arxiv.org/abs/2404.05695 . Canonical PPO
  zero-shot recipe (XBot-S/L); popularized the **sim-to-sim validation
  gate** (train Isaac → verify MuJoCo → hardware), which track D1 adapts as
  Warp → vanilla MuJoCo.

## Delay / latency randomization (justifies A4 and the existing delay terms)

- Multi-Modal Delay Randomization — https://arxiv.org/pdf/2109.14549 :
  randomizing policy-lag during training is what buys robustness to real
  latencies.
- Multiple platform reports (incl. HULKs student work, PACE study) state
  transfer *failed outright* until observation delay + motor dynamics were
  modeled; PACE's identified system-level delay was 7.5 ms vs 0.4 ms motor
  dead-time (details in `systematic-sim2real-pace.md`).
- Action delay and observation delay are functionally equivalent from the
  agent's perspective — randomize both, but don't double-count the same
  physical latency.

## Symmetry (justifies C2)

- "Coordinated Humanoid Robot Locomotion with Symmetry Equivariant RL
  Policy" (2025) — https://arxiv.org/html/2508.01247v1 : plain RL produces
  asymmetric-gait pathologies; equivariance/augmentation fixes them.
- Symmetric data augmentation (mirror obs/actions into the batch) is the
  common lightweight variant and appears in the FastSAC minimal recipe
  (see `fastsac-15min-recipe.md`). Preferred over symmetry *reward* terms.
- rsl-rl ships a symmetry mechanism (`symmetry_cfg`: data augmentation
  and/or mirror loss, taking user mirror functions) in recent versions —
  check the installed version before hand-rolling.

## Torque-space perturbation injection (justifies A5)

- "Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space
  Perturbation Injection" — https://arxiv.org/pdf/2504.06585 : inject
  filtered random torque disturbances per joint during training instead of
  (or in addition to) explicit actuator modeling; robustness to unmodeled
  actuator dynamics at near-zero implementation cost. Use as a comparison
  arm against BAM-modeled actuators (A1 vs A5 vs A1+A5).

## Adaptation-method landscape (context for the C4 deferral)

- RMA (Kumar 2021) two-stage → largely superseded.
- DreamWaQ (VAE latent, 2023), HIM (contrastive, 2023/24), CTS (concurrent
  regression, RA-L 2024 — see `cts-teacher-student.md`), ROA.
- Counter-signal: the strongest 2025 flat-ground humanoid results (FastSAC
  recipe, BeyondMimic-adjacent velocity policies) ship with **no adaptation
  module** — asymmetric critic + DR suffices for flat walking. Hence C4 is
  gated on hardware evidence, not built pre-emptively.
