# FastSAC/FastTD3 — "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes"

**Paper:** https://arxiv.org/abs/2512.01996 (Unitree G1, Booster T1)
**Relevance to this plan:** NOT the algorithm (we stay on PPO) — the recipe
*shape*: minimal reward stack, constant difficulty, symmetry augmentation,
and the γ ablation. Cited by tracks B2, C2, C3.

## Recipe facts

- Off-policy (SAC/TD3 variants) stabilized at thousands of parallel envs;
  locomotion trains in 15 min on one RTX 4090. Claimed to beat PPO
  wall-clock *under strong DR*.
- **Fewer than 10 reward terms** for locomotion: velocity tracking,
  foot-height (swing) tracking, default-pose penalty, foot
  parallel/no-crossing penalties, alive bonus, torso orientation,
  action-rate, **symmetry augmentation**. Weights ramp with episode length;
  no staged reward rewrites.
- Termination: torso/non-foot ground contact (like our fell_over).
- DR: pushes every 1–3 s (strong) or 5–10 s, action delay, PD gains, mass,
  friction, CoM, mixed flat+rough terrain. (Ranges deliberately not
  published.)
- Key tunings that made off-policy work at scale: average (not min) of twin
  Q-values, layer norm everywhere, distributional critic (C51), max policy
  std 1.0, lr 3e-4, β₂=0.95, batch up to 8k.

## Ablations we borrow

- **γ = 0.97 beat 0.99 for velocity-tracking locomotion** (0.99 better for
  whole-body tracking) → C3 cell.
- Layer norm essential for stability at scale (off-policy context; weak
  evidence for our PPO, not adopted).
- More parallel envs → better; simulation throughput was the bottleneck.

## What we explicitly do NOT adopt (and why)

- The off-policy stack: young, big infra change, our PPO+cluster works.
  Revisit only if iteration speed becomes the binding constraint after
  the physics fixes.
