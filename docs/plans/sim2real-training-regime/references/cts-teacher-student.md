# CTS — Concurrent Teacher-Student RL (deferred track C4)

**Paper:** "CTS: Concurrent Teacher-Student Reinforcement Learning for
Legged Locomotion", IEEE RA-L vol. 9 no. 11, 2024 —
https://arxiv.org/html/2405.10830v2
**Relevance:** the current best-evidenced successor to two-stage
RMA/teacher-student if adaptation is ever needed (gated on hardware evidence
per plan doc 04, C4). Do not build pre-emptively.

## Idea

Train teacher and student in ONE stage: shared policy network πθ and critic,
two encoders. Teacher envs feed a privileged encoder (terrain, contacts,
torques, dynamics params); student envs feed a proprioceptive encoder over an
observation history. Student encoder is trained both by PPO gradient AND a
regression loss to match the teacher's latent.

## Results vs alternatives

- vs two-stage teacher-student: velocity-tracking error −17.85% (slopes),
  −19.12% (rough slopes), −21.85% (discrete obstacles); survival under
  pushes +5–6%.
- vs ROA: similar single-stage idea, but CTS actually trains the policy
  under the student encoder's input with RL (ROA doesn't).
- Related family: DreamWaQ (VAE latent), HIM (contrastive); CTS uses plain
  regression — simplest to implement.
- Caveat for us: demonstrated on quadrupeds (and some biped work followed);
  flat-ground humanoid evidence thin. Strong 2025 flat-walking results ship
  with NO adaptation module.

## Reproduction constants

| Param | Value |
|---|---|
| History length H | 5 frames |
| Encoders | MLP [512, 256], ELU |
| Latent | 32-dim, L2-normalized |
| Policy | [512, 256, 128] (matches our current net) |
| Teacher:student env ratio | 3:1 (6144:2048) |
| Epochs / γ / λ | 5 / 0.99 / 0.95 (matches ours) |
| DR used | link mass ±20%, friction [0.2,1.7], action delay [0,20] ms |
| Convergence | ~3000 iters, ~105 min on RTX 4090 |

## Mapping to our stack (when triggered)

- Privileged encoder input = the C1 DR-parameter vector + foot states
  (already critic obs after Phase 0).
- Student encoder input = 5-frame actor-obs history (C4 step 1).
- Env-group split: partition envs 3:1 at runner level; both groups share
  rollout storage; add the latent-regression term to the PPO update.
- This is surgery on rsl-rl's OnPolicyRunner/PPO — the most expensive item
  in the plan; hence the gate.
