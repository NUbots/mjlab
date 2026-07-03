# Track C — Policy & learning

C1 (critic DR obs) is part of Phase 0. This doc covers the rest. Run after a
stable Track-B base exists (v17/v18 winner), so gains are attributable.

---

## C2 — Mirror (symmetry) augmentation

**Why:** no symmetry mechanism exists (the `limb_symmetry` reward is wired
but disabled with a "debugging" comment); modern recipes include symmetry
augmentation (FastSAC recipe lists it; equivariant-policy work shows
asymmetric-gait pathologies without it — see
`references/reward-curriculum-symmetry.md`). Directly targets gait quality
and left/right turn consistency, which matter for RoboCup.

**Preferred mechanism:** data augmentation (mirrored transitions added to the
PPO batch), NOT re-enabling the `limb_symmetry` reward cost (reward-based
symmetry is the weaker tool). Check the installed rsl-rl version for its
built-in symmetry support (`symmetry_cfg` on the PPO algorithm cfg with
`use_data_augmentation` and a user-supplied mirror function); if present,
implement the mirror functions to its API instead of hand-rolling storage
surgery.

### The mirror map (the bug-prone part — build it carefully)

Joint convention facts from `nugus.xml` (verify signs against the XML before
trusting this table — left hip_roll range is (0, 0.6) while right is
(−0.6, 0), which confirms the flip convention):

| Joint pair (L↔R swap) | Sign |
|---|---|
| hip_yaw | negate |
| hip_roll | negate |
| hip_pitch | keep |
| knee_pitch | keep |
| ankle_pitch | keep |
| ankle_roll | negate |
| shoulder_pitch | keep |
| shoulder_roll | negate |
| elbow_pitch | keep |
| neck_yaw (unpaired) | negate |
| head_pitch (unpaired) | keep |

Non-joint observation channels:

| Obs term | Mirror transform |
|---|---|
| base_ang_vel (gyro x,y,z = roll,pitch,yaw rates) | negate roll rate (x) and yaw rate (z); keep pitch (y) |
| projected_gravity (x,y,z) | negate y |
| command (vx, vy, wz) | negate vy and wz |
| actions (last action) | same map as the action vector |
| gait_clock = [sin 2πφ, cos 2πφ] with foot offsets (0, 0.5) | mirroring swaps feet = φ → φ+0.5 → **negate both channels** (sin(2π(φ+.5)) = −sin, cos likewise) |
| actuator_current (if CURRENT_OBS) | swap L/R pairs, no sign change (magnitudes) |

Critic-only terms: base_lin_vel negate y; foot_height/air_time/contact swap
L/R; foot_contact_forces swap feet AND negate lateral (y) force components;
height_scan mirror the scan grid about the body y-axis (check the raycast
pattern ordering before writing this); the C1 DR-param vector swaps any
per-joint entries L/R.

Action vector: identical map to the joint table (position targets).

### Required unit tests (write these FIRST)

In a new `tests/test_mirror_map.py`:
1. **Involution:** `mirror(mirror(obs)) == obs`, same for actions.
2. **Physics consistency (the real test):** reset env to a randomized state,
   apply action sequence A for N steps, record trajectory; reset to the
   *mirrored* state, apply `mirror(A)`, assert the resulting base twist and
   joint trajectories equal the mirror of the first trajectory to tolerance.
   Run with DR events disabled and pushes off. This catches wrong signs that
   test 1 cannot.
3. Obs-vector index audit: assert the map's permutation indices were built
   from the live obs term ordering (compute from the ObservationManager's
   term slices at runtime, never hardcode offsets — CURRENT_OBS and C1
   change the layout).

**Signal:** fixed-eval falls_per_min at (0,±0.3,0) and (0,0,±0.5) become
left/right symmetric (they currently aren't measured — E0.1 provides this);
gait visually symmetric in the viewer; no reward regression.

**Cost:** medium — the map is fiddly; the tests make it safe.
**Conflicts:** any obs-layout change (CURRENT_OBS, C1) invalidates hardcoded
indices — hence test 3.

---

## C3 — Optimizer hygiene for long runs (cheap piggyback cells)

Motivation: v15 oscillated for 15k iters with adaptive-KL LR reaching 1e-3
and entropy_coef fixed at 0.01 — plausible contributors to never settling.
Config: `src/mjlab/tasks/velocity/config/nugus/rl_cfg.py`.

Cells to piggyback (+1 run each on any Track-B batch):
1. **Entropy decay:** 0.01 → 0.001 linearly over training. rsl-rl has no
   native schedule; implement via a small runner hook or a curriculum term
   writing `alg.entropy_coef` from `common_step_counter` (same pattern as
   reward-weight curriculum, which mutates cfg live).
2. **LR cap late:** after the difficulty ramp (or from iter 2500 in
   hard-from-start), clamp adaptive LR to ≤3e-4 (or switch to fixed 3e-4).
3. **γ = 0.97:** one cell. The FastSAC paper's ablation favors 0.97 over
   0.99 for velocity-tracking locomotion (shorter effective horizon
   stabilizes value learning); evidence is off-policy but cheap to test.

**Signal:** late-training fixed-eval variance shrinks; no peak-performance
loss. **Cost:** trivial.

---

## C4 — DEFERRED: observation history / concurrent teacher-student (CTS)

Do NOT build until (a) a policy has been deployed (D2) and (b) hardware
failures look like dynamics misidentification the policy should adapt to
online (e.g. works with fresh batteries, fails at low voltage; works on one
robot, fails on another).

When triggered, implement in this order:
1. **5-frame actor obs history** (stack last 5 actor obs). Cheap, breaks
   checkpoint compat (actor input dim ×5), needed as the student encoder
   input for CTS anyway. Test on its own first — history alone sometimes
   captures enough implicit adaptation.
2. **CTS** per `references/cts-teacher-student.md`: shared actor/critic,
   privileged encoder (input: the C1 DR-param vector + foot states) vs
   proprioceptive encoder (input: 5-frame history), 3:1 teacher:student env
   split, student encoder regression loss to teacher latent (32-dim,
   L2-normalized), both groups trained with PPO. This is runner/alg surgery
   in rsl-rl — the single most expensive item in this plan; that is why it
   is gated on hardware evidence.

**Honest uncertainty:** CTS's gains were demonstrated mostly on quadrupeds
and rough terrain. For flat-ground kid-size walking, the strongest 2025
results ship without any adaptation module.
