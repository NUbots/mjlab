# BAM — Better Actuator Models (Rhoban)

**Paper:** "Extended Friction Models for the Physics Simulation of Servo
Actuators" — https://arxiv.org/abs/2410.08650
**Code + identified parameters:** https://github.com/Rhoban/bam
**Relevance:** identified models for the **Dynamixel MX-64 and MX-106 — the
exact servo families on NUgus** (arms/head/neck = MX-64, hip_yaw = MX-106;
legs are XH540-W270, same vendor, not covered — needs A2 bench ID).
Used by track A1/A2.

## Key claims (with numbers)

- The standard Coulomb-viscous friction model materially under-fits cheap
  servo gearboxes. Their extended models (M2–M6: adding Stribeck effect and
  load-dependent friction) reduce trajectory MAE vs Coulomb-viscous by:
  **2.93× (MX-64)**, **2.02× (MX-106)**, 2.34× (eRob80:100),
  1.51× (eRob80:50).
- On 2R-arm validation tasks, simulation error dropped >50% vs the
  Coulomb-viscous baseline.
- Identification protocol: pendulum rig + logged position-target
  trajectories; open-source implementation includes the fitting code and
  MuJoCo integration.

## Getting the parameters (for A1)

Clone the repo; identified parameter sets are stored per-actuator/per-model
in the repo (JSON). Extract for MX-64 and MX-106:

- Coulomb friction term → map to MuJoCo `dof_frictionloss`
- viscous term → passive joint `damping` (NOT the PD kd gain)
- rotor/gearbox inertia → `armature` (compare to current NUgus values
  0.0266 / 0.01195; prefer BAM's if they differ)
- Stribeck / load-dependent terms (M5/M6) → cannot be expressed as static
  MuJoCo dof fields; approximate by *randomizing* frictionloss over the
  range those terms span at representative gait loads (see plan doc
  02, A1 step 4)

## Caveats

- Rhoban's units, voltage, and firmware mode ≠ ours. Treat as starting
  point; A2 (bench ID of our servos) replaces it. The systematic-sim2real
  study (see its reference file) found firmware/compensation mode changes
  effective dynamics — identify and deploy in the same servo control mode.
- XH540-W270 params don't exist upstream: provisional start = MX-106 values
  scaled by gear-ratio ratio (270.4:1 vs 225:1), flagged PROVISIONAL in code.
