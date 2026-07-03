# Systematic sim-to-real transfer for diverse legged robots (PACE)

**Paper:** "Towards bridging the gap: Systematic sim-to-real transfer for
diverse legged robots" — https://arxiv.org/html/2509.06342v1
**Relevance:** the strongest recent evidence that *identification beats blind
randomization* on the actuator side; source of concrete parameter/delay
guidance used in tracks A1/A2/A4 and the joule-weight recommendation.

## Core result

Zero-shot transfer across 13+ platforms (ANYmal, Tytan, minimal quadruped…)
**without any dynamics randomization**, by identifying per joint:

1. armature inertia `I_a`
2. viscous damping `d`
3. Coulomb friction `τ_f`
4. one global lumped control-loop delay `T_d`

Transfer function per joint:
`H_q(s) = e^(−s·T_d) · P_τ / (I_a·s² + (d + D_τ)·s + P_τ)`

## Numbers worth remembering

- **Identified `T_d` ≈ 7.5 ms** on two very different platforms — ~20×
  larger than the motor-level dead time (≈400 µs); the delay lives in
  electronics/firmware/mechanics, not the motor. Expect Dynamixel serial
  chains to be worse → measure (track A4), don't assume.
- Identified armature came out 4–6× the rotor-alone value (gearbox +
  compensation effects absorb into it).
- The fitted model generalized across unseen PD gain settings (±14%
  parameter consistency) — i.e. it's physical, not overfit.
- URDF-only baseline (no actuator ID) failed at 1 m/s on ANYmal; the
  identified model succeeded.
- **Firmware modes matter:** with firmware compensations enabled a "virtual
  inertia" offset (≈8.1e-3 kg·m²) appeared → *identify and deploy in
  identical firmware/control modes* (for Dynamixels: same control mode,
  return-delay-time, voltage).
- Identification: CMA-ES over 4096 parallel sims, 10–24 h on one RTX 3080;
  excitation = 20–60 s chirps; in-air identification transferred to contact
  tasks.
- **Energy penalty weight guidance: ~1e-5** (their reward tables) —
  independently matches our v13 finding that joule 3e-4 over-penalizes.

## How this plan uses it

- Safe synthesis adopted (their claim is the aggressive end): *identify,
  then randomize modestly (±15–25%) around identified values* rather than
  either extreme.
- Justifies prioritizing friction/armature/delay fidelity (E0.2, A1, A2, A4)
  over piling on more randomization.
- Their alternative comparison: PACE matched an LSTM actuator-network with
  ~20 s of data vs ~4 h — if we ever consider actuator nets, do this
  instead.
