# Small / cheap-servo humanoid precedents

The three data points closest to NUgus's platform class (kid-size, Dynamixel
serial-bus servos). Net lesson: **actuator fidelity + latency modeling
decide transfer on this hardware class; reward design does not.**

## 1. DeepMind OP3 soccer — the success case

**Paper:** "Learning agile soccer skills for a bipedal robot with deep RL",
Science Robotics 2024 — https://www.science.org/doi/10.1126/scirobotics.adi8022
**Platform:** Robotis OP3 — 51 cm, 3.5 kg, 20× Dynamixel XM430 — the closest
published relative of NUgus.

- Transferred full soccer behaviors (walk, turn, kick, get-up) zero-shot.
- What they credited: system-identified position-controlled servo model
  (torque limits + gains), **explicit latency modeling**, moderate DR
  (mass, friction, orientation offset), perturbations during training, and
  regularization for smooth actions.
- They did NOT need adaptation modules or motion priors for transfer.

## 2. FRASA (Rhoban Sigmaban) — kid-size RL on hardware

**Paper:** https://arxiv.org/pdf/2410.08655 (fall recovery + stand-up,
CrossQ algorithm, Sigmaban kid-size with Dynamixel MX servos).
Same lab as BAM — their pipeline pairs RL with their identified actuator
models (see `bam-actuator-models.md`). Demonstrates kid-size Dynamixel
transfer is achievable with actuator fidelity as the foundation.

## 3. UToronto Bez thesis — the failure case (cautionary)

**Paper:** "Sim2Real RL for Soccer skills" —
https://arxiv.org/pdf/2512.12437 (B.A.Sc. thesis, 2023, arXiv'd Dec 2025).
Platform: Bez kid-size (MX-28/AX-12). Trained kick/walk/jump with PPO + AMP
+ curriculum in Isaac Gym + Webots; **sim-to-real transfer failed
outright**. The thesis's own diagnosis points at actuator/dynamics mismatch
on low-cost servos. This is what skipping tracks A/E0.2 looks like.

## Context: RoboCup kid-size state of practice

As of 2024–2025 no kid-size team is known to have deployed an RL walk in
competition (CIT Brains won 2024 with classical control on SUSTAINA-OP2;
Bit-Bots won German Open 2025). An RL walk that transfers would be a
competitive differentiator, and nothing in the successful-precedent list
(OP3, FRASA) is beyond NUgus hardware — the gap is actuator modeling
discipline, which is exactly tracks E0.2/A1/A2/A4.
