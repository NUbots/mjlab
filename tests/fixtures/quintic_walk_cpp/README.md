# Golden data for the quintic walk port

Two generators here compile the **real** NUbots C++ and dump its output, so the
Python port is checked against the code it came from rather than against a
reimplementation of the same idea:

| Generator | Fixture | Checked by |
|---|---|---|
| `dump_ik.cpp` | `../quintic_walk_ik_golden.csv` | `tests/test_quintic_walk_kinematics.py` |
| `dump_walk.cpp` | `../quintic_walk_trace_golden.csv` | `tests/test_quintic_walk_generator.py` |

`dump_ik.cpp` sweeps foot poses through `calculate_leg_joints`.
`dump_walk.cpp` runs `WalkGenerator` through four command profiles — steady
forward, a start/stop cycle, an omnidirectional command, and one with
`only_switch_when_planted` enabled — dumping the engine state, phase, clock and
all four poses every control step.

## Regenerating

Needs a C++20 compiler, Eigen, and a checkout of
[NUbots](https://github.com/NUbots/NUbots). From this directory:

```sh
NUBOTS=~/NUbots
COMMON=(-std=c++20 -O2 -I "$NUBOTS/shared" -I shim -I /usr/include/eigen3
        "$NUBOTS/shared/utility/input/ServoID.cpp"
        "$NUBOTS/shared/utility/input/LimbID.cpp")

g++ "${COMMON[@]}" dump_ik.cpp   -o /tmp/dump_ik   && /tmp/dump_ik   > ../quintic_walk_ik_golden.csv
g++ "${COMMON[@]}" dump_walk.cpp -o /tmp/dump_walk && /tmp/dump_walk > ../quintic_walk_trace_golden.csv
```

Regenerate whenever the upstream engine changes, and expect the tolerances
(1e-12) to keep holding: the port is an exact transcription, not an
approximation. Both currently agree to within machine epsilon — 1.7e-14 rad for
the IK, 7.2e-16 for the trace.

## The shims

The NUbots headers include protobuf-generated messages and `<nuclear>`, none of
which the ported code actually reads from. `shim/` provides the smallest
stand-ins that satisfy the compiler:

- `message/actuation/KinematicsModel.hpp` — the leg dimensions the IK uses.
- `message/behaviour/state/WalkState.hpp` — the `State` and `Phase` enums, with
  values matching `WalkState.proto`. `State` mimics the generated wrapper
  struct, which exposes its enumerator through `.value`.
- `message/input/Sensors.hpp`, `nuclear` — empty stubs.

## A note on precision

`KinematicsModel.hpp` declares the leg dimensions as `double`, whereas the real
`.proto` declares them `float`. That difference matters more than it looks:
rounding `0.2` to `float` perturbs `acos` arguments near full leg extension
enough to move the IK solution by up to **3e-6 rad**.

That is physically irrelevant (2e-4 degrees) but far above float64 noise, so it
would swamp a 1e-12 regression test. Generating in `double` isolates the port's
correctness from the message format's precision. Against a `float`-valued model
the port instead agrees to ~3e-6 rad — the agreement the robot itself runs at.
