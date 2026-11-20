# K1 ball skills: benchmarks of the current best set

## Real-robot gains set (2026-11-20) — use this one

All six retrained on the real robot's PD gains (legs kp 80 / kd 4, ankles 50 / 2,
arms 10 / 1, head 4 / 0.25: the same as the old walk, so no gain switch at hand-overs).
The receiving skills train on passes up to 8 m, missing the robot by the same angle as
before (about 6.4 deg), with the camera's range raised to 9 m (check this against the
real detector). Several fine-tunes collapsed at the end of their run, so each policy
below is the checkpoint that tested best, not always the last one; use exactly the
file named. Every picked checkpoint has its own `model_<n>.onnx` next to it
(`scripts/dribbling/export_onnx.py`, checked identical to training's export).
The low-gain set is archived in `~/k1_policy_archive/2026-11-19_low_gains/`.

| Skill | File |
|---|---|
| Receive | `k1_receive_walk_v5_stable/model_19500` (candidate: `k1_receive_walk_v6_stable/model_20000`) |
| Dribble | `k1_dribble_v13b_arms_low/model_33700` (arms kept down at low speed; previous: `k1_dribble_v11_handover/model_33500`) |
| Kick (ground) | `k1_kick_v10_handover/model_40493` |
| Air shot | `k1_kick_air_v6_handover` (its only checkpoint) |
| Directional first touch | `k1_first_touch_v5_aim/model_26999` (aim fix, `MJLAB_K1_TOUCH_AIM=1`) |
| First-time pass | `k1_first_touch_pass_v6_long_more/model_34290` |

| Skill | Measure | Low-gain set | Real-gains set |
|---|---|---|---|
| Receive | Trapped and held, walking / standing | 96 / 98% | 97 / 98% (v6: 98 / 99%) |
| | Trapped on 5-8 m passes | not trained | 57-61% (v6: 65%) |
| | Falls, usual / 5-8 m passes | 0% / - | 0% / 1% |
| Dribble | Ball speed at 1.0 / 1.3 / 1.8 m/s | 1.02 / 1.21 / 1.43 | 1.08 / 1.30 / 1.54, no falls (v11: 1.10 / 1.29 / 1.52) |
| | Figure of eight 0.8 / 0.6 / 0.5 / 0.4 m | 92 / 85 / 72 / - % | 100 / 92 / 83 / 66% (v11 same run: 95 / 82 / 71 / 53%) |
| | Elbow bend at 0.3 m/s, pitch / yaw from arms-down | - | 0.12 / 0.64 rad (v11: 0.65 / 1.18) |
| Kick | Aim error (median) | 1.4-1.8 deg | 1.3-1.5 deg |
| | Head over the motor's 7.85 rad/s rating | 19-22% | 7% (pass speeds) |
| | Exit speed / commanded | 1.05-1.14x | 1.08-1.16x |
| Air shot | Aim error (median) | about 1 deg | 0.8-0.9 deg |
| | Peak height, passes / shots | 0.4 / 1.1 m | 0.38 / 1.14 m |
| Directional first touch | Median off line / within 20 deg | 8.7 deg / 85% | 9.1 deg / 85% (v4: 11.8 deg / 72%) |
| | Touched on 5-8 m passes | not trained | 72% (v4: 78%) |
| First-time pass | Median off line / within 20 deg | 3.7 deg / 95% | 4.0 deg / 94% |
| | Touched on 5-8 m passes | not trained | 78% |

Hand-overs (real-gains set):

| Hand-over | Result |
|---|---|
| Walk -> first-time pass (old walk) | 1% fell, median 4.1 deg, 94% within 20 deg |
| Walk -> directional first touch (old walk) | 0% fell, median 9.2 deg, 81% within 20 deg (v4: 12.0 deg, 67%) |
| Walk up to a still ball -> dribble (old walk) | 0.9-1.0% fell, 98-99% kept the ball 6 s, 0.83 m/s vs 0.8 |
| Dribble -> walk | 0.2% fell (stand, walk, walk and turn, turn on the spot) |
| Receive -> dribble | 99% trapped; then 0% fell, 100% kept the ball 8 s, 0.97 m/s vs 1.0 |
| Directional first touch -> dribble | 1% fell, 97% kept the ball 6 s, 0.84 m/s vs 0.8 |
| Dribble -> kick (dribble v13b states; kick trained on v11's) | 99% struck, 1% fell |

Trained hand-overs: walk -> receive / first touch / pass (walk states); receive and
first touch -> dribble (half the dribble's episodes start from their hand-over states,
`MJLAB_K1_DRIBBLE_BANK`); dribble, receive and first touch -> kick and air shot
(60 / 25 / 15% of starts, `MJLAB_K1_KICK_BANK`). Not trained: kick or dribble -> walk
(the walk is fixed). Walk up to a still ball -> dribble was trained in
`k1_dribble_v12_walk_approach/model_33800` (0.2-0.5% fell, 100% kept, faster: 1.17 / 1.35 /
1.46 / 1.58 m/s) but it lost tight turns (figure of eight 93 / 64 / 61 / 20%), so v11 stays.

Weak points: the directional first touch is fixed by the aim fix (direction rewarded
relative to the commanded speed, plus again 0.2 s after the touch; it was still improving
at the end of its 1,500 iterations); long
passes (the robot reaches 5-8 m passes only 50-80% of the time). Fine-tunes must use
`--agent.algorithm.entropy-coef 0.002`: with 0.01 the action noise grows on a harder
task until the robot falls (receive and first touch collapsed that way).

---

## Low-gain set (2026-11-19, superseded)

Collected 2026-11-19 from the test runs of 2026-11-18 (all in simulation, mjlab / MuJoCo,
corrected foot-ball contact, calibrated camera, motor torque-speed limits). Each line names
the script that measures it, under `scripts/dribbling/`. Sample sizes are simulated attempts.
Nothing here has run on the real robot.

## Current best set

| Skill | Policy folder |
|---|---|
| Walk | `k1_walk_booster_mjlab` (Vini, own gains); candidate `k1_walk_booster_mjlab_v2_our_gains` |
| Receive (standing or walking) | `k1_receive_walk_v3_contact` |
| Dribble | `k1_dribble_v8_contact` |
| Kick (ground) | `k1_kick_v7_head` |
| Air shot | `k1_kick_air_v4_head` |
| Directional first touch | `k1_first_touch_v2` |
| First-time pass | `k1_first_touch_pass_v4_hold` |

## Each policy on its own

| Skill | Measure | Result |
|---|---|---|
| Receive | Trapped and held, standing / walking | 98% / 96%, 0% falls |
| | First touch with the inside of the foot | 99% |
| Dribble | Ball speed at 1.0 / 1.3 / 1.5 / 1.8 m/s commands | 1.02 / 1.21 / 1.31 / 1.43 m/s, no falls or lost balls |
| | Figure of eight at 0.6 m/s, circle 1.5 / 1.0 / 0.8 / 0.6 / 0.5 m | 100 / 98 / 92 / 85 / 72% |
| Kick (ground) | Aim error (median, every angle, pass and shot speeds) | 1.4–1.8° |
| | Exit speed / commanded | 1.05–1.14× (measured on v5; v6 and v7 kept it) |
| | Ball speed / striking-foot speed | about 1.1–1.2× (realistic) |
| | Body turn from strike to end of episode | 6° |
| | From mid-dribble (v8 dribble states): strike | 100% |
| | Head joint over the motor's 7.85 rad/s rating | 19% of pass-speed kicks, 22% of shots |
| Air shot | Aim error (median) | about 1° |
| | Peak height, passes / shots | 0.4 / 1.1 m |
| | Exit speed / commanded, passes / shots | 1.11 / 0.97 (measured on v3) |
| | Body turn from strike to end | 10° (measured on v3) |
| | Head joint over the motor rating | 0–1% |
| Directional first touch | Off line (median) / within 20° | 8.7° / 85% |
| | Pace / commanded | 0.98× |
| | Ball within 1 m of the robot 1.5 s after the touch | 97% |
| First-time pass | Off line (median) / within 20° | 3.7° / 95% |
| | Wide passes (60–90°) within 20° | 91% |
| | Pace / commanded | 0.98× |
| | Body turn from touch to end | 11° |
| | Ball / foot speed at the touch | about 1.6×, ball pressed about 14 mm into the foot (measured on v2; check first on hardware) |

## Hand-overs (latest version of every policy)

| Hand-over | Attempts | Result |
|---|---|---|
| Receive → dribble | 966 | 0% fell, 100% kept the ball for 8 s, 1.02 m/s against 1.0 commanded |
| Walk → receive (old walk, with the gain switch) | 30 | 30 trapped, no falls |
| Receive → dribble → shoot, ground kick | 40 | 40 goals, no falls |
| Receive → dribble → shoot, air shot | 40 | 36 goals, no falls (misses likely over the demo's 0.6 m crossbar) |
| Walk → first-time pass (old walk) | 512 | 0% fell, median 4.9° off line, 87% within 20° |
| Directional first touch → dribble | 512 | 3% fell, 95% kept the ball, 0.83 m/s against 0.8 commanded |
| Walk → directional first touch (old walk) | 512 | 0% fell, median 9.6° off line, 79% within 20° |

## Weak points

1. **Directional first touch** is the least accurate link: 85% within 20° on its own, 79% after a walk.
   A retune with a tighter direction reward (`k1_first_touch_v3`) made it worse (median 16.5°,
   pace 0.75×) and is not used. Ideas not yet tried: a curriculum from small to large direction
   changes, or rewarding the ball's direction later in the touch window.
2. **Ground kick head speed:** about 20% of kicks exceed the head motor's rating; a stronger
   penalty (`k1_kick_v8_head2`) didn't help and is not used. Limit the head's target rate in
   the controller instead (see CONTROLLER_INTERFACE.md).
3. **Walk → first-time pass** drops from 95% to 87% within 20°. Not retuned because it depends on
   the walk, which is being replaced.

## Walk command tracking in this repo's sim (2026-11-19)

256 robots, each command held 8 s, velocity averaged over the last 5 s
(`check_walk_tracking.py`). No falls with either walk.

| Command (vx, vy, wz) | Old walk, own gains | New walk (v2), ball-skill gains |
|---|---|---|
| (0.3, 0, 0) | 0.17 forward | 0.26 forward |
| (0.6, 0, 0) | 0.52 | 0.65 |
| (0.9, 0, 0) | 0.86 | 0.88 |
| (0, ±0.3, 0) sideways | 0.00 sideways (turns instead) | +0.10 / −0.04 sideways |
| (0, 0, ±0.8) turn on the spot | ±0.20 rad/s | ±0.02 rad/s |
| (0.5, 0, 0.5) walk and turn | 0.42 m/s, 0.38 rad/s | 0.53 m/s, 0.46 rad/s |

### Walk → first-touch hand-overs, old walk against new (2026-11-19)

512 robots each, walking 0.3–0.9 m/s when the pass is played (`check_handover_walk_skill.py`,
walk chosen with the `WALK_ONNX` environment variable). With the new walk the gain switch
does nothing, since its gains are the ball skills' gains.

| Hand-over | Old walk | New walk (v2) |
|---|---|---|
| Walk → directional first touch | 0% fell, median 9.3°, 78% within 20°, pace 1.01× | 0% fell, median 9.5°, 77% within 20°, pace 0.99× |
| Walk → first-time pass | 0% fell, median 5.0°, 88% within 20°, pace 0.96× | 0% fell, median 5.3°, 82% within 20°, pace 0.92× |
