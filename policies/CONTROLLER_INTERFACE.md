# K1 soccer skill policies: controller interface

For whoever builds the controller (a window or gamepad where a person sets the inputs, a
behaviour layer, or another program) that drives these policies. Updated 2026-11-18 17:45.
Everything here describes the policies as trained and measured in simulation (mjlab /
MuJoCo, with motor torque-speed limits). None of the ball skills has run on the real robot.
The walk policy is a teammate's and is reported to work on the real robot.

## 1. The policies

All files are under `~/mjlab/policies/`. Each folder has an `.onnx` (for deployment) and a
`.pt` (for the training code). **Only the `.onnx` is needed by a controller.** All ball skills
below were trained on the corrected foot-ball contact (2026-11-18), the calibrated camera and
motor torque-speed limits.

| Skill | Folder / ONNX file | Input | Command | Gains |
|---|---|---|---|---|
| Walk | `k1_walk_booster_mjlab/booster_mjlab_walk.onnx` | 75 numbers, no history | Body twist `(vx, vy, wz)` | Its own (section 4) |
| Receive (standing or walking) | `k1_receive_walk_v3_contact/2026-11-18_16-27-04.onnx` | 234 numbers | Always `(0, 0)` | Ball-skill set (section 4) |
| Dribble | `k1_dribble_v8_contact/2026-11-18_15-24-59.onnx` | 234 numbers | Ball velocity `(vx, vy)` | Ball-skill set |
| Kick (ground) | `k1_kick_v7_head/2026-11-18_23-49-54.onnx` | 234 numbers | Ball exit velocity `(vx, vy)` | Ball-skill set |
| Air shot | `k1_kick_air_v4_head/2026-11-19_00-57-57.onnx` | 234 numbers | Ball exit velocity `(vx, vy)` (ground part; the ball also rises at 30°) | Ball-skill set |
| Directional first touch | `k1_first_touch_v2/2026-11-18_13-22-47.onnx` | 234 numbers | Ball velocity wanted off the first touch `(vx, vy)` | Ball-skill set |
| First-time pass | `k1_first_touch_pass_v4_hold/2026-11-18_22-38-15.onnx` | 234 numbers | Ball velocity wanted off the first touch `(vx, vy)` | Ball-skill set |

Every ball skill has the same input and output layout, so swapping one for another is a file
and command change only.

## 2. What the person (or controller) supplies

| Skill | Input | Limits (trained ranges) | Measured (simulation) |
|---|---|---|---|
| Walk | `vx` forward, `vy` left (m/s), `wz` turn left (rad/s), robot frame | Reported by its author up to about 1 m/s forward. **In our simulation it only responds to forward speed** (sideways and turning barely move it). | Ask its author for its trained ranges before exposing sideways and turning. |
| Receive | Nothing (command `(0, 0)`) | Pass from in front: up to about 4 m, within ±48° of the heading, arriving at up to 3.5 m/s, up to about 45 cm to the side | Trapped and held 98% from standing, 96% from walking; 0% falls; first touch 99% inside of the foot |
| Dribble | Desired **ball** velocity, m/s | 0.4–1.8 m/s any direction, or `(0, 0)` to hold; may change or curve at any time | Ball speed reaches 1.02 / 1.21 / 1.31 / 1.43 m/s at 1.0 / 1.3 / 1.5 / 1.8 m/s commands, no falls. Figure of eight at 0.6 m/s: 98% of 1.0 m circles, 92% of 0.8 m, 85% of 0.6 m. |
| Kick (ground) | Desired **ball** exit velocity, one command per kick, held until the ball has gone | Passes 2–4.5 m/s within ±135° of heading; shots 4.5–7 m/s within ±45° | Aim 1.5–2° at every angle; speed 1.05–1.14× command; from mid-dribble 99% strike, 0% falls |
| Air shot | As the kick | As the kick | Rises at about 30°: 0.34 m peak at passes, 0.85 m at shots; aim about 1° |
| Directional first touch | Ball velocity wanted off the first touch | 0.5–1.2 m/s, within ±90° of the direction the pass came from (sideways or back towards the passer) | Median 9° off line, 85% within 20°; pace 0.98×; ball within 1 m of the robot 1.5 s later 97% (ready for the dribble) |
| First-time pass | Ball velocity wanted off the first touch | 2–4 m/s, within ±90° of the direction the pass came from | Median 4° off line, 98% within 20°; pace 0.94–0.96× |

### Frames

- **Ball-skill command `(vx, vy)`** is in the robot's **yaw frame**: x forward, y left, rotated
  by the robot's heading only (not pitch or roll). A controller working in field coordinates
  must rotate it by minus the robot's heading every step.
- **Walk command `(vx, vy, wz)`** is in the robot's body frame.

### Hard limits to enforce in the controller

| Quantity | Clamp to |
|---|---|
| Dribble ball speed | 0 or 0.4–1.8 m/s |
| Kick speed | 2.0–7.0 m/s |
| Kick direction | Within ±135° of heading (±45° above 4.5 m/s) |
| First-touch direction | Within ±90° of the direction the pass came from |
| Walk | 0–1.0 m/s forward until its sideways and turning ranges are confirmed |

## 3. Switching between skills

Every ball skill (receive, dribble, kick, air shot, first touch, first-time pass) shares one input and output layout, so switching
between them means changing which network runs and what the command is. **Keep the
observation history, including the previous actions, running across the switch.** Walk is
different (section 4): switching to or from walk also changes the PD gains, the default pose,
the action scale and the input layout.

| From → to | Trained for it? | Tested? | How to switch |
|---|---|---|---|
| Walk → receive | Yes: receive fine-tuned from mid-walk states | Yes, 8 of 10 trapped, 0 falls, including the gain change | When a pass is on its way within about 3 m. Switch policy and gains in the same step. |
| Receive → dribble | No, not needed | Yes, 966 robots: 0% fell, 99% kept the ball over 8 s | After the ball has been held still in front (slower than 0.3 m/s, within about 0.2 m, for 0.5 s). Switching while the ball still moves is untested. |
| Dribble → kick | Yes: kick fine-tuned from mid-dribble states of this dribble (75% of training) | Yes. From mid-dribble: 96% strike, 0% falls. Receive → dribble → shoot demo: 16 goals in 20 with the previous kick. | Switch and give the kick command in the same step. |
| Standing → kick | Yes (25% of training) | Yes, 99% strike, 0% falls | Ball 0.3–0.8 m ahead, within about ±23°. |
| Walk → first touch / first-time pass | Yes: trained from mid-walk states (75%) | Measured in the policy tests above, not yet with the gain switch | As walk → receive, giving the wanted ball velocity as the command. |
| Directional first touch → dribble | Partly: the touch is rewarded for staying with the ball | Not yet run as a hand-over; the ball is within 1 m of the robot 1.5 s after the touch 97% of the time | Switch once the touch window is over (about 0.4 s after the first touch). |
| Dribble → air shot | Yes, as for the kick | Strike tests only | As dribble → kick. |
| Kick → anything | No | No | After the kick the robot stands still and does not chase. Any hand-over afterwards is untested. |
| Dribble → walk, walk → dribble, receive → walk | No | No | Untested. |

Reference implementations of the switching, in Python:
- `scripts/dribbling/demo_walk_receive.py` (class `Gains` and `Operator`): walk → receive with the gain switch.
- `scripts/dribbling/demo_receive_dribble_shoot.py` (class `Operator`): receive → dribble → kick.
- `scripts/dribbling/walk_onnx.py` (class `WalkPolicy`): building the walk policy's input and converting its output.

## 4. Low-level details

Control rate **50 Hz** (one network evaluation every 20 ms) for every policy.

### Joint order (all policies)

All five use the same 22 joints in the same order. The walk ONNX calls the head joints
`Head_Yaw` / `Head_Pitch`; the ball policies call them `AAHead_yaw` / `Head_pitch`. Same joints.

| # | Joint | # | Joint |
|---|---|---|---|
| 1 | Head yaw | 12 | Left_Hip_Roll |
| 2 | Head pitch | 13 | Left_Hip_Yaw |
| 3 | Left_Shoulder_Pitch | 14 | Left_Knee_Pitch |
| 4 | Left_Shoulder_Roll | 15 | Left_Ankle_Pitch |
| 5 | Left_Elbow_Pitch | 16 | Left_Ankle_Roll |
| 6 | Left_Elbow_Yaw | 17 | Right_Hip_Pitch |
| 7 | Right_Shoulder_Pitch | 18 | Right_Hip_Roll |
| 8 | Right_Shoulder_Roll | 19 | Right_Hip_Yaw |
| 9 | Right_Elbow_Pitch | 20 | Right_Knee_Pitch |
| 10 | Right_Elbow_Yaw | 21 | Right_Ankle_Pitch |
| 11 | Left_Hip_Pitch | 22 | Right_Ankle_Roll |

### Output (all policies)

22 numbers. `target_angle = default_angle + action_scale × action`, sent to a PD position
controller with that policy's gains. Not clipped. **Default angles, action scales and gains
differ between the walk policy and the ball skills:**

| Joint group | Ball skills: default / scale / kp / kd | Walk: default / scale / kp / kd |
|---|---|---|
| Head yaw | 0.0 / 0.380 / 4 / 0.25 | 0.0 / 0.375 / 4 / 0.25 |
| Head pitch | 0.5 / 0.380 / 4 / 0.25 | 0.0 / 0.375 / 4 / 0.25 |
| Shoulder pitch | 0.0 / 0.887 / 10 / 1 | 0.0 / 0.350 / 10 / 1 |
| Shoulder roll (left, right) | −1.3, 1.3 / 0.887 / 10 / 1 | −1.4, 1.4 / 0.350 / 10 / 1 |
| Elbow pitch | 0.0 / 0.887 / 10 / 1 | 0.0 / 0.350 / 10 / 1 |
| Elbow yaw (left, right) | −0.15, 0.15 / 0.887 / 10 / 1 | −0.4, 0.4 / 0.350 / 10 / 1 |
| Hip pitch | −0.2 / 0.563 / 80 / 4 | −0.4 / 0.213 / 80 / 4 |
| Hip roll | 0.0 / 0.886 / 80 / 4 | 0.0 / 0.237 / 80 / 4 |
| Hip yaw | 0.0 / 0.537 / 80 / 4 | 0.0 / 0.120 / 80 / 4 |
| Knee | 0.4 / 0.464 / 80 / 4 | 0.8 / 0.350 / 80 / 4 |
| Ankle pitch | −0.2 / 0.268 / 50 / 2 | −0.4 / 0.191 / 50 / 2 |
| Ankle roll | 0.0 / 0.268 / 50 / 2 | 0.0 / 0.191 / 50 / 2 |

Ball-skill gains are the real robot's (2026-11-20 real-gains set; the earlier low-gain set
used Booster's Isaac Lab values). The old walk uses the same gains, so no gain switch is
needed at hand-overs. Every ONNX file also stores its own
joint names, defaults, action scales and gains as metadata (`default_joint_pos`,
`action_scale`, `joint_stiffness`, `joint_damping`), so a controller can read them rather
than hard-coding this table.

**ONNX files exported before 2026-11-21 10:00 from the real-gains set carry kp 1 / kd 0
for every joint** (the motor model applies the PD law in software, and the exporter read
MuJoCo's placeholder gains). The picked policies have been re-exported with the right gains
(legs 80 / 4, ankles 50 / 2, arms 10 / 1, head 4 / 0.25) as `model_<n>.onnx`; the old
training exports are renamed `*.WRONG_GAIN_METADATA.onnx`. Files now also list
`action_joint_names`: the joints the outputs drive, in order (20 for our walk, which leaves
the head alone; 22 for the ball skills).

All policies, including walk, drive the head joints. The controller must not command the head
separately.

### Walk policy input: 75 numbers, current step only

In order: IMU angular velocity, body frame (3); gravity direction in the body frame (3);
joint angles minus **the walk policy's** defaults (22); joint velocities (22); the walk
policy's own previous raw output (22); twist command `(vx, vy, wz)` (3).

### Ball-skill input: 234 numbers

Seven terms, each the last 3 steps **oldest first**, concatenated in this order. On start-up
fill the history by repeating the first observation.

| # | Term | Per step | × 3 | Definition |
|---|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 9 | IMU angular velocity, body frame, rad/s |
| 2 | `projected_gravity` | 3 | 9 | Unit gravity vector in the body frame, about `(0, 0, −1)` upright |
| 3 | `joint_pos` | 22 | 66 | Joint angle minus **the ball-skill** default, rad |
| 4 | `joint_vel` | 22 | 66 | Joint velocity, rad/s |
| 5 | `actions` | 22 | 66 | The previous raw output |
| 6 | `command` | 2 | 6 | Commanded ball velocity, yaw frame, m/s |
| 7 | `ball` | 4 | 12 | `(x, y, z, visible)` from the head camera |

The ball term is the ball centre in the head camera frame (x forward along the camera, y left,
z up, metres) and a visible flag. With no detection this step, repeat the last seen
`(x, y, z)` with `visible = 0`; before the first detection use `(0.5, 0, −0.8, 0)`. The field
of view is the K1 head camera's calibration (544 × 448 image, 106.2° × 95.0°), modelled as a
symmetric window; range 5 m, noise 1 cm + 3% of range, 3% of detections dropped. **Policies
dated before 2026-11-17 19:15 were trained with a 90° × 60° placeholder** (section 6). The
camera's mounting position and tilt on the head are still placeholders.

### Switching to or from walk

When the walk policy hands over to a ball skill (or back), in the same control step:
1. Change the PD gains (table above).
2. Change the default pose and action scale used to turn the network's output into targets.
3. Start the new policy's input from the current robot state. The ball skills need a 3-step
   history; fill it by repeating the current step. Their "previous action" entries should be
   the current joint targets expressed in the ball-skill convention:
   `(target − ball_default) / ball_scale`.

`demo_walk_receive.py` does steps 1 and 2. For step 3 it simply lets the receive policy's
history start fresh; it worked in testing.

## 5. Measured behaviour

See the "Measured" column in section 2. All measured on 1,024 simulated attempts per figure,
with the corrected contact.

## 6. Known problems

- **Foot-ball contact was fixed on 2026-11-18.** Before that the simulated contact added
  energy (a ball could leave at 3.5x the foot's speed), and every kicking policy exploited
  it. All policies in section 1 were retrained on the fix; ball speed is now about 1.1-1.2x
  the striking foot's speed, as in a real kick. **Do not use any policy dated earlier for
  kicking or passing.**
- **The first-time pass presses the ball 14 mm into the foot and ground** at the touch
  (median); the ball leaves at about 1.6x the foot's speed, which a moving incoming ball can
  explain, but it is the policy to watch first on hardware.
- **Walk sideways and turning** barely respond in our simulation (section 2).
- **Gain change at walk <-> ball skills** works in simulation; untested on hardware.
- **Head speed:** the ground kick still exceeds the head motor's 7.85 rad/s rating in about
  20% of kicks (training could not remove it). **Rate-limit the head targets on the robot.**
  The air shot and the other skills stay within it.
- **Camera:** field of view from calibration; the camera's mounting position and tilt on the
  head are still guesses.
- **Training disturbances were mild:** pushes up to 0.5 m/s every 3-10 s, flat ground, one
  ball weight. No stress testing yet.
- **No policy handles a fall** or getting up.
