=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- ``scripts/eval/eval_competence_grid.py`` gained ``--engine quintic``: the
  ported walk engine now runs the same competence grid a policy does, through
  the same collector, shove train, stopping rule and output format. The two
  paths differ in one unavoidable place -- a policy runs inside a
  ``ManagerBasedRlEnv`` that ends and resets episodes for it, an engine does not
  -- so ``WalkEvalHarness`` gained ``reset_idx`` for per-environment resets and
  applies the task's termination rule itself, reproduced from raw state as
  ``competence.episode_end`` against ``competence.FELL_OVER_UPRIGHT``. That bound
  is the task's 50 degrees, deliberately not the pipeline's more generous 60
  degree ``FALL_UPRIGHT_THRESHOLD``, which dates a fall rather than deciding
  whether the episode is one a policy would have been terminated for. The
  distilled policy is not offered: it reads nothing at all, so a disturbance
  axis would measure the plant carrying it, not a controller reacting.
- ``plot_competence_grid.py``'s default curve commands no longer include
  1.0 m/s forward. Both competence-trained policies stop walking and march in
  place above roughly 0.81 m/s commanded -- perfect tracking at 0.80, 0.04 m/s
  delivered at 0.82 -- so that panel put a policy that refused the command
  beside one that attempted it and read as a tracking collapse. A backwards ask
  takes its place. Where each controller's band ends is a real finding, but it
  belongs on the envelope heatmaps, which show the whole plane.
- Added the competence grid: ``scripts/eval/eval_competence_grid.py`` and
  ``scripts/eval/plot_competence_grid.py``, over a new
  ``mjlab.evaluation.competence``. It crosses the commanded velocity with a
  shove magnitude and reports, per cell, a distribution over episodes of the
  five per-episode quantities the training competence tracker accumulates --
  attainment (delivered speed projected on the command), the signed per-axis
  attainments, wobble (the fraction of steps tilted past 25 degrees), whether
  the robot fell, and how much of the episode it survived. The definitions are
  the tracker's, but none of its smoothing is: the EMAs and their pessimistic
  initialisation exist to give a curriculum controller a population signal, and
  read offline they are a filtered statistic still carrying its init. What the
  grid keeps is the layer underneath, disaggregated, so a cell reports
  quartiles rather than a mean -- the interesting cells are the high-variance
  ones, and a mean cannot show that. A command under 0.15 m/s takes no
  attainment sample at all, and those cells are reported undefined rather than
  zero, which would read as the worst sandbagging on the grid.

  This is the first run in the pipeline that needs episodes to *end*: it
  restores the ``fell_over`` termination and the training episode length that
  every other run deliberately removes, via ``build_rl_env(episodic=True)``.
  The shove is the training push driven deterministically -- magnitude pinned
  per cell, heading drawn, four events per episode at fixed onsets -- so
  ``|dv_xy|`` is the cell rather than something binned after the fact.
  Nothing in ``mjlab.evaluation.competence`` reads the environment, so the same
  numbers can be produced for a controller that never sees a
  ``ManagerBasedRlEnv``.
- Added ``mjlab.tasks.velocity.mdp.competence``, the training-side competence
  tracker, gated penalty curriculum and frontier diagnostics, taken unchanged
  from the ``obs-history-competence`` and ``competence-reward-gating`` branches
  (the file is identical in both). The eval takes its metric definitions and
  its push mechanism from here so the two cannot drift. The curriculum terms it
  defines are inert until a config installs them, and none does on this branch:
  the gated penalties want reward terms this branch does not carry.
- Added ``scripts/eval/plot_mocap_profile_pair.py``, a one-off that draws the
  profile strip from two motion-capture logs instead of one. The RL capture had
  to be flown twice -- the first attempt went down during the yaw stage -- and
  between them the two logs hold the twelve-phase schedule exactly once, so the
  figure stitches the head of one onto the rest of the other. The two captures
  are calibrated separately but fitted over one window, measured from the
  longer of them, and the seam between them is drawn heavier than a phase
  boundary and named in the key. Everything but the drawing is imported from
  ``plot_mocap_profile`` rather than copied.
- Added ``scripts/eval/plot_mocap_profile.py``, which draws the velocity-profile
  figure from a motion-capture log of the real robot rather than from a
  simulated run: command against response, with the capture's commanded phases
  laid end to end on one axis the way the simulated figure lays its schedules.
  The rests between commands are cut, and each phase draws only the axes it
  commanded -- on a real robot every axis carries a stride larger than the
  command, so drawing all three everywhere buries the measurement. It reads an
  ``nbs2json`` export directly, taking the command from ``WalkState``, the
  motion from the tracked rigid body, and the IMU to settle which way the
  capture frame points and whether it is mirrored. The profile figure only -- a sweep or
  a grid is hundreds of runs at held commands, which a capture cannot be.

  The capture frame is calibrated from the data and the calibration is printed:
  the floor plane by a robust PCA, its sign by agreeing with the accelerometer
  over samples that are both still and upright, the handedness by correlating
  the captured yaw rate against the robot's gyroscope, and forward from the
  direction of travel under a forward command. The plane is fitted at torso
  height rather than floor height, so a fallen robot leaves it by as much as a
  carried one and the sign cannot be had by counting which side is busier. Velocity is a
  straight-line fit over 1.43 measured strides -- the window at which a
  least-squares slope stops seeing the sway, which is not the one stride a
  moving mean would need -- and the stride itself is measured from the sway's
  own spectrum. Off its feet is a lift over 4 cm above the walking plane or a
  fall past the same 60° the simulated metrics use -- the two are opposite
  signs of one quantity -- and both are cut out of the traces, with a fall also
  shaded to the end of its block and dated within it.
- Added ``scripts/eval/figure_style.py``, the palette, matplotlib defaults and
  save helpers shared by ``plot_comparison.py`` and ``plot_mocap_profile.py``,
  so a simulated figure and a captured one can sit on the same page.
- Added ``scripts/eval/eval_push_recovery.py`` and ``mjlab.evaluation.push``,
  which measure how hard a shove a controller can take. A constant force is
  applied through the torso for 0.2 s and the robot is watched for four seconds
  afterwards, swept over how hard the push is, which direction it comes from and
  where in the gait cycle it lands. Magnitude is parameterised as the velocity
  change the impulse would give a free body of the robot's mass rather than as a
  force, so plants that do not weigh the same are comparable; the force and the
  impulse are both recorded. Alongside the usual walking metrics each trial
  reports ``withstood``, ``time_to_fall``, ``recovered``, ``recovery_time``,
  ``peak_speed_error``, ``min_upright_after`` and ``heading_error``, and the
  summary carries the push *envelope*: for each direction, the magnitude at
  which half the trials end on the floor, interpolated from the survival curve.
- Added two push batteries per controller to
  ``scripts/eval/collect_comparison.sh`` -- one shoving the robot while it walks
  and one while it stands -- and figures 7 to 9 to
  ``scripts/eval/plot_comparison.py``: the envelope as a polar plot, survival
  against magnitude by direction, and what a survived push cost in recovery
  time, speed excursion and lean. The battery is sized so that repeating a
  collection reproduces it: each (direction, magnitude) point averages 48 trials
  spread over the gait cycle, since whether a marginal push topples the robot
  depends on where in the stride it lands. Collected twice at the defaults, the
  walk engine reported 30.0% and 30.4% withstood over the whole battery and its
  envelope moved by at most 0.028 m/s, under a third of a magnitude step.
  ``PUSH=0`` skips the batteries and the usual ``PUSH_*`` environment variables
  resize them.
- Added ``scripts/eval/eval_velocity_profile.py`` and
  ``mjlab.evaluation.profile``, which drive any of the three controllers with a
  *moving* velocity command -- forward, then sideways, then turning, then the
  three combinations -- and record the commanded and measured base velocity at
  every control step to ``trace.csv``. This is the tracking figure DeepWalk
  (Rodriguez and Behnke, ICRA 2021, Fig. 3) uses; the command sweeps measure
  steady state and say nothing about how the robot gets there. The six
  schedules run in independent slices of the batch, so a controller that falls
  under one command does not contaminate the rest of the sequence.
- Added ``--warmup`` to the three ``scripts/eval`` entry points, discarding the
  front of a run before the walking metrics start averaging. A robot starts from
  standing, so a mean over the whole run reports the acceleration as well as the
  tracking: the quintic engine averages 0.199 m/s over 10 s of a 0.3 m/s command
  and 0.212 over 30 s, against a steady state of 0.219. Survival is not
  windowed, and an environment that fell inside the warm-up reports NaN rather
  than a zero it never walked.
- Added ``scripts/eval/collect_comparison.sh`` and
  ``scripts/eval/plot_comparison.py``: the runs a controller needs for a
  velocity-tracking and stability comparison, and the figures drawn from them.
  Any number of controllers can be compared -- two policies against each other,
  or both against the walk engine -- each named on the command line as a
  ``engine=...,name=...,checkpoint=...`` list rather than being fixed by the
  script. ``task=`` selects the registered task supplying a policy's
  observation pipeline, which is what lets policies with different observation
  vectors sit in one comparison; the collection records what it collected in
  ``controllers.json``, and ``plot_comparison.py`` draws whatever it finds
  there. The run length, command ranges and replica count are read from the
  environment.
- Added ``mjlab.rl.obs_history``: an end-to-end observation-history encoder for
  the actor. The environment publishes a ``"history"`` observation group -- a
  25-step window of the actor observation stream, shaped
  ``[num_envs, T, obs_dim]`` -- and ``HistoryActor`` compresses it with a TCN
  whose latent is concatenated onto the current observation before the policy
  MLP. The encoder lives inside the model, so the latent never crosses the
  observation boundary and plain PPO trains it end to end. It is stateless by
  design, so deployment needs no cross-tick hidden state:
  ``OnnxHistoryPolicy`` takes one flat ``[B, T*D]`` window (time-major, oldest
  frame first), slices the current observation out of the last frame, and emits
  actions from a single graph. ONNX metadata gains ``history_window``,
  ``history_obs_dim`` and ``history_layout``.
- Added ``Mjlab-Velocity-Flat-Nubots-Nugus-History`` and
  ``Mjlab-Velocity-Rough-Nubots-Nugus-History``, the NUgus velocity tasks with
  that window published and ``HistoryActor`` as the actor. They are registered
  alongside the plain tasks rather than replacing them: a checkpoint only loads
  against the task that builds its observation layout, so comparing a plain
  policy with a history policy needs both. The actor's input to the policy MLP
  goes from 71 to 87 floats (71 proprio plus a 16-dim latent), the history
  group is ``[num_envs, 25, 71]``, and the exported ONNX input is 1775 floats.
- Added ``--profile.rest`` to ``scripts/eval/eval_velocity_profile.py``,
  setting how long a lane rests at zero between two commanded plateaus
  separately from ``--profile.hold``, which now only sets how long a *commanded*
  plateau is held. The rest exists to separate one command from the next and to
  let the robot come back to a stand, not to be measured, so holding it as long
  as a commanded plateau spent a third of the run standing still. At the
  comparison's timing a lane goes from 36 s to 24 s with every commanded
  plateau untouched.
- Changed ``fig1`` of ``scripts/eval/plot_comparison.py`` from six panels to a
  single continuous trace, the six schedules laid end to end on one time axis
  with each block named and washed alternately. The schedules ran in parallel
  slices of the batch, so a boundary is a change of robot rather than a change
  of command, and each lane is drawn over its own schedule only -- the tail a
  short lane spends held at rest while the others finish is padding, not
  measurement.
- Added ``--task-id`` to ``scripts/eval/eval_rl_walk.py`` and
  ``scripts/eval/eval_velocity_profile.py``, naming the registered task whose
  observation, action and command pipeline a checkpoint is played back against.
  A checkpoint only loads against the task it was trained on, so a policy with a
  non-default observation layout could not be evaluated before.

- Added ``scripts/eval/eval_distilled_quintic_walk.py`` and
  ``mjlab.controllers.distilled_walk``, which run NUbots' distilled walk policy
  -- the MLP their ``module/skill/NeuralWalk`` deploys, trained to copy the
  quintic walk engine's joint targets -- as a third controller in the
  evaluation, on the same plants and metrics as the other two. The deployed
  ``walk_policy.onnx`` ships with mjlab and its weights are replayed in torch so
  a whole batch infers at once. The policy reads no measurement of the robot, so
  it runs on all four plants. ``--track-teacher`` runs the walk engine alongside
  and reports how far the copy's joint targets ran from the original's, and
  ``--history-init`` selects what the run starts from. See
  ``scripts/eval/README.md``.
- Added ``stand_on_leg_targets`` to the quintic walk playback module: the
  standing-pose solve (adopt the joint angles, level the stance sole, drop it
  onto the floor) that both ``WalkPlayback`` and the evaluation harnesses use to
  start a robot in whatever pose its controller is about to ask for.
- The quintic walk engine now computes in ``torch.float64`` by default
  (``ENGINE_DTYPE``), matching the C++ ``WalkGenerator<double>``. Its phase
  clock accumulates the control period and switches feet on
  ``t >= step_period``; in float32 thirty-two additions of 0.01 fell just short
  of the period, so every step took 0.33 s instead of 0.32 -- a three percent
  cadence error. Simulation stays in float32, and the controller casts at the
  boundary. Reported speeds shift by a few percent as a result.
- Added ``--record`` to ``scripts/tools/play_quintic_walk.py``, writing a
  per-control-step CSV of commanded joint targets, measured joint positions and
  velocities, torso and foot orientation, gyro and sole heights, plus a metadata
  JSON describing the run. Intended for comparing mjlab against another
  simulator trace against trace; see ``WalkRecorder``.
- Added a ``--viser`` flag to both ``scripts/eval`` entry points, streaming one
  environment of a batched evaluation run to a viser server for visual
  inspection, with ``--viser-port``, ``--viser-env`` and ``--viser-realtime``.
  It hangs off the harnesses' existing ``on_step`` callback and only ever draws
  and sleeps, so a watched run and a headless one produce identical metrics.
- Added ``mjlab.evaluation`` and the ``scripts/eval`` entry points: a batched,
  GPU-parallel pipeline for comparing walk controllers over a 2x2 of engine
  (ported quintic walk, trained policy) and plant (evaluation, training).
  ``eval_quintic_walk.py`` and ``eval_rl_walk.py`` are thin wrappers over a
  shared harness, and both feed one metrics recorder from raw simulator state,
  so per-environment survival, velocity tracking, torso attitude and cadence
  are computed identically for either engine. Results are written as a
  per-environment CSV plus a JSON summary. See ``scripts/eval/README.md``.
- Added ``sole_poses_from_body_states``, a batched version of the quintic
  walk's foot-pose measurement, so planted-phase sensing works on device for a
  whole batch rather than only for a single ``MjData``.
- The verbatim NUbots NUgus config now wraps its MJCF's own ``<position>``
  actuators with ``XmlActuatorCfg``, leaving the compiled model identical but
  letting mjlab drive it with joint position targets.
- Added two NUgus asset configs for running controllers against something
  other than the training model: ``get_nugus_eval_robot_cfg`` (sim-to-real
  randomisation at nominal and the hardware ±π leg joint limits, otherwise
  the training model unchanged) and ``get_nugus_nubots_robot_cfg`` (NUbots'
  own MJCF verbatim). The training model is untouched, so trained policies
  are unaffected.
- Added ``mjlab.controllers.quintic_walk.playback``, the rig that drives the
  ported NUbots quintic walk engine against a compiled NUgus, and gave
  ``scripts/tools/play_quintic_walk.py`` a ``--plant`` selector over the
  training, evaluation, NUbots-parity and NUbots-verbatim models. With the
  deployed walk parameters the engine walks on the evaluation model and falls
  after 1.72 s on the training model; the passive backlash joints are the
  whole of that difference.
- Added ``sole_poses_in_torso`` to the quintic walk controller, which measures
  the foot poses ``detect_planted_phase`` consumes, so playback supplies the
  sensed foot phase every control step and ``only_switch_when_planted`` can be
  exercised.
- Added a ``feet_swing_height_clock`` velocity reward and a matching
  ``gait_clock`` observation. An independent, fixed-frequency gait clock (not
  controlled by the policy) drives a desired per-foot swing-height arc, densely
  rewarding the foot for tracking it; the same clock is fed to the policy as a
  ``[sin, cos]`` phase observation. A larger clock ``period`` commands a slower
  cadence. Enabled for NUbots Nugus to encourage larger, slower steps with more
  foot clearance.
- Added an optional ``feet_flat_orientation`` reward for velocity tasks that
  penalizes foot-sole tilt during swing, encouraging flat-footed stepping so
  the toe/front edge does not pitch down and dig into the ground on touchdown.
  Enabled and tuned for NUbots Nugus.
- Added ``ContactSensor.primary_names`` property to expose the resolved
  primary names in the order they appear along the per-contact axis of the
  output tensors. This makes it possible to map a contact-data column back
  to the primary it belongs to (:issue:`914`).
- Added a servo backlash model to the NUbots Nugus robot. Each actuated
  joint now has a passive ``_backlash`` sibling joint with the same axis
  bounded to ±``NUGUS_BACKLASH_VALUE`` (default 0.035 rad) that models
  transmission play between the motor and the link. The nugus velocity
  task observations, rewards, and reset events are scoped to motor joints
  only via ``NUGUS_MOTOR_JOINT_REGEX`` so the passive joints do not appear
  in the policy's view.

Fixed
^^^^^

- Fixed the quintic walk engine re-parking the whole batch instead of only the
  environments it was resetting. ``WalkGenerator.update`` re-parks its stopped
  environments on every control step, and ``reset`` regenerated the standing
  trajectories for every environment rather than for ``env_ids``, so a single
  zero velocity command made every other environment in the batch march on the
  spot. A batched command sweep -- which always contains a zero command --
  therefore reported one speed for every command. Single-environment playback,
  and any batch without a zero command, were unaffected.

Changed
^^^^^^^

- Added ``power`` and ``only_below`` parameters to the ``feet_clearance``
  velocity reward. ``power=2`` uses a squared height error (stronger gradient
  far below target) and ``only_below=True`` penalizes only feet below the
  target height, leaving a high swing apex unpenalized. Defaults preserve the
  previous linear, symmetric behavior; enabled for NUbots Nugus.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Updated velocity task training defaults to improve gait naturalness: added
  optional left-right limb symmetry and actuation power cost rewards, and
  slowed velocity command curriculum ramp-up. Enabled/tuned these rewards for
  NUbots Nugus.
- Enabled the left-right limb symmetry reward for NUbots Nugus (previously
  disabled) and expanded it to cover the full leg (hip yaw/roll/pitch, knee,
  ankle pitch/roll) rather than only the sagittal pitch joints, to address a
  lop-sided trained gait.
- Added an optional feet-separation penalty for velocity tasks to discourage
  feet from getting too close; enabled and tuned for NUbots Nugus.
- Added optional ``cot_proxy`` and ``gait_phase_regularity`` rewards for
  velocity tasks to improve locomotion efficiency and left-right gait timing.
  Enabled and tuned these for NUbots Nugus.
- Task package load failures during ``mjlab`` import now print the full
  traceback (and the entry point's module path) to ``stderr`` instead of
  just the exception message, making it easier to pinpoint the source of
  import errors when running commands like ``list-envs`` (:issue:`910`).
  Contribution by @saikishor.
- Clarified ``ContactSensor`` shape conventions: per-contact fields
  (``found``, ``force``, ``torque``, ``dist``, ``pos``, ``normal``,
  ``tangent``) have shape ``[B, P * num_slots, ...]`` while per-primary
  air-time fields (``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time``) have shape ``[B, P]``,
  where ``P`` is the number of resolved primaries (:issue:`914`).

Fixed
^^^^^

- Fixed ``out_of_terrain_bounds`` using stale terrain dimensions. It read
  ``TerrainGeneratorCfg.num_cols`` directly, which is ignored in curriculum
  mode (the generator uses ``len(sub_terrains)`` columns instead), and it
  did not account for ``border_width``. The termination now reads the
  effective grid shape from ``terrain.terrain_origins`` and includes the
  border in the footprint, so robots no longer reset while still on valid
  terrain (or fail to reset after running off it) (:issue:`923`).
- ``ObservationManager`` now skips observation groups that end up with
  zero active terms (e.g. all terms set to ``None``) with a log message,
  instead of crashing later in ``torch.stack``/``torch.cat``. This lets
  a shared runner config define groups that become empty under certain
  runtime flags (e.g. model-specific terms all disabled for one variant).
  The whole group can still be set to ``None`` to disable it explicitly.
- Fixed a runtime broadcast error in ``ContactSensor`` when combining
  ``num_slots > 1`` with ``track_air_time=True`` and more than one primary.
  Air-time tracking now reduces ``found`` across slots so that a primary is
  considered in contact when any of its slots reports a match (:issue:`914`).
- Updated the ``create_new_task.ipynb`` Colab tutorial to import
  ``XmlActuatorCfg`` instead of the removed ``XmlVelocityActuatorCfg``.
  Added a regression test (``tests/test_notebooks.py``) that parses each
  notebook cell and verifies that every ``from mjlab... import X``
  reference resolves, so future renames in the mjlab public API can't
  silently rot the tutorials (:issue:`913`).
- Fixed ``ObservationManager`` silently sharing a single ``NoiseModelCfg``
  instance across observation groups that declared terms with the same
  name. ``_group_obs_class_instances`` was keyed by term name alone, so
  the last group processed in ``_prepare_terms`` overwrote earlier
  groups' instances. Symptoms included the wrong noise config being
  applied, shared per-episode state for ``NoiseModelWithAdditiveBias``
  (e.g. bias drawn from the wrong ``bias_noise_cfg``), and missed
  ``reset()`` calls for overwritten instances. Instances are now keyed
  by ``(group_name, term_name)`` so each group owns its own noise model.
- Fixed ``CurriculumManager.get_active_iterable_terms`` raising
  ``TypeError`` when a term's state was a dict. The dict branch indexed
  the output list by term name instead of appending to the local ``data``
  list. No in-tree caller currently invokes this method, so the bug was
  latent.

Version 1.3.0 (April 14, 2026)
------------------------------

Added
^^^^^

- Added ``ManagerBasedRlEnvCfg.auto_reset`` flag. When ``True`` (default),
  ``step()`` continues to reset done environments in place and returns the
  post-reset observation. When ``False``, ``step()`` skips the reset block
  and returns the terminal observation directly; the caller must call
  ``reset(env_ids=...)`` for done environments before the next ``step()``
  or a ``RuntimeError`` is raised. Enables access to the true terminal
  state for algorithms that need it. Note that mjlab's bundled ``train.py``
  uses rsl_rl's ``OnPolicyRunner``, which does not drive manual resets, so
  ``auto_reset=False`` is intended for custom training loops (:issue:`900`).
- Added ``ActuatorCfg.viscous_damping`` for passive velocity proportional
  damping (``f = -b·v``), distinct from the PD derivative gain ``damping``
  used by position and velocity actuators. Maps to ``<joint damping>`` for
  JOINT transmission and ``<tendon damping>`` for TENDON transmission.
  Defaults to ``None`` (preserves the XML value).
- Added :class:`~mjlab.managers.RecorderManager` for logging observations,
  actions, or arbitrary environment data during rollouts. Implement a
  :class:`~mjlab.managers.RecorderTerm` subclass and register it in the
  ``recorders`` dict on ``ManagerBasedRlEnvCfg``. The manager provides
  ``record_pre_reset``, ``record_post_reset``, and ``record_post_step``
  lifecycle hooks with no opinion on how data is stored.
- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``), with an
  ``isotropic=True`` option that mirrors the symmetric tangent and roll
  axes so single-axis randomization does not leave the paired axis stale.
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Renamed the ``list_envs`` console script to ``list-envs`` for consistency
  with the other hyphenated entry points (``viz-nan``, ``export-scene``).
  Invoke via ``uv run list-envs``.
- ``ActuatorCfg.armature`` and ``ActuatorCfg.frictionloss`` now default to
  ``None`` instead of ``0.0``. ``None`` preserves the value defined in the
  XML. Previously, builtin actuators would silently overwrite XML joint and
  tendon properties with zero when these fields were not explicitly set.
  To restore the old behavior, pass ``armature=0.0`` or ``frictionloss=0.0``
  explicitly.
- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).
- ``get_wandb_checkpoint_path`` now filters checkpoints server-side via the
  ``pattern`` parameter, avoiding unnecessary pagination and tolerance to
  corrupted metadata (:issue:`898`).

Fixed
^^^^^

- ``train`` and ``play`` now print a top-level usage message when invoked
  with ``-h`` / ``--help`` and no task argument, pointing users at
  ``list-envs`` and ``<TASK> --help`` (:issue:`905`).
- Fixed ghost geom filtering in the Viser viewer. Ghost geoms were selected
  by collision flags, so collision-disabled robot geoms appeared as ghosts.
  The viewer now uses visual alpha to determine which geoms to render.
- Scene now warns when an attached entity or terrain spec has non-default
  ``<option>`` fields (e.g. ``<flag contact="disable"/>``), which are
  silently dropped by ``MjSpec.attach()``. Use ``MujocoCfg`` to set
  simulation options instead (:issue:`885`).
- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed W&B checkpoint resume for runs where ``run.files()`` fails but direct
  file lookup still works, by trying ``model_<summary_step>.pt`` and then
  fallback names such as ``last.pt``/``model.pt``.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
