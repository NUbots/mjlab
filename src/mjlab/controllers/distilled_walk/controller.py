"""The distilled walk policy wired up as a joint-position controller.

Reproduces NUbots' ``module/skill/NeuralWalk``: slew the velocity command, run
the walk engine's state machine for its phase clock, hand the policy that clock
plus its own three previous outputs, and emit what comes back as leg joint
targets.

The clock comes from
:class:`~mjlab.controllers.quintic_walk.walk_generator.WalkGenerator` rather
than from a hand-rolled state machine. That is the engine whose output the
policy was distilled from -- ``WalkDataCollector`` drove exactly this object to
generate the training set -- and mjlab's port of it is pinned against the C++ by
golden traces. ``NeuralWalk.cpp`` re-implements the same state machine inline;
at a fixed control period the two agree step for step, and replaying a recorded
episode's commands through this one reproduces its clock columns to 3e-8
(``test_distilled_walk_controller.py``).

**The policy is blind.** Its observation contains no measurement of the robot:
only the command, the phase, and its own history. So this controller ignores
attitude, gyro and sensed contact, and takes no arguments for them. Everything
it produces is a function of the command sequence alone, which is what makes it
a copy of a trajectory generator rather than a feedback controller.

**It is not a copy of *this* engine's joint angles.** ``WalkDataCollector`` fed
the engine's foot poses through NUbots' deployed IK -- the analytical solution
refined numerically against ``models/robot.urdf`` -- so the policy learned that
solver's joint angles, not the ones
:func:`~mjlab.controllers.quintic_walk.kinematics.calculate_leg_joints`
produces. Standing, the difference is half a radian at the knee (1.359 against
0.845; solving against the *MJCF* geometry instead gives 1.723). Compare joint
angles between the two engines with that in mind, and see ``--track-teacher`` in
``scripts/eval/eval_distilled_quintic_walk.py``.

Observation layout, from ``recordings/walk_data/metadata.yaml``:

===========  =====  =======================================================
Index        Width  Contents
===========  =====  =======================================================
``0:3``      3      slew-limited velocity command ``(vx, vy, vtheta)``
``3:5``      2      ``sin`` and ``cos`` of the step phase
``5:6``      1      planted foot: ``+1`` left, ``-1`` right
``6:10``     4      engine state one-hot: stopped, starting, walking, stopping
``10:22``    12     joint targets from the previous control step
``22:34``    12     the step before that
``34:46``    12     the step before that
===========  =====  =======================================================
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from mjlab.controllers.distilled_walk.policy import TARGET_DIM, DistilledWalkPolicy
from mjlab.controllers.quintic_walk.controller import JOINT_NAMES
from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_LEG,
  LegModel,
  calculate_leg_joints,
)
from mjlab.controllers.quintic_walk.walk_generator import (
  ENGINE_DTYPE,
  NUGUS_MAX_ACCELERATION,
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
  WalkGenerator,
  WalkParameters,
)

HISTORY_FRAMES = 3
"""Previous joint-target frames the observation carries."""

POLICY_DTYPE = torch.float32
"""The policy's own precision, matching the ONNX export and the deployed
OpenVINO inference. Only the engine clock underneath it runs in double."""

SETTLE_ITERATIONS = 256
"""Cap on the iterations :meth:`DistilledWalkController.settled_targets` runs."""

SETTLE_TOLERANCE = 1e-7
"""Radians of movement below which the standing output counts as settled."""

HistoryInit = Literal["settled", "stance", "zeros"]
"""How the run starts; see :class:`DistilledWalkController`."""

_STATE_ONEHOT: dict[int, tuple[float, float, float, float]] = {
  int(EngineState.STOPPED): (1.0, 0.0, 0.0, 0.0),
  int(EngineState.STARTING): (0.0, 1.0, 0.0, 0.0),
  int(EngineState.WALKING): (0.0, 0.0, 1.0, 0.0),
  int(EngineState.STOPPING): (0.0, 0.0, 0.0, 1.0),
  int(EngineState.UNKNOWN): (0.0, 0.0, 0.0, 0.0),
}
"""One-hot rows in the order ``WalkDataCollector`` wrote them.

``UNKNOWN`` never comes out of the engine -- it resets to ``STOPPED`` -- and has
no column of its own in the observation, so it encodes as all zeros.
"""


class DistilledWalkController:
  """Batched distilled walk policy producing leg joint position targets.

  Interchangeable with
  :class:`~mjlab.controllers.quintic_walk.controller.QuinticWalkController` at
  the harness: same twelve joints in :data:`JOINT_NAMES` order, same 100 Hz,
  same slew-limited command. It just needs far less to compute them.
  """

  joint_names = JOINT_NAMES

  def __init__(
    self,
    num_envs: int,
    policy: DistilledWalkPolicy,
    device: torch.device | str = "cpu",
    walk_params: WalkParameters = NUGUS_WALK_PARAMETERS,
    max_acceleration: tuple[float, float, float] = NUGUS_MAX_ACCELERATION,
    leg_model: LegModel = NUGUS_LEG,
    history_init: HistoryInit = "settled",
    dtype: torch.dtype = ENGINE_DTYPE,
  ) -> None:
    """Build a controller.

    Args:
      policy: Loaded by :meth:`DistilledWalkPolicy.from_onnx`. Must already be
        on ``device``.
      max_acceleration: Command slew limit. Defaults to the ``0.2, 0.2, 0.5``
        of ``Walk.yaml``, which is what ``WalkDataCollector`` ramped the
        training commands with. ``NeuralWalk.hpp`` hardcodes ``0.5, 0.5, 0.5``
        and reads no value from its config, so the deployed robot ramps two and
        a half times faster in x and y than anything the policy was trained on.
      leg_model: Geometry for the engine stance the history can start from.
      history_init: What the three history frames hold at reset, and with them
        the pose :attr:`home_targets` asks the robot to start in.

        ``settled``
          The policy's own standing output, from
          :meth:`settled_targets`. This is the pose it holds at rest, and it
          agrees with the stance recorded in NUbots' training data to 1e-4 rad
          -- so it recovers the training-time stance without needing the IK
          that produced it.
        ``stance``
          The engine's stopped stance as *this* port solves it, which is the
          same start ``eval_quintic_walk.py`` uses and half a radian of knee
          away from what the policy expects. It leaves the stance within three
          control steps.
        ``zeros``
          Thirty-six zeros, as ``NeuralWalk.cpp`` assigns on start -- an
          observation unlike anything in the training set, for the three steps
          it takes to flush. The robot still starts in the engine's stance,
          which is where it is standing when the walk task takes over.
      dtype: Precision of the engine clock underneath the policy. The policy
        itself always runs in :data:`POLICY_DTYPE`.
    """
    self.device = torch.device(device)
    self.dtype = dtype
    self.num_envs = num_envs
    self.policy = policy
    self.leg_model = leg_model
    self.history_init = history_init

    self.generator = WalkGenerator(
      num_envs, device=self.device, params=walk_params, dtype=dtype
    )
    self._max_acceleration = torch.tensor(
      max_acceleration, device=self.device, dtype=dtype
    )
    self._command = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
    self._onehot = torch.tensor(
      [_STATE_ONEHOT[int(state)] for state in EngineState],
      device=self.device,
      dtype=POLICY_DTYPE,
    )
    self._history = torch.zeros(
      num_envs, HISTORY_FRAMES, TARGET_DIM, device=self.device, dtype=POLICY_DTYPE
    )
    # Both stances are fixed properties of the engine and the policy, so they
    # are solved once here rather than on every reset.
    self._stance = self.stance_targets()
    self._settled = self.settled_targets()
    self.reset()

  @property
  def velocity_command(self) -> torch.Tensor:
    """Shape ``(N, 3)`` slew-limited command the phase clock is running on."""
    return self._command

  @property
  def history(self) -> torch.Tensor:
    """Shape ``(N, 3, 12)`` previous joint targets, most recent first."""
    return self._history

  @property
  def home_targets(self) -> torch.Tensor:
    """Shape ``(N, 12)`` pose the robot should be standing in at the first step.

    The policy's own standing pose when it starts from one, and the engine's
    otherwise; a robot started anywhere else spends the first control steps
    being dragged to wherever the controller is asking for.
    """
    return self._settled if self.history_init == "settled" else self._stance

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset the engine clock, the command filter and the action history."""
    self.generator.reset(env_ids)
    index = slice(None) if env_ids is None else env_ids
    self._command[index] = 0.0
    if self.history_init == "zeros":
      self._history[index] = 0.0
    elif self.history_init == "stance":
      self._history[index] = self._stance[index].unsqueeze(1)
    else:
      self._history[index] = self._settled[index].unsqueeze(1)

  def stance_targets(self) -> torch.Tensor:
    """Shape ``(N, 12)`` joint targets holding the *engine's* current stance.

    Solved from the trajectories the generator is holding, which is how
    ``WalkDataCollector`` seeded its history: reset the engine, take its foot
    poses, and run them through the leg IK. Through a different IK, though --
    see the module docstring.
    """
    legs = [
      calculate_leg_joints(
        self.generator.foot_pose(left=left), left=left, model=self.leg_model
      )
      for left in (True, False)
    ]
    return torch.cat(legs, dim=-1).to(POLICY_DTYPE)

  def settled_targets(self) -> torch.Tensor:
    """Shape ``(N, 12)`` joint targets the policy holds standing still.

    The policy is autoregressive, so "what does it output at rest" is a fixed
    point rather than a constant: feed it a stopped engine and a history of its
    own last answer until the answer stops moving. From the engine's stance it
    converges to 1e-6 rad within one step period.

    Requires the engine to be in its reset state, which is where the standing
    observation comes from. Iteration stops at :data:`SETTLE_TOLERANCE` or
    :data:`SETTLE_ITERATIONS`, whichever comes first -- a policy that limit
    cycles instead of converging yields its last iterate.
    """
    targets = self.stance_targets()
    for _ in range(SETTLE_ITERATIONS):
      history = targets.unsqueeze(1).expand(-1, HISTORY_FRAMES, -1)
      with torch.no_grad():
        settled = self.policy(self.observation(history))
      moved = float((settled - targets).abs().max())
      targets = settled
      if moved < SETTLE_TOLERANCE:
        break
    return targets

  def observation(self, history: torch.Tensor | None = None) -> torch.Tensor:
    """Shape ``(N, 46)`` observation for the engine's current phase.

    Read after the clock has been advanced, matching the order
    ``WalkDataCollector`` recorded in: the observation describes the phase the
    targets are being asked for, not the one they came from.

    Args:
      history: Shape ``(N, 3, 12)`` action history to build with. Defaults to
        the controller's own.
    """
    angle = 2.0 * math.pi * self.generator.time / self.generator.step_period
    indicator = torch.where(
      self.generator.phase == int(Phase.LEFT),
      torch.ones_like(angle),
      -torch.ones_like(angle),
    )
    clock = torch.stack((torch.sin(angle), torch.cos(angle), indicator), dim=-1)
    return torch.cat(
      (
        torch.cat((self._command, clock), dim=-1).to(POLICY_DTYPE),
        self._onehot[self.generator.state],
        (self._history if history is None else history).flatten(start_dim=1),
      ),
      dim=-1,
    )

  def compute(self, dt: float, velocity_command: torch.Tensor) -> torch.Tensor:
    """Advance the clock by ``dt`` and run the policy once.

    Args:
      dt: Control period in seconds. The policy was distilled at 100 Hz and its
        history is a fixed number of control steps deep, so this is not a knob:
        running it slower stretches the gait's memory as well as its clock.
      velocity_command: Shape ``(N, 3)`` requested ``(vx, vy, vtheta)``.

    Returns:
      Shape ``(N, 12)`` joint position targets ordered as :data:`JOINT_NAMES`.
    """
    velocity_command = velocity_command.to(self.dtype)
    delta = self._max_acceleration * min(dt, 1.0)
    self._command = self._command + (velocity_command - self._command).clamp(
      -delta, delta
    )

    self.generator.update(dt, self._command)

    with torch.no_grad():
      targets = self.policy(self.observation())

    self._history = torch.cat((targets.unsqueeze(1), self._history[:, :-1]), dim=1)
    return targets
