"""Drive the NUgus with the ported quintic walk engine, in MuJoCo.

This is the empirical check on the port: no learned policy, no mjlab
environment, just the walk engine commanding leg joint targets against a
compiled NUgus and a floor. The rig itself lives in
:mod:`mjlab.controllers.quintic_walk.playback` so the regression tests drive
exactly what this script drives.

Defaults are the deployed ones -- ``Walk.yaml``'s tuning, the idealised leg IK
the robot ships, and the evaluation plant (see ``--plant``) -- and they walk.

Examples::

  # Interactive viewer, walking forward at 0.2 m/s.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2

  # Headless, write a video.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --video /tmp/walk.mp4

  # Against the model policies are trained on. Falls; that is the point.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --plant training

  # Open loop, without the FootController balance correction. mjlab runs tyro
  # with flag conversion off, so booleans take an explicit value.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --balance False

  # Solve the legs against the compiled geometry instead of the idealised leg.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --exact-ik True
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import tyro

import mjlab
from mjlab.controllers.quintic_walk.playback import (
  DEFAULT_PLANT,
  Plant,
  PlaybackResult,
  WalkPlayback,
)
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  WalkParameters,
)


@dataclass
class Args:
  vx: float = 0.2
  """Forward velocity command, in m/s."""
  vy: float = 0.0
  """Lateral velocity command, in m/s."""
  wz: float = 0.0
  """Yaw rate command, in rad/s."""
  duration: float = 10.0
  """Simulated duration, in seconds."""
  plant: Plant = DEFAULT_PLANT
  """Which robot model to drive.

  ``eval`` is the reference: sim-to-real randomisation at nominal and hardware
  joint limits. ``training`` is the model policies are trained against, which
  the engine falls on. ``nubots-sim`` is NUbots' dynamics on mjlab's kinematic
  tree, ``nubots-xml`` their MJCF verbatim."""
  control_hz: float = 100.0
  """Walk engine update rate. The robot runs at 100 Hz."""
  balance: bool = True
  """Apply the FootController torso-orientation correction."""
  video: Path | None = None
  """Write an mp4 here instead of opening the viewer."""
  video_fps: int = 50
  """Frame rate of the written video."""

  # Experiment knobs. Every one of these departs from what the robot deploys;
  # they are here to isolate one behaviour at a time, not to make the walk work.
  torso_height: float | None = None
  """Override the walk torso height, in metres. Deployed value is 0.44."""
  step_width: float | None = None
  """Override the lateral foot separation, in metres. Deployed value is 0.27."""
  step_period: float | None = None
  """Override the step duration, in seconds. Deployed value is 0.32."""
  step_height: float | None = None
  """Override the swing foot apex height, in metres. Deployed value is 0.085."""
  switch_when_planted: bool = False
  """Wait for the sensed foot phase before switching the planted foot.

  What ``Walk.yaml`` asks for and what the deployed binary fails to apply; see
  :data:`~mjlab.controllers.quintic_walk.walk_generator.NUGUS_WALK_PARAMETERS`.
  """
  exact_ik: bool = False
  """Solve the legs against the compiled geometry instead of the engine's
  idealised leg. Not the deployed behaviour -- the robot runs the idealised
  solver, and this isolates the algorithm from its kinematic model error."""


def walk_parameters(args: Args) -> WalkParameters:
  """Apply any command-line overrides to the deployed NUgus tuning."""
  overrides = {
    name: value
    for name, value in (
      ("torso_height", args.torso_height),
      ("step_width", args.step_width),
      ("step_period", args.step_period),
      ("step_height", args.step_height),
    )
    if value is not None
  }
  if args.switch_when_planted:
    overrides["only_switch_when_planted"] = True
  return replace(NUGUS_WALK_PARAMETERS, **overrides)


def summarise(
  args: Args, playback: WalkPlayback, params: WalkParameters, result: PlaybackResult
) -> str:
  fate = f"FELL OVER at {result.fall_time:.2f} s" if result.fell else "stayed upright"
  ik = "exact (compiled geometry)" if playback.controller.uses_exact_ik else "idealised"
  switching = "sensed foot contact" if params.only_switch_when_planted else "step clock"
  return (
    f"plant             : {args.plant}\n"
    f"leg IK            : {ik}\n"
    f"foot switching    : {switching}\n"
    f"engine state      : {result.engine_state.name}\n"
    f"simulated         : {result.elapsed:.2f} s at "
    f"{1.0 / playback.control_dt:.0f} Hz control\n"
    f"torso height      : {result.torso_height:.3f} m "
    f"(walk torso_height {params.torso_height:.3f})\n"
    f"upright           : {result.upright:+.3f}, minimum {result.min_upright:+.3f} "
    f"({fate})\n"
    f"displacement (x,y): {result.displacement[0]:+.3f}, "
    f"{result.displacement[1]:+.3f} m\n"
    f"mean forward speed: {result.mean_speed:+.3f} m/s (commanded {args.vx:+.3f})"
  )


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)

  params = walk_parameters(args)
  playback = WalkPlayback(
    plant=args.plant,
    walk_params=params,
    control_hz=args.control_hz,
    use_balance_control=args.balance,
    exact_ik=args.exact_ik,
  )
  command = (args.vx, args.vy, args.wz)

  if args.video is not None:
    import mediapy
    import mujoco

    frames = []
    every = max(1, round((1.0 / args.video_fps) / playback.control_dt))
    renderer = mujoco.Renderer(playback.model, height=480, width=640)

    def capture(step: int) -> None:
      if step % every == 0:
        renderer.update_scene(playback.data, camera="track")
        frames.append(renderer.render())

    result = playback.run(command, args.duration, on_step=capture)
    renderer.close()
    args.video.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(args.video), frames, fps=args.video_fps)
    print(f"wrote {len(frames)} frames to {args.video}")
  else:
    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(playback.model, playback.data) as viewer:

      def sync(step: int) -> bool:
        viewer.sync()
        return viewer.is_running()

      result = playback.run(command, args.duration, on_step=sync)

  print(summarise(args, playback, params, result))


if __name__ == "__main__":
  main()
