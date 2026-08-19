"""Optional live playback of an evaluation run, in the browser.

Evaluation is a headless, batched activity: thousands of robots, no rendering,
numbers out the far end. This module is the exception for when a number looks
wrong and you want to see what the robot is actually doing -- one environment,
streamed to a viser scene, at roughly real time.

It is deliberately outside the measurement path. The only thing it touches is
the ``on_step`` callback both harnesses already accept, and the only thing it
does there is copy state out to the browser and sleep. Sleeping changes how long
the run takes on the wall clock and nothing else: the physics has already been
integrated by then, at a fixed timestep, so a paced run and a headless run
produce the same trajectories and the same metrics.

Rendering thousands of environments through this would be miserable, and that is
fine -- it is for looking at one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol

import tyro

if TYPE_CHECKING:
  import mujoco

  from mjlab.sim import Simulation

DEFAULT_VISER_PORT = 8080
"""viser's own default, so the URL is the one people already have bookmarked."""


class Viewable(Protocol):
  """What :class:`LiveView` needs from a harness.

  Both :class:`~mjlab.evaluation.harness.QuinticEvalHarness` and
  :class:`~mjlab.evaluation.harness.RlEvalHarness` satisfy this.
  """

  num_envs: int
  control_dt: float

  @property
  def sim(self) -> Simulation: ...


@dataclass
class LiveViewCfg:
  """Command-line surface, shared by both entry points.

  Embed it with :data:`LIVE_VIEW_FLAGS` so the flags read ``--viser``,
  ``--viser-port`` and so on rather than carrying a nested prefix.
  """

  viser: bool = False
  """Stream one environment to a viser server in the browser.

  For looking at a gait, not for collecting data: it throttles the run to real
  time and renders every control step. Use it with ``--num-envs 1``."""
  viser_port: int = DEFAULT_VISER_PORT
  """Port for the viser server."""
  viser_env: int = 0
  """Index of the environment to show."""
  viser_realtime: bool = True
  """Throttle stepping so the playback runs no faster than real time.

  Only ever sleeps, so it changes the wall time of a run and not its physics or
  its metrics. Turn it off to watch the run go past as fast as it computes."""


LIVE_VIEW_FLAGS = Annotated[LiveViewCfg, tyro.conf.OmitArgPrefixes]
"""Annotation that flattens :class:`LiveViewCfg` onto the top-level flags."""


class RealTimePacer:
  """Sleeps so that simulated time does not outrun the wall clock.

  Split out from :class:`LiveView` so the timing can be tested without standing
  up a web server: the clock and the sleep are injectable.
  """

  def __init__(
    self,
    control_dt: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
  ) -> None:
    self._control_dt = control_dt
    self._clock = clock
    self._sleep = sleep
    self._started = clock()
    self._steps = 0

  def wait(self) -> float:
    """Count one step and wait out any lead. Returns the seconds slept."""
    self._steps += 1
    behind = self._steps * self._control_dt - (self._clock() - self._started)
    if behind <= 0.0:
      return 0.0
    self._sleep(behind)
    return behind


class LiveView:
  """Streams one environment of a batched run to a viser scene.

  Use as a context manager, and hand :meth:`on_step` to the harness::

    with LiveView(harness, cfg) as view:
      metrics = harness.run(command, duration, on_step=view.on_step)
  """

  def __init__(
    self,
    harness: Viewable,
    env_index: int = 0,
    port: int = DEFAULT_VISER_PORT,
    realtime: bool = True,
  ) -> None:
    if not 0 <= env_index < harness.num_envs:
      raise ValueError(
        f"viser_env {env_index} is outside the batch of {harness.num_envs}"
      )

    # Imported after the argument check and inside the constructor rather than
    # at module scope: viser pulls in a web server and a mesh stack, and a
    # headless collection run should not pay for either.
    import viser

    from mjlab.viewer.viser.scene import MjlabViserScene

    self._harness = harness
    self._env_index = env_index
    self._pacer = RealTimePacer(harness.control_dt) if realtime else None
    self._server = viser.ViserServer(port=port, label="mjlab eval")
    self._scene = MjlabViserScene(
      server=self._server,
      mj_model=self.mj_model,
      num_envs=harness.num_envs,
    )

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._harness.sim.mj_model

  @property
  def url(self) -> str:
    """Address to open. ``get_host`` reports the bind address, which is
    ``0.0.0.0`` by default and not something a browser can follow."""
    host = self._server.get_host()
    if host in ("0.0.0.0", "::"):
      host = "localhost"
    return f"http://{host}:{self._server.get_port()}"

  def on_step(self, step: int) -> None:
    """Callback for ``harness.run(..., on_step=...)``.

    Draws the current state, then waits until the wall clock has caught up with
    simulated time. Called after the physics for the step is already done, so
    the wait cannot influence it.
    """
    del step
    self._scene.update(self._harness.sim.data, self._env_index)
    if self._pacer is not None:
      self._pacer.wait()

  def close(self) -> None:
    self._server.stop()

  def __enter__(self) -> LiveView:
    return self

  def __exit__(self, *exc_info) -> None:
    del exc_info
    self.close()


def open_live_view(harness: Viewable, cfg: LiveViewCfg) -> LiveView | None:
  """Start a viser server if ``cfg.viser`` is set, and announce the URL.

  Returns ``None`` when the flag is off, which is what keeps the headless path
  free of viser entirely -- no import, no server, no callback.
  """
  if not cfg.viser:
    return None
  view = LiveView(
    harness,
    env_index=cfg.viser_env,
    port=cfg.viser_port,
    realtime=cfg.viser_realtime,
  )
  pacing = "real time" if cfg.viser_realtime else "as fast as it computes"
  print(f"viser             : {view.url} (env {cfg.viser_env}, {pacing})")
  return view
