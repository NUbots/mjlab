"""Video recording wrapper for environments."""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Literal

import mediapy as media
import numpy as np
import torch
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnv

# mediapy shells out to an ffmpeg binary; headless training images often
# ship without one. imageio-ffmpeg bundles a static build, so fall back to
# it when ffmpeg is not on PATH (otherwise the first capture crashes).
if shutil.which("ffmpeg") is None:
  try:
    import imageio_ffmpeg

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  except Exception:  # pragma: no cover - best effort; mediapy errors later.
    pass


class _WorldSlice:
  """Index-by-original-world-id view over captured per-world rows."""

  def __init__(self, rows: dict[int, torch.Tensor]) -> None:
    self._rows = rows

  def __getitem__(self, i: int) -> torch.Tensor:
    return self._rows[int(i)]


class _StateSnapshot:
  """Minimal stand-in for ``sim.data`` that OffscreenRenderer.update reads.

  Holds CPU copies of qpos/qvel (and mocap) for just the rendered worlds,
  captured on the training hot path (a few KB per frame), so rendering and
  encoding can happen later on a worker thread.
  """

  def __init__(
    self,
    nworld: int,
    qpos: dict[int, torch.Tensor],
    qvel: dict[int, torch.Tensor],
    mocap_pos: dict[int, torch.Tensor] | None,
    mocap_quat: dict[int, torch.Tensor] | None,
  ) -> None:
    self.nworld = nworld
    self.qpos = _WorldSlice(qpos)
    self.qvel = _WorldSlice(qvel)
    if mocap_pos is not None:
      self.mocap_pos = _WorldSlice(mocap_pos)
    if mocap_quat is not None:
      self.mocap_quat = _WorldSlice(mocap_quat)


class _AsyncCapture:
  """State capture + background render/encode for VideoRecorder.

  The training step only snapshots the rendered worlds' generalized
  coordinates (GPU->CPU copy of a few KB). A persistent daemon thread owns
  the EGL context: it replays snapshots through an OffscreenRenderer,
  encodes incrementally, and moves the finished mp4 into the video folder
  atomically (partial files must never be visible - the W&B logger uploads
  each *.mp4 it finds exactly once).
  """

  def __init__(self, env: ManagerBasedRlEnv, verbose: bool) -> None:
    import copy

    from mjlab.viewer.offscreen_renderer import OffscreenRenderer

    self._verbose = verbose
    unwrapped = env.unwrapped
    self._sim_data = unwrapped.sim.data
    self._nworld = int(unwrapped.num_envs)
    self._nmocap = int(unwrapped.sim.mj_model.nmocap)
    # Own a private copy of the viewer cfg: the worker retargets
    # cfg.env_idx per capture (cohort cycling) and must not mutate the
    # env's shared config.
    self._viewer_cfg = copy.deepcopy(unwrapped.cfg.viewer)
    # Constructing the renderer does no GL work (initialize() does); it is
    # safe here and gives us the neighbor-selection logic. GL init happens
    # lazily on the worker thread, which then owns the context.
    self._renderer = OffscreenRenderer(
      model=unwrapped.sim.mj_model, cfg=self._viewer_cfg, scene=unwrapped.scene
    )
    self._frames: list[_StateSnapshot] = []
    self._queue: queue.Queue = queue.Queue(maxsize=4)
    self._worker = threading.Thread(
      target=self._worker_loop, name="video-recorder", daemon=True
    )
    self._worker.start()
    self.begin_capture(int(self._viewer_cfg.env_idx))

  def begin_capture(self, env_idx: int) -> None:
    """Point the capture at a primary world (plus its render neighbors)."""
    self._env_idx = max(0, min(int(env_idx), self._nworld - 1))
    ids = [self._env_idx] + self._renderer._get_extra_env_ids(
      self._nworld, self._env_idx
    )
    self._ids = ids
    self._ids_tensor = torch.as_tensor(
      ids, dtype=torch.long, device=self._sim_data.qpos.device
    )

  def snapshot(self) -> None:
    """Capture the rendered worlds' state for one frame (hot path)."""
    qpos = self._sim_data.qpos[self._ids_tensor].detach().cpu()
    qvel = self._sim_data.qvel[self._ids_tensor].detach().cpu()
    mocap_pos = mocap_quat = None
    if self._nmocap > 0:
      mocap_pos = self._sim_data.mocap_pos[self._ids_tensor].detach().cpu()
      mocap_quat = self._sim_data.mocap_quat[self._ids_tensor].detach().cpu()
    self._frames.append(
      _StateSnapshot(
        self._nworld,
        {w: qpos[k] for k, w in enumerate(self._ids)},
        {w: qvel[k] for k, w in enumerate(self._ids)},
        None
        if mocap_pos is None
        else {w: mocap_pos[k] for k, w in enumerate(self._ids)},
        None
        if mocap_quat is None
        else {w: mocap_quat[k] for k, w in enumerate(self._ids)},
      )
    )

  def frame_count(self) -> int:
    return len(self._frames)

  def submit(self, path: Path, fps: float) -> None:
    """Hand the buffered trajectory to the worker and reset the buffer."""
    frames, self._frames = self._frames, []
    if not frames:
      return
    try:
      self._queue.put_nowait((frames, path, fps, self._env_idx))
    except queue.Full:
      if self._verbose:
        print(f"[WARN] Video encoder backlog; dropping capture {path.name}.")

  def close(self) -> None:
    self._frames = []
    self._queue.put(None)
    self._worker.join(timeout=120.0)

  def _worker_loop(self) -> None:
    initialized = False
    while True:
      job = self._queue.get()
      if job is None:
        break
      frames, path, fps, env_idx = job
      try:
        if not initialized:
          self._renderer.initialize()
          initialized = True
        # Retarget the renderer at this capture's primary world (jobs run
        # sequentially; the cfg copy is private to this capture pipeline).
        self._viewer_cfg.env_idx = env_idx
        self._encode(frames, path, fps)
        if self._verbose:
          print(f"[INFO] Saved video to {path}")
      except Exception as exc:  # Visualization must never take down training.
        print(f"[WARN] Video render/encode failed for {path.name}: {exc!r}")
    if initialized:
      try:
        self._renderer.close()
      except Exception:
        pass

  def _encode(self, frames: list[_StateSnapshot], path: Path, fps: float) -> None:
    # Encode outside the video folder, then move: the rsl-rl W&B logger
    # rglobs *.mp4 under the log dir every iteration and uploads each file
    # it sees exactly once, so a half-written file must never be visible.
    with tempfile.TemporaryDirectory(prefix="mjlab-video-") as tmp:
      tmp_path = Path(tmp) / path.name
      writer: media.VideoWriter | None = None
      try:
        for snap in frames:
          self._renderer.update(snap)
          frame = np.asarray(self._renderer.render())
          if writer is None:
            writer = media.VideoWriter(str(tmp_path), shape=frame.shape[:2], fps=fps)
            writer.__enter__()
          writer.add_image(frame)
      finally:
        if writer is not None:
          writer.__exit__(None, None, None)
      if writer is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), str(path))


class VideoRecorder(ManagerBasedRlEnv):
  """Wraps an environment to record video during interaction.

  A minimal wrapper that records frames as the environment steps.
  Delegates all attribute access and method calls to the wrapped environment.

  For vectorized environments, only records the first environment (index 0) and
  tracks its episode boundaries. This matches gymnasium's RecordVideo behavior.

  Recording is asynchronous by default: the training step only snapshots the
  rendered worlds' state (a few KB), and a background thread does the actual
  rendering and mp4 encoding - the GPU-bound training loop pays essentially
  nothing. If the async capture cannot be constructed (e.g. a mock env in
  tests), the recorder falls back to the synchronous ``env.render()`` path.

  Note: Unlike gymnasium's RecordVideo, this wrapper allows both episode_trigger
  and step_trigger to be used simultaneously. If both are provided, recording will
  start when either trigger fires. The filename will reflect which trigger started
  the recording (e.g., "rl-video-step-1000.mp4" or "rl-video-episode-5.mp4").

  Args:
      env: The environment to wrap and record.
      video_folder: Directory to save videos to.
      episode_trigger: Callable that returns True if should record this episode.
          Receives the actual episode count (increments when env[0] episodes end).
      step_trigger: Callable that returns True if should record this step.
          Receives the global step count.
      video_length: Maximum frames per video. If None, records until env[0] episode ends.
          If set, records exactly that many frames regardless of episode boundaries.
      name_prefix: Prefix for video filenames.
      disable_logger: Whether to disable logging.
      async_render: Render/encode on a background thread (default). Set False
          to force the synchronous ``env.render()`` capture path.
      cycle_env_ids: Optional world indices to cycle the primary rendered
          robot through, one per capture (e.g. ``(0, -1)`` alternates the
          first and last env - for NUgus, a push-cohort and a clean-cohort
          robot). Negative indices count from the end. The env id is
          appended to the filename. Requires async rendering; ignored on
          the sync fallback.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    video_folder: str | Path,
    episode_trigger: Callable[[int], bool] | None = None,
    step_trigger: Callable[[int], bool] | None = None,
    video_length: int | None = None,
    name_prefix: str = "rl-video",
    disable_logger: bool = False,
    async_render: bool = True,
    cycle_env_ids: tuple[int, ...] | list[int] | None = None,
  ):
    # Don't call super().__init__() - we're wrapping an existing env.
    self._wrapped_env = env
    self.video_folder = Path(video_folder)
    self.video_folder.mkdir(parents=True, exist_ok=True)

    self.episode_trigger = episode_trigger
    self.step_trigger = step_trigger
    self.video_length = video_length
    self.name_prefix = name_prefix
    self.disable_logger = disable_logger
    self._async_requested = async_render
    self._async: _AsyncCapture | None = None
    self.cycle_env_ids = tuple(cycle_env_ids) if cycle_env_ids else None

    self.step_count: int = 0
    self.episode_count: int = 0  # Tracks actual episodes
    self.video_count: int = 0  # Tracks completed videos
    self.is_recording: bool = False
    self.current_video_frames: list[np.ndarray] = []
    self.current_video_path: Path | None = None
    self.trigger_type: Literal["step", "episode"] | None = None

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to wrapped environment."""
    return getattr(self._wrapped_env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    """Get the unwrapped environment."""
    return self._wrapped_env.unwrapped

  def reset(self, **kwargs: Any) -> Any:
    """Reset the environment."""
    return self._wrapped_env.reset(**kwargs)

  def step(self, action: torch.Tensor) -> Any:
    """Step the environment and optionally record video.

    Args:
        action: Action tensor.

    Returns:
        Tuple of (obs, reward, terminated, truncated, info) from env.step().
    """
    # Check if we should start recording.
    step_triggered = self.step_trigger is not None and self.step_trigger(
      self.step_count
    )
    episode_triggered = self.episode_trigger is not None and self.episode_trigger(
      self.episode_count
    )

    if (step_triggered or episode_triggered) and not self.is_recording:
      # Track which trigger started the recording for filename generation
      if step_triggered:
        self.trigger_type = "step"
      else:
        self.trigger_type = "episode"
      self._start_recording()

    # Step the environment.
    obs, reward, terminated, truncated, info = self._wrapped_env.step(action)

    # Track episode boundaries (only for the first environment, which we're recording)
    # This matches gymnasium's behavior for vectorized environments.
    if terminated[0] or truncated[0]:
      self.episode_count += 1

    # Record frame if recording.
    if self.is_recording:
      self._record_frame()

      # Check if we should stop recording.
      # If video_length is set, stop only when reaching that length.
      # If video_length is None, stop when the first environment (being recorded) terminates.
      if self.video_length is not None:
        should_stop = self._recorded_frames() >= self.video_length
      else:
        should_stop = terminated[0] or truncated[0]

      if should_stop:
        self._finish_recording()

    self.step_count += 1

    return obs, reward, terminated, truncated, info

  def render(self) -> np.ndarray | None:
    """Render the environment."""
    return self._wrapped_env.render()

  def close(self) -> None:
    """Close the environment and finalize any open videos."""
    if self.is_recording:
      self._finish_recording()
    if self._async is not None:
      self._async.close()
      self._async = None
    self._wrapped_env.close()

  def _get_async(self) -> _AsyncCapture | None:
    """Build (once) the async capture pipeline; fall back to sync on failure."""
    if not self._async_requested:
      return None
    if self._async is None:
      try:
        self._async = _AsyncCapture(self._wrapped_env, not self.disable_logger)
      except Exception as exc:
        self._async_requested = False
        print(
          "[WARN] Async video capture unavailable "
          f"({exc!r}); falling back to synchronous rendering."
        )
    return self._async

  def _recorded_frames(self) -> int:
    if self._async is not None and self._async_requested:
      return self._async.frame_count()
    return len(self.current_video_frames)

  def _start_recording(self) -> None:
    """Start recording a new video."""
    self.is_recording = True
    self.current_video_frames = []

    # Cohort/world cycling: retarget the async capture at this video's
    # primary env and tag the filename with it.
    env_tag = ""
    capture = self._get_async()
    if capture is not None and self.cycle_env_ids:
      idx = self.cycle_env_ids[self.video_count % len(self.cycle_env_ids)]
      if idx < 0:
        idx += capture._nworld
      capture.begin_capture(idx)
      env_tag = f"-env{capture._env_idx}"

    # Generate video filename based on which trigger started recording.
    assert self.trigger_type is not None, "trigger_type must be set before recording"

    if self.trigger_type == "step":
      video_filename = f"{self.name_prefix}-step-{self.step_count}{env_tag}.mp4"
    elif self.trigger_type == "episode":
      video_filename = f"{self.name_prefix}-episode-{self.episode_count}{env_tag}.mp4"
    else:
      assert_never(self.trigger_type)

    self.current_video_path = self.video_folder / video_filename

    if not self.disable_logger:
      print(f"[INFO] Recording video to {self.current_video_path}")

  def _record_frame(self) -> None:
    """Record a frame from the environment.

    For vectorized environments, only records env[0].
    """
    capture = self._get_async()
    if capture is not None:
      capture.snapshot()
      return
    if self._wrapped_env.render_mode == "rgb_array":
      frame = self._wrapped_env.render()
      if frame is not None:
        # For vectorized envs: frame shape is (num_envs, height, width, 3).
        # Extract the first environment's frame.
        rgb_frame = (
          frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
        )
        self.current_video_frames.append(rgb_frame)

  def _finish_recording(self) -> None:
    """Finish recording and save the video."""
    fps_val = self._wrapped_env.metadata.get("render_fps", 30)
    fps = float(fps_val) if isinstance(fps_val, (int, float)) else 30.0
    if self._async is not None and self._async_requested:
      assert self.current_video_path is not None
      self._async.submit(self.current_video_path, fps)
    elif self.current_video_frames:
      # Convert frames to uint8 format.
      video_frames = []
      for frame in self.current_video_frames:
        frame = np.asarray(frame) if not isinstance(frame, np.ndarray) else frame
        if frame.dtype != np.uint8:
          frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        video_frames.append(frame)

      media.write_video(str(self.current_video_path), video_frames, fps=fps)

      if not self.disable_logger:
        print(f"[INFO] Saved video to {self.current_video_path}")

    self.is_recording = False
    self.current_video_frames = []
    self.current_video_path = None
    self.video_count += 1
    self.trigger_type = None  # Reset trigger type after recording
