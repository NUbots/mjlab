"""Tests for video recording with mediapy."""

from pathlib import Path
from unittest.mock import Mock

import mediapy as media
import numpy as np
import torch


def _make_mock_env(num_envs: int = 1):
  """Create a mock environment that produces random RGB frames."""
  env = Mock()
  env.render_mode = "rgb_array"
  env.metadata = {"render_fps": 30}
  env.render.return_value = np.random.randint(0, 255, (num_envs, 64, 64, 3), np.uint8)
  env.step.return_value = (
    torch.zeros(num_envs),  # obs
    torch.zeros(num_envs),  # reward
    torch.zeros(num_envs, dtype=torch.bool),  # terminated
    torch.zeros(num_envs, dtype=torch.bool),  # truncated
    {},  # info
  )
  env.close.return_value = None
  env.unwrapped = env
  return env


def test_step_trigger_writes_video(tmp_path: Path):
  """VideoRecorder writes a readable mp4 when the step trigger fires."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step == 0,
    video_length=5,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(6):
    recorder.step(action)

  recorder.close()

  videos = list(tmp_path.glob("*.mp4"))
  assert len(videos) == 1

  # Verify the file is a valid video readable by mediapy.
  frames = media.read_video(str(videos[0]))
  assert len(frames) == 5
  assert frames[0].shape == (64, 64, 3)


def test_accepts_string_path(tmp_path: Path):
  """VideoRecorder accepts a string path for video_folder."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  folder = str(tmp_path / "vids")
  recorder = VideoRecorder(
    env,
    video_folder=folder,
    step_trigger=lambda step: step == 0,
    video_length=3,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(4):
    recorder.step(action)

  recorder.close()

  assert list(Path(folder).glob("*.mp4"))


class _FakeRenderer:
  """OffscreenRenderer stand-in: no GL, tracks the worlds it was shown."""

  instances: list["_FakeRenderer"] = []

  def __init__(self, model, cfg, scene):
    self.cfg = cfg
    self.initialized = False
    self.rendered_env_ids: list[int] = []
    _FakeRenderer.instances.append(self)

  def _get_extra_env_ids(self, nworld: int, env_idx: int) -> list[int]:
    return [(env_idx + 1) % nworld]

  def initialize(self) -> None:
    self.initialized = True

  def update(self, snap) -> None:
    # Consume the snapshot exactly like the real renderer does.
    env_idx = int(self.cfg.env_idx)
    _ = snap.qpos[env_idx].cpu().numpy()
    _ = snap.qvel[env_idx].cpu().numpy()
    self.rendered_env_ids.append(env_idx)

  def render(self) -> np.ndarray:
    return np.random.randint(0, 255, (32, 32, 3), np.uint8)

  def close(self) -> None:
    pass


def _make_sim_env(num_envs: int = 8):
  """Mock env with real tensors where the async capture needs them."""
  env = _make_mock_env(num_envs)
  env.num_envs = num_envs
  env.sim.data.qpos = torch.randn(num_envs, 7)
  env.sim.data.qvel = torch.randn(num_envs, 6)
  env.sim.mj_model.nmocap = 0
  env.cfg.viewer.env_idx = 0
  return env


def test_async_capture_renders_on_worker(tmp_path: Path, monkeypatch):
  """Async mode: the hot path only snapshots state (env.render is never
  called); the worker thread renders and writes the mp4."""
  import mjlab.viewer.offscreen_renderer as osr
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  _FakeRenderer.instances.clear()
  monkeypatch.setattr(osr, "OffscreenRenderer", _FakeRenderer)

  env = _make_sim_env()
  env.render_mode = None  # async path must not need env.render()
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step == 0,
    video_length=5,
    disable_logger=True,
  )
  action = torch.zeros(1)
  for _ in range(6):
    recorder.step(action)
  recorder.close()  # joins the worker

  env.render.assert_not_called()
  videos = list(tmp_path.glob("*.mp4"))
  assert len(videos) == 1
  assert len(media.read_video(str(videos[0]))) == 5


def test_async_capture_cycles_cohort_envs(tmp_path: Path, monkeypatch):
  """cycle_env_ids alternates the primary world per capture (push cohort =
  low indices, clean cohort = high), tags filenames, and the worker renders
  each capture from its own world."""
  import mjlab.viewer.offscreen_renderer as osr
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  _FakeRenderer.instances.clear()
  monkeypatch.setattr(osr, "OffscreenRenderer", _FakeRenderer)

  env = _make_sim_env(num_envs=8)
  env.render_mode = None
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step % 10 == 0,
    video_length=3,
    disable_logger=True,
    cycle_env_ids=(0, -1),
  )
  action = torch.zeros(1)
  for _ in range(20):
    recorder.step(action)
  recorder.close()

  names = sorted(v.name for v in tmp_path.glob("*.mp4"))
  assert names == ["rl-video-step-0-env0.mp4", "rl-video-step-10-env7.mp4"]
  fake = _FakeRenderer.instances[0]
  assert fake.rendered_env_ids == [0, 0, 0, 7, 7, 7]
