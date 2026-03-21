import torch
from tests.conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.inference_observation_logger import (
  InferenceObservationTensorboardLogger,
  build_observation_dimension_labels,
  extract_actor_observations,
)


class DummyWriter:
  def __init__(self):
    self.scalars: list[tuple[str, float, int]] = []
    self.histograms: list[tuple[str, torch.Tensor, int]] = []
    self.closed = False

  def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None:
    self.histograms.append((tag, values.clone(), global_step))

  def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
    self.scalars.append((tag, scalar_value, global_step))

  def close(self) -> None:
    self.closed = True


def test_extract_actor_observations_from_mapping() -> None:
  actor_obs = torch.randn(2, 5)
  obs = {"actor": actor_obs, "critic": torch.randn(2, 7)}
  extracted = extract_actor_observations(obs)
  assert torch.equal(extracted, actor_obs)


def test_inference_obs_logger_interval_and_dimension_limit() -> None:
  writer = DummyWriter()
  logger = InferenceObservationTensorboardLogger(
    log_dir="unused",
    enabled=True,
    interval=2,
    max_dims=3,
    env_index=1,
    writer=writer,
  )

  obs = {"actor": torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])}

  logger.log(obs)  # step 0, logs
  logger.log(obs)  # step 1, skipped due to interval
  logger.log(obs)  # step 2, logs

  assert len(writer.histograms) == 2
  assert all(hist[1].numel() == 3 for hist in writer.histograms)
  dim_scalars = [s for s in writer.scalars if "/dim_" in s[0]]
  assert len(dim_scalars) == 2 * 3


def test_inference_obs_logger_close_closes_writer() -> None:
  writer = DummyWriter()
  logger = InferenceObservationTensorboardLogger(
    log_dir="unused", enabled=True, writer=writer
  )
  logger.close()
  assert writer.closed


def test_build_observation_dimension_labels_term_major_order() -> None:
  labels = build_observation_dimension_labels(
    term_names=["base_lin_vel", "joint_pos", "actions"],
    term_dims=[(3,), (2, 2), (1,)],
  )
  assert labels == [
    "base_lin_vel[0]",
    "base_lin_vel[1]",
    "base_lin_vel[2]",
    "joint_pos[0]",
    "joint_pos[1]",
    "joint_pos[2]",
    "joint_pos[3]",
    "actions",
  ]


def test_inference_obs_logger_uses_named_dimension_tags() -> None:
  writer = DummyWriter()
  logger = InferenceObservationTensorboardLogger(
    log_dir="unused",
    enabled=True,
    interval=1,
    max_dims=3,
    env_index=0,
    dim_labels=["base_lin_vel[0]", "base_lin_vel[1]", "base_lin_vel[2]"],
    writer=writer,
  )

  obs = {"actor": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
  logger.log(obs)

  scalar_tags = [tag for tag, _, _ in writer.scalars]
  assert "inference/actor_obs/dim_000_base_lin_vel[0]" in scalar_tags
  assert "inference/actor_obs/dim_001_base_lin_vel[1]" in scalar_tags
  assert "inference/actor_obs/dim_002_base_lin_vel[2]" in scalar_tags


def test_nugus_play_env_actor_label_mapping_matches_runtime_boundaries() -> None:
  import mjlab.tasks  # noqa: F401

  task_id = "Mjlab-Velocity-Flat-Nubots-Nugus"
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    term_names = env.unwrapped.observation_manager.active_terms["actor"]
    term_dims = env.unwrapped.observation_manager.group_obs_term_dim["actor"]
    labels = build_observation_dimension_labels(term_names=term_names, term_dims=term_dims)

    assert term_names, "Expected at least one actor observation term"
    assert len(term_names) == len(term_dims)

    # Validate overall size against runtime actor observation tensor width.
    obs = env.unwrapped.observation_manager.compute(update_history=True)
    actor_obs = extract_actor_observations(obs)
    assert actor_obs.ndim == 2
    assert len(labels) == actor_obs.shape[1]

    # Validate boundary naming for every term segment in term-major order.
    offset = 0
    for term_name, dims in zip(term_names, term_dims, strict=True):
      term_width = int(torch.tensor(dims).prod().item()) if dims else 1
      segment = labels[offset : offset + term_width]
      assert len(segment) == term_width
      if term_width == 1:
        assert segment == [term_name]
      else:
        assert segment[0] == f"{term_name}[0]"
        assert segment[-1] == f"{term_name}[{term_width - 1}]"
      offset += term_width

    assert offset == len(labels)

    # Extra explicit boundary check (first and last labels).
    first_width = int(torch.tensor(term_dims[0]).prod().item()) if term_dims[0] else 1
    last_width = int(torch.tensor(term_dims[-1]).prod().item()) if term_dims[-1] else 1
    expected_first = term_names[0] if first_width == 1 else f"{term_names[0]}[0]"
    expected_last = (
      term_names[-1]
      if last_width == 1
      else f"{term_names[-1]}[{last_width - 1}]"
    )
    assert labels[0] == expected_first
    assert labels[-1] == expected_last
  finally:
    env.close()
