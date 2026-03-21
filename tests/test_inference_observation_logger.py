import torch

from mjlab.utils.inference_observation_logger import (
  InferenceObservationTensorboardLogger,
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
