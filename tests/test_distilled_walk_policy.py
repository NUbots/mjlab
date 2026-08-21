"""Tests for loading NUbots' distilled walk policy into torch.

The module reads an ONNX file and replays it with torch operators, so the thing
worth testing is that the two agree: once against onnxruntime on arbitrary
inputs, and once against the data the policy was fit to.

``distilled_walk_episode_golden.csv`` is the first two hundred control steps of
``recordings/walk_data/episode_00000.bin`` from the NUbots tree, written out
verbatim -- forty-six observation columns then twelve target columns, the layout
``metadata.yaml`` describes. Regenerate it with::

  np.fromfile(path, dtype=np.float32).reshape(-1, 58)[:200]
"""

import numpy as np
import pytest
import torch

from mjlab.controllers.distilled_walk import DEFAULT_POLICY_PATH, DistilledWalkPolicy

GOLDEN = "distilled_walk_episode_golden.csv"


@pytest.fixture
def policy() -> DistilledWalkPolicy:
  return DistilledWalkPolicy.from_onnx(DEFAULT_POLICY_PATH)


def load_golden(fixtures_dir) -> tuple[np.ndarray, np.ndarray]:
  """Recorded observations and the engine targets they were labelled with."""
  data = np.loadtxt(fixtures_dir / GOLDEN, delimiter=",", skiprows=1, dtype=np.float32)
  return data[:, :46], data[:, 46:]


def test_torch_replay_matches_onnxruntime(policy):
  """The point of the module: same file, same arithmetic, on a batch."""
  ort = pytest.importorskip("onnxruntime")
  session = ort.InferenceSession(
    str(DEFAULT_POLICY_PATH), providers=["CPUExecutionProvider"]
  )
  obs = np.random.default_rng(0).normal(size=(16, 46)).astype(np.float32)

  expected = session.run(None, {"observation": obs})[0]
  actual = policy(torch.from_numpy(obs)).numpy()

  # Float32 rounding, not a difference in the computation.
  assert np.abs(expected - actual).max() < 1e-6


def test_policy_reproduces_the_targets_it_was_fit_to(policy, fixtures_dir):
  """Teacher forced on recorded data, the copy is within a twentieth of a degree.

  This is the whole chain -- weights, normalisation, layer order -- against
  ground truth the policy never saw during this test, so a transcription error
  anywhere in the loader shows up here rather than as a limp in a simulation.
  """
  observations, targets = load_golden(fixtures_dir)

  predictions = policy(torch.from_numpy(observations)).numpy()

  assert np.abs(predictions - targets).mean() < np.deg2rad(0.05)
  assert np.abs(predictions - targets).max() < np.deg2rad(1.0)


def test_loading_rejects_a_graph_that_is_not_this_policy(tmp_path):
  """A re-export with a different architecture must fail loudly, not walk badly."""
  onnx = pytest.importorskip("onnx")
  model = onnx.load(str(DEFAULT_POLICY_PATH))
  del model.graph.node[-1]  # drop the output denormalisation
  path = tmp_path / "mutated.onnx"
  onnx.save_model(model, str(path), save_as_external_data=False)

  with pytest.raises(ValueError, match="not the distilled walk policy"):
    DistilledWalkPolicy.from_onnx(path)


def test_missing_policy_says_so(tmp_path):
  with pytest.raises(FileNotFoundError):
    DistilledWalkPolicy.from_onnx(tmp_path / "absent.onnx")
