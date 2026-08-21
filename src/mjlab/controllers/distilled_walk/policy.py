"""NUbots' distilled walk policy, lifted out of its ONNX export into torch.

The deployed artifact is an ONNX file: ``module/skill/NeuralWalk`` loads it
through OpenVINO and runs it on one robot at a time. Evaluation here needs the
same arithmetic on a batch of a few thousand, on the GPU, which is what torch is
for -- so the weights are read out of the export and replayed by an equivalent
module rather than the file being handed to a runtime.

The module is deliberately shaped like ``tools/walk_distillation/model.py``,
down to the ``net`` attribute and the layer indices, because that is what makes
the transfer a rename rather than a translation: the exporter writes its
parameters out as ``policy.net.<i>.<weight|bias>``, so dropping the ``policy.``
prefix turns the graph's initialisers straight into a state dict. The four
normalisation constants ``export_onnx.py`` bakes around the network keep their
names, and are buffers here for the same reason.

That is still a re-implementation, and re-implementations drift, so
``test_distilled_walk_policy.py`` pins this module against onnxruntime running
the same file. The graph is a fixed twelve nodes (:data:`_EXPECTED_OPS`) and
:meth:`DistilledWalkPolicy.from_onnx` refuses anything else, so a re-export with
a different architecture fails at load rather than quietly producing a different
gait.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import numpy_helper
from torch import nn

OBS_DIM = 46
"""Width of the observation vector; see :mod:`.controller` for its layout."""

TARGET_DIM = 12
"""Leg joint targets out, ordered as ``JOINT_NAMES``."""

HIDDEN_SIZES: tuple[int, int, int] = (256, 256, 128)
"""Hidden widths, from ``tools/walk_distillation/model.py``."""

DEFAULT_POLICY_PATH = Path(__file__).parent / "data" / "walk_policy.onnx"
"""The policy NUbots deploys, copied verbatim from
``module/skill/NeuralWalk/data/model/walk_policy.onnx`` with its external weight
file folded in, so this is one self-contained file."""

_EXPECTED_OPS: tuple[str, ...] = (
  "Sub",
  "Div",
  "LayerNormalization",
  "Gemm",
  "Elu",
  "Gemm",
  "Elu",
  "Gemm",
  "Elu",
  "Gemm",
  "Mul",
  "Add",
)
"""Node sequence :meth:`DistilledWalkPolicy.from_onnx` knows how to replay:
normalise, the four-layer network, denormalise."""

_DEFAULT_EPSILON = 1e-5
"""ONNX's default for ``LayerNormalization``, used if the node omits it."""


class DistilledWalkPolicy(nn.Module):
  """The distilled walk policy: 46 observations in, 12 joint targets out.

  Takes raw observations and returns raw joint angles in radians -- the dataset
  normalisation is folded in at both ends, as it is in the export.

  Blind by construction. The observation carries the velocity command, the walk
  engine's phase clock and the policy's own three previous outputs -- no
  proprioception, no attitude, no contact. It is a learned copy of a trajectory
  generator, not a feedback controller, and it will happily keep producing a
  gait for a robot lying on the floor.
  """

  obs_mean: torch.Tensor
  obs_std: torch.Tensor
  target_mean: torch.Tensor
  target_std: torch.Tensor

  def __init__(self, epsilon: float = _DEFAULT_EPSILON) -> None:
    super().__init__()
    first, second, third = HIDDEN_SIZES
    self.net = nn.Sequential(
      nn.LayerNorm(OBS_DIM, eps=epsilon),
      nn.Linear(OBS_DIM, first),
      nn.ELU(),
      nn.Linear(first, second),
      nn.ELU(),
      nn.Linear(second, third),
      nn.ELU(),
      nn.Linear(third, TARGET_DIM),
    )
    self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
    self.register_buffer("obs_std", torch.ones(OBS_DIM))
    self.register_buffer("target_mean", torch.zeros(TARGET_DIM))
    self.register_buffer("target_std", torch.ones(TARGET_DIM))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """Map observations to joint targets.

    Args:
      obs: Shape ``(N, 46)`` raw observations.

    Returns:
      Shape ``(N, 12)`` joint position targets in radians.
    """
    normalised = (obs - self.obs_mean) / self.obs_std
    return self.net(normalised) * self.target_std + self.target_mean

  @classmethod
  def from_onnx(
    cls,
    path: Path | str = DEFAULT_POLICY_PATH,
    device: torch.device | str = "cpu",
  ) -> DistilledWalkPolicy:
    """Load the weights of an exported policy.

    Args:
      path: The ``.onnx`` export. Weights held in a sidecar ``.onnx.data`` are
        resolved relative to it, as they are in the NUbots tree.
      device: Device to run inference on.

    Returns:
      The policy, in evaluation mode with gradients off.
    """
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"policy not found: {path}")
    graph = onnx.load(str(path)).graph

    ops = tuple(node.op_type for node in graph.node)
    if ops != _EXPECTED_OPS:
      raise ValueError(
        f"{path} is not the distilled walk policy: expected the node sequence "
        f"{_EXPECTED_OPS}, found {ops}"
      )

    # ``load_state_dict`` is strict, so a renamed or resized parameter is an
    # error here rather than a silently untrained layer.
    policy = cls(epsilon=_layer_norm_epsilon(graph))
    policy.load_state_dict(
      {
        initialiser.name.removeprefix("policy."): torch.from_numpy(
          numpy_helper.to_array(initialiser).copy()
        )
        for initialiser in graph.initializer
      }
    )
    policy.to(device).eval()
    policy.requires_grad_(False)
    return policy


def _layer_norm_epsilon(graph: onnx.GraphProto) -> float:
  """The exported ``LayerNormalization`` epsilon, or the ONNX default."""
  for node in graph.node:
    if node.op_type != "LayerNormalization":
      continue
    for attribute in node.attribute:
      if attribute.name == "epsilon":
        return float(attribute.f)
  return _DEFAULT_EPSILON
