"""Computes the XY distance between foot-site pairs"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

FOOT_SITE_NAMES = ("left_foot", "right_foot")


def build_env() -> ManagerBasedRlEnv:
  cfg = nubots_nugus_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def print_feet_distance(env: ManagerBasedRlEnv) -> None:
  env.reset()

  robot = env.scene["robot"]
  site_ids, _ = robot.find_sites(FOOT_SITE_NAMES)
  foot_pos_w = robot.data.site_pos_w[:, site_ids, :]  # [1, N, 3]

  num_feet = foot_pos_w.shape[1]

  root_quat_w = robot.data.root_link_quat_w

  pair_i, pair_j = torch.triu_indices(num_feet, num_feet, offset=1)
  foot_a = foot_pos_w[:, pair_i, :]
  foot_b = foot_pos_w[:, pair_j, :]

  # Rotate the foot-foot vector into the robot frame to get lateral distance.
  quat_exp = root_quat_w.unsqueeze(1).expand(-1, 1, -1)  # Assuming one env
  delta_b = quat_apply_inverse(quat_exp, foot_a - foot_b)

  y_dist = torch.abs(delta_b[..., 1])
  xy_dist = torch.norm(foot_a[..., :2] - foot_b[..., :2], dim=-1)  # [1, P]
  full_dist = torch.norm(foot_a - foot_b, dim=-1)  # [1, P]

  print("Foot positions (world frame, stand-bent-knees keyframe):")
  for k, name in enumerate(FOOT_SITE_NAMES):
    pos = foot_pos_w[0, k].tolist()
    print(f"  {name}: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")

  print()
  print("Pairwise distances:")
  for p, (i, j) in enumerate(zip(pair_i.tolist(), pair_j.tolist(), strict=True)):
    a, b = FOOT_SITE_NAMES[i], FOOT_SITE_NAMES[j]
    print(
      f"  {a} <-> {b}:  lateral_dist={y_dist[0, p].item():.4f} m"
      f"   xy={xy_dist[0, p].item():.4f} m"
      f"  (3d={full_dist[0, p].item():.4f} m)"
    )


def main() -> None:
  env = build_env()
  try:
    print_feet_distance(env)
  finally:
    if hasattr(env, "close"):
      env.close()


if __name__ == "__main__":
  main()
