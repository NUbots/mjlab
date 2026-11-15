"""Check the mirror maps against the physics.

Two envs: env 1 is put in the mirror image of env 0's state and fed the
mirrored actions. If the maps in ``mdp/symmetry.py`` are right, env 1's
observations stay equal to the mirror of env 0's. Terms with history carry
frames from before the mirroring for the first steps, so ball and command
show a mismatch there that is an artifact of this test.

Run with: uv run python scripts/dribbling/check_symmetry.py
"""

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.dribbling.config.booster_k1.env_cfgs import (
  booster_k1_dribble_env_cfg,
)
from mjlab.tasks.dribbling.mdp.symmetry import _get_mirror


def main() -> None:
  cfg = booster_k1_dribble_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.scene.env_spacing = 0.0
  # Remove everything that differs between envs by chance.
  for name in ("foot_friction", "encoder_bias", "base_com", "pd_gains"):
    cfg.events.pop(name, None)
  cfg.events.pop("ball_friction", None)
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = replace(
    robot_cfg.articulation,
    actuators=tuple(
      replace(a, delay_min_lag=0, delay_max_lag=0)
      for a in robot_cfg.articulation.actuators
    ),
  )
  for group in cfg.observations.values():
    for term in group.terms.values():
      term.delay_min_lag = 0
      term.delay_max_lag = 0

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  mirror = _get_mirror(env)
  robot, ball = env.scene["robot"], env.scene["ball"]
  cmd = env.command_manager.get_term("ball_velocity")
  one = torch.tensor([1], device=env.device)
  vec = torch.tensor([1.0, -1.0, 1.0], device=env.device)
  axial = torch.tensor([-1.0, 1.0, -1.0], device=env.device)
  quat_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)

  def mirror_root(entity) -> None:
    data = entity.data
    q = data.data.qpos[0, data.indexing.free_joint_q_adr]
    v = data.data.qvel[0, data.indexing.free_joint_v_adr]
    pose = torch.cat([q[:3] * vec, q[3:7] * quat_sign]).unsqueeze(0)
    # qvel holds world linear velocity and body-frame angular velocity; both
    # mirror by the same component rules.
    data.data.qpos[1, data.indexing.free_joint_q_adr] = pose[0]
    data.data.qvel[1, data.indexing.free_joint_v_adr] = torch.cat(
      [v[:3] * vec, v[3:] * axial]
    )

  # Let env 0 move for a while under random actions, then mirror it into env 1.
  torch.manual_seed(0)
  n_act = env.action_manager.total_action_dim
  for _ in range(15):
    a = 0.3 * torch.randn(1, n_act, device=env.device)
    env.step(torch.cat([a, mirror.mirror_actions(a)], dim=0))
  env.reset()
  mirror_root(robot)
  mirror_root(ball)
  jp = robot.data.joint_pos[0:1][:, mirror.joint_perm] * mirror.joint_sign
  jv = robot.data.joint_vel[0:1][:, mirror.joint_perm] * mirror.joint_sign
  robot.write_joint_state_to_sim(jp, jv, env_ids=one)
  cmd.vel_command_w[1] = cmd.vel_command_w[0] * vec

  worst: dict[str, float] = {}
  for step in range(12):
    a = 0.3 * torch.randn(1, n_act, device=env.device)
    obs = env.step(torch.cat([a, mirror.mirror_actions(a)], dim=0))[0]
    for group in ("actor", "critic"):
      expected = mirror.mirror_group(group, obs[group][0:1])
      start = 0
      for name, dims in mirror.layout[group]:
        err = (
          expected[0, start : start + dims[0]] - obs[group][1, start : start + dims[0]]
        )
        key = f"{group}/{name}"
        if step < 6:
          worst[key] = max(worst.get(key, 0.0), err.abs().max().item())
        start += dims[0]
  print("max |mirror(obs env0) - obs env1| over the first 6 steps:")
  for key, value in worst.items():
    flag = "" if value < 2e-2 else "   <-- MISMATCH"
    print(f"  {key:28s} {value:.2e}{flag}")
  env.close()


if __name__ == "__main__":
  main()
