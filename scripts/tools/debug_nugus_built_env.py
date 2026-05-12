"""Built-env diagnostics for the NUgus velocity task.

Builds the flat NUgus env with num_envs=1, then prints three sections useful
for finding training/deployment mismatches that the static-config scripts
can't see (because they need an actually compiled MjModel and a stepped
sim):

  1. Physics dump — default_joint_pos and per-DOF dynamics
     (armature, damping, frictionloss, plus viscous_damping if exposed)
     and actuator forcerange / gain / bias parameters. Run this at the
     pre- and post-merge snapshots and diff the output.

  2. PR #857 delay path sanity check — pushes a fixed non-zero action,
     steps once, and prints data.ctrl. With delay_max_lag=0 the
     post-merge ``apply_delay`` path should be a strict passthrough.
     If step-0 ctrl is zero (the previous "no command" state) and the
     action's effect only appears at step 1, the delay refactor is
     silently introducing a one-substep lag during training.

  3. ``last_action`` semantics — runs one step with a known action,
     then compares ``env.action_manager.action``,
     ``action_term.raw_action``, and the slot of the actor observation
     that the ``actions`` term occupies. Confirms which one feeds the
     policy's previous-action input. If pre- and post-merge return
     different things, the policy learns a different feedback signal
     than the deployment expects.

Usage: ``uv run python scripts/tools/debug_nugus_built_env.py``
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg


def _safe_attr(obj, name):
  return getattr(obj, name, None)


def build_env() -> ManagerBasedRlEnv:
  cfg = nubots_nugus_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def dump_physics(env: ManagerBasedRlEnv) -> None:
  print("=" * 78)
  print("[1] Physics dump (post-mj_compile)")
  print("=" * 78)

  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  # Use native MjModel (numpy) — that's the source of truth post-compile
  # and avoids the Warp-backed torch ModelBridge.
  mj_model = env.sim.mj_model

  default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()

  joint_v_adr = robot.indexing.joint_v_adr.detach().cpu().numpy()

  armature = np.asarray(mj_model.dof_armature)
  damping = np.asarray(mj_model.dof_damping)
  frictionloss = np.asarray(mj_model.dof_frictionloss)

  print(
    f"{'joint':<24s} {'default_qpos':>13s} {'armature':>12s} "
    f"{'damping':>12s} {'frictionloss':>14s}"
  )
  for i, name in enumerate(joint_names):
    vadr = int(joint_v_adr[i])
    print(
      f"{name:<24s} {default_joint_pos[i]:>+13.6f} "
      f"{armature[vadr]:>12.6f} {damping[vadr]:>12.6f} "
      f"{frictionloss[vadr]:>14.6f}"
    )

  # Actuator-level params: forcerange, gainprm, biasprm.
  if mj_model.nu > 0:
    print()
    print("Actuator params (model.ctrl order):")
    forcerange = np.asarray(mj_model.actuator_forcerange)
    gainprm = np.asarray(mj_model.actuator_gainprm)
    biasprm = np.asarray(mj_model.actuator_biasprm)
    actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    print(
      f"{'idx':>3s} {'name':<28s} {'forcerange':>22s} "
      f"{'gainprm[:3]':>26s} {'biasprm[:3]':>26s}"
    )
    for i in range(mj_model.nu):
      fr = f"[{forcerange[i, 0]:+.3f}, {forcerange[i, 1]:+.3f}]"
      gp = f"[{gainprm[i, 0]:+.3f}, {gainprm[i, 1]:+.3f}, {gainprm[i, 2]:+.3f}]"
      bp = f"[{biasprm[i, 0]:+.3f}, {biasprm[i, 1]:+.3f}, {biasprm[i, 2]:+.3f}]"
      print(f"{i:>3d} {actuator_names[i]:<28s} {fr:>22s} {gp:>26s} {bp:>26s}")

  # Optional viscous damping field — only present on newer mujoco builds.
  vd = _safe_attr(mj_model, "dof_viscousdamping") or _safe_attr(
    mj_model, "actuator_viscousdamping"
  )
  if vd is not None:
    print()
    print(
      f"viscous_damping field present: shape={np.asarray(vd).shape} "
      f"nonzero={np.count_nonzero(np.asarray(vd))}"
    )
  else:
    print()
    print("viscous_damping field: not present on this mujoco build")


def test_delay_passthrough(env: ManagerBasedRlEnv) -> None:
  print()
  print("=" * 78)
  print("[2] PR #857 delay path sanity check")
  print("=" * 78)

  env.reset()
  action_dim = env.action_manager.total_action_dim
  device = env.device

  # Push a fixed identifiable action on joint index 0 only.
  action = torch.zeros((env.num_envs, action_dim), device=device)
  action[:, 0] = 1.0

  print(f"Pushing action[0]=1.0 (rest zero), action_dim={action_dim}")

  robot = env.scene["robot"]
  ctrl_ids = robot.indexing.ctrl_ids.detach().cpu().numpy()

  # action[0]=1.0 targets joint_id 0 (left_hip_yaw). With offset =
  # default_joint_pos[0] ≈ 0.0339 and scale ≈ 0.0494, the expected
  # position target after applying is ≈ 0.0833 if delay is passthrough.
  default_q0 = float(robot.data.default_joint_pos[0, 0].item())
  print(f"default_joint_pos[joint_0='left_hip_yaw'] = {default_q0:+.6f}")
  print("  -> action=0 should drive ctrl_target to this value (or near it)")
  print("  -> action[0]=1 should drive ctrl_target above default for that joint")

  ctrl_history = []
  for step in range(3):
    env.step(action)
    # mjwarp.Data ctrl is laid out (num_envs, total_nu). Pull this robot's slice.
    full_ctrl = robot.data.data.ctrl[0].detach().cpu().numpy()
    ctrl = full_ctrl[ctrl_ids]
    ctrl_history.append(ctrl.copy())
    # Find the actuator index that drives joint 0 (left_hip_yaw). Use the
    # actuator name to locate it within ctrl_ids.
    print(f"  step {step}: ctrl (this entity, model.ctrl order) =")
    for j, v in enumerate(ctrl):
      print(f"    [{j:2d}] {v:+.6f}")

  print()
  print(
    "Interpretation: a position actuator's ctrl IS the position target, so "
    "for a constant action it should be identical across steps. With "
    "delay_max_lag=0 the expected ctrl for joint_id=0 is "
    f"{default_q0:+.6f} + scale*1.0 every step. If step 0 differs from "
    "later steps (e.g. equals default_pos exactly, meaning the action "
    "hasn't taken effect yet), apply_delay is introducing a one-substep "
    "buffer that wasn't there pre-merge."
  )
  if np.allclose(ctrl_history[0], ctrl_history[1]) and np.allclose(
    ctrl_history[1], ctrl_history[2]
  ):
    print("Result: ctrl is identical across all 3 steps -> passthrough confirmed.")
  else:
    print(
      "Result: ctrl differs across steps -> investigate further. "
      "Compare step 0 vs later steps with the same diff at the pre-merge commit."
    )


def test_last_action_semantics(env: ManagerBasedRlEnv) -> None:
  print()
  print("=" * 78)
  print("[3] last_action observation semantics")
  print("=" * 78)

  env.reset()
  action_dim = env.action_manager.total_action_dim
  device = env.device

  # Use a distinctive non-uniform action so we can tell raw vs processed apart.
  action = torch.linspace(-0.5, 0.5, steps=action_dim, device=device)
  action = action.unsqueeze(0).expand(env.num_envs, -1).clone()

  obs, *_ = env.step(action)

  am_action = env.action_manager.action[0].detach().cpu().numpy()
  term = env.action_manager.get_term("joint_pos")
  raw_action = term.raw_action[0].detach().cpu().numpy()
  processed_action = term._processed_actions[0].detach().cpu().numpy()

  # Pull the actor obs and find where the "actions" term landed. For nugus the
  # layout is [ang_vel(3), gravity(3), joint_pos(20), joint_vel(20),
  # actions(20), command(3)] = 69 dims; slice 46:66 covers the actions term.
  actor_obs = obs["actor"]
  if isinstance(actor_obs, torch.Tensor):
    actor_arr = actor_obs[0].detach().cpu().numpy()
  else:
    # Some configs return dict-of-tensors.
    actor_arr = actor_obs["actions"][0].detach().cpu().numpy()
    actor_arr = np.concatenate([np.zeros(46), actor_arr, np.zeros(3)])
  obs_actions_slot = actor_arr[46:66]

  pushed = action[0].detach().cpu().numpy()
  print(f"pushed action (first 5):                  {pushed[:5]}")
  print(f"action_manager.action (first 5):          {am_action[:5]}")
  print(f"action_term.raw_action (first 5):         {raw_action[:5]}")
  print(f"action_term._processed_actions (first 5): {processed_action[:5]}")
  print(f"actor obs 'actions' slot (first 5):       {obs_actions_slot[:5]}")

  matches_raw = np.allclose(obs_actions_slot, raw_action, atol=1e-5)
  matches_am = np.allclose(obs_actions_slot, am_action, atol=1e-5)
  matches_proc = np.allclose(obs_actions_slot, processed_action, atol=1e-5)
  print()
  print(f"obs 'actions' slot == raw_action?               {matches_raw}")
  print(f"obs 'actions' slot == action_manager.action?    {matches_am}")
  print(f"obs 'actions' slot == _processed_actions?       {matches_proc}")
  if matches_raw and not matches_proc:
    print(
      "=> last_action returns RAW policy output (pre-scale/offset). This is "
      "what PR #712 documented as the contract — what the deployment should "
      "feed back to the policy as the previous-action obs."
    )
  elif matches_proc and not matches_raw:
    print(
      "=> last_action returns PROCESSED actions (post-scale/offset). If this "
      "is post-merge but pre-merge returned raw, the policy is being fed a "
      "different feedback signal during training than the deployment loop "
      "provides."
    )
  elif matches_raw and matches_proc:
    print(
      "=> raw == processed for this step (action and offset/scale gave the "
      "same numeric values). Run with a stronger push to disambiguate."
    )
  else:
    print("=> obs 'actions' slot matches neither — investigate further.")


def main() -> None:
  env = build_env()
  try:
    dump_physics(env)
    test_delay_passthrough(env)
    test_last_action_semantics(env)
  finally:
    if hasattr(env, "close"):
      env.close()


if __name__ == "__main__":
  main()
