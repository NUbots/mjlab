"""Sim-to-sim validation gate: ONNX policy in vanilla MuJoCo (D1)."""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import tyro
from nugus_eval import (
  COMMAND_GRID,
  DEFAULT_EPISODE_LENGTH_S,
  DEFAULT_SEED,
  EvalMetricsState,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.nugus.mirror_map import MOTOR_JOINT_ORDER
from mjlab.tasks.velocity.config.nugus.runtime_obs import (
  actor_obs_layout_from_env,
  advance_policy_phase,
  build_actor_obs,
  build_servo_params,
  build_vanilla_indices,
  dc_servo_torque,
  projected_gravity_from_data,
)
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Velocity-Flat-Nubots-Nugus"
PHYSICS_DT = 0.005
CONTROL_DT = 0.02
FALL_ANGLE_RAD = np.radians(50.0)
GAIT_PERIOD = 0.7


@dataclass(frozen=True)
class Sim2SimEvalConfig:
  """Configuration for vanilla-MuJoCo sim-to-sim evaluation."""

  onnx_file: str | None = None
  checkpoint_file: str | None = None
  """Optional checkpoint used to locate sibling ONNX export."""
  seed: int = DEFAULT_SEED
  episode_length_s: float = DEFAULT_EPISODE_LENGTH_S
  output_file: str | None = None
  freeze_memory: bool = False
  """Ablation: zero the recurrent hidden state every tick (recurrent
  policies only). Attributes cross-engine deltas to the memory: if the
  frozen policy transfers better, the memory overfits the training
  engine; if worse, the memory was compensating (adaptation working)."""
  kp_scale: float = 1.0
  """Perturbation sweep (backlog 15c/15g): scale all servo kp."""
  kd_scale: float = 1.0
  """Perturbation sweep: scale all servo kd."""
  effort_scale: float = 1.0
  """Perturbation sweep: scale servo saturation/effort limits
  (bus-brownout / hot-servo derating proxy)."""
  torso_mass_scale: float = 1.0
  """Perturbation sweep: scale the torso body mass (payload proxy)."""
  foot_friction: float | None = None
  """Perturbation sweep: override foot geom sliding friction (waxed
  turf proxy). None keeps the model default."""


def _load_onnx_session(cfg: Sim2SimEvalConfig) -> tuple[ort.InferenceSession, Path]:
  if cfg.onnx_file is not None:
    path = Path(cfg.onnx_file)
  elif cfg.checkpoint_file is not None:
    ckpt = Path(cfg.checkpoint_file)
    path = ckpt.parent / f"{ckpt.parent.name}.onnx"
  else:
    raise ValueError("Provide --onnx-file or --checkpoint-file.")

  if not path.exists():
    raise FileNotFoundError(f"ONNX model not found: {path}")
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  return session, path


def _read_onnx_metadata(path: Path) -> dict[str, str]:
  model = onnx.load(str(path))
  return {entry.key: entry.value for entry in model.metadata_props}


def _fallen(data: mujoco.MjData) -> bool:
  gravity_b = projected_gravity_from_data(data)
  tilt = np.arctan2(np.linalg.norm(gravity_b[:2]), -gravity_b[2])
  return bool(tilt > FALL_ANGLE_RAD)


def _root_twist_body(data: mujoco.MjData) -> tuple[np.ndarray, float]:
  quat = data.qpos[3:7]
  lin_w = data.qvel[:3]
  ang_w = data.qvel[3:6]
  w, x, y, z = quat
  q_vec = np.array([x, y, z], dtype=np.float64)

  def quat_inv_apply(vec: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q_vec, vec)
    return vec - w * t + np.cross(q_vec, t)

  lin_b = quat_inv_apply(lin_w)
  ang_b = quat_inv_apply(ang_w)
  return lin_b[:2], float(ang_b[2])


def _copy_reset_state(
  setup_env: ManagerBasedRlEnv,
  mj_model: mujoco.MjModel,
  mj_data: mujoco.MjData,
  seed: int,
) -> np.ndarray:
  setup_env.reset(seed=seed)
  # env.sim.data is the BATCHED warp data ([num_envs, ...]); env.sim.mj_data
  # is an unbatched host mirror whose [0] element is a scalar — indexing it
  # silently broadcast a single float into the whole reset state.
  src = setup_env.sim.data
  mj_data.qpos[:] = np.asarray(src.qpos.numpy(), dtype=np.float64)[0]
  mj_data.qvel[:] = np.asarray(src.qvel.numpy(), dtype=np.float64)[0]
  ctrl = np.asarray(src.ctrl.numpy(), dtype=np.float64)[0]
  mj_data.ctrl[:] = ctrl
  mujoco.mj_forward(mj_model, mj_data)
  return ctrl


def run_sim2sim_eval(cfg: Sim2SimEvalConfig) -> dict[str, object]:
  configure_torch_backends()
  session, onnx_path = _load_onnx_session(cfg)
  input_name = session.get_inputs()[0].name
  # RMA student policies take a flat T-frame history window (time-major,
  # oldest first) of the actor obs vector; the window length rides in the
  # ONNX metadata written at export.
  onnx_meta = _read_onnx_metadata(onnx_path)
  rma_window = int(onnx_meta["history_window"]) if "rma" in onnx_meta else 0
  # Gated students (backlog 15d Stage 2) carry a cross-tick memory: extra
  # graph inputs z_state/evidence, matching outputs to feed back. Detect
  # from the graph itself so the contract can't silently drift.
  graph_inputs = [inp.name for inp in session.get_inputs()]
  gated = "z_state" in graph_inputs and "evidence" in graph_inputs
  gated_z_dim = 0
  gated_out_idx: dict[str, int] = {}
  # Recurrent memory students (v59 line) carry the GRU hidden state:
  # input "h" [1, L*H], output "h_out" fed back each tick, boot zeros.
  recurrent = "h" in graph_inputs
  rnn_h_dim = 0
  rnn_out_idx: dict[str, int] = {}
  if recurrent:
    h_inp = session.get_inputs()[graph_inputs.index("h")]
    rnn_h_dim = int(h_inp.shape[1])
    rnn_names = [out.name for out in session.get_outputs()]
    rnn_out_idx = {name: i for i, name in enumerate(rnn_names)}
    if "h_out" not in rnn_out_idx:
      raise ValueError(f"Recurrent policy missing h_out; outputs: {rnn_names}")
    print(f"[INFO] Recurrent student: carrying hidden state h[{rnn_h_dim}].")
  if gated:
    z_inp = session.get_inputs()[graph_inputs.index("z_state")]
    gated_z_dim = int(z_inp.shape[1])
    out_names = [out.name for out in session.get_outputs()]
    gated_out_idx = {name: i for i, name in enumerate(out_names)}
    if "z_state_out" not in gated_out_idx or "evidence_out" not in gated_out_idx:
      raise ValueError(
        f"Gated policy missing state outputs; graph outputs: {out_names}"
      )
    print(f"[INFO] Gated student: carrying (z_state[{gated_z_dim}], evidence).")

  env_cfg = load_env_cfg(TASK_ID, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = cfg.episode_length_s
  env_cfg.curriculum.clear()
  env_cfg.events.pop("push_robot", None)
  env_cfg.observations["actor"].enable_corruption = False

  setup_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    layout = actor_obs_layout_from_env(setup_env)
    indices = build_vanilla_indices(setup_env, layout)
    servo = build_servo_params(setup_env)
    mj_model = setup_env.sim.mj_model
    action_dim = setup_env.action_manager.total_action_dim
    # Head joints are driven off-policy in training (HEAD_SCRIPTED) and by
    # the vision system on hardware: the policy's head channels are
    # untrained noise, so hold the head at its default pose here.
    head_slots = np.array(
      [
        i
        for i, name in enumerate(MOTOR_JOINT_ORDER)
        if name in ("neck_yaw", "head_pitch")
      ],
      dtype=np.int64,
    )
    # clock_owned: the 21st action channel drives the policy-owned gait
    # phase; replicate PhaseDeltaAction's integration deployment-side.
    phase_cfg = None
    if "phase_delta" in setup_env.action_manager.active_terms:
      pd_cfg = setup_env.action_manager.get_term("phase_delta").cfg
      phase_cfg = {
        "period": float(pd_cfg.period),
        "raw_min": pd_cfg.raw_min,
        "raw_max": pd_cfg.raw_max,
        "command_threshold": float(pd_cfg.command_threshold),
      }
  finally:
    pass

  # Perturbation sweep knobs (backlog 15c/15g): applied to the SERVO
  # emulation and the vanilla model only — the policy is never told.
  if cfg.kp_scale != 1.0:
    servo.kp = servo.kp * cfg.kp_scale
  if cfg.kd_scale != 1.0:
    servo.kd = servo.kd * cfg.kd_scale
  if cfg.effort_scale != 1.0:
    servo.saturation_effort = servo.saturation_effort * cfg.effort_scale
    servo.effort_limit = servo.effort_limit * cfg.effort_scale
  if cfg.torso_mass_scale != 1.0:
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso")
    if torso_id < 0:
      torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
      raise ValueError("torso body not found for torso_mass_scale")
    mj_model.body_mass[torso_id] *= cfg.torso_mass_scale
  if cfg.foot_friction is not None:
    n_set = 0
    for gid in range(mj_model.ngeom):
      name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if "foot" in name:
        mj_model.geom_friction[gid, 0] = cfg.foot_friction
        n_set += 1
    if n_set == 0:
      raise ValueError("no foot geoms found for foot_friction")

  mj_data = mujoco.MjData(mj_model)
  metrics = EvalMetricsState()

  n_joints = len(indices.motor_qpos_adr)
  for cmd_idx, command in enumerate(COMMAND_GRID):
    _copy_reset_state(setup_env, mj_model, mj_data, cfg.seed + cmd_idx)
    last_action = np.zeros(action_dim, dtype=np.float32)
    sim_time = 0.0
    policy_phase = 0.0
    steps = int(cfg.episode_length_s / CONTROL_DT)
    acc = metrics.per_command[cmd_idx]
    fell = False
    ep_len_s = 0.0
    frames: deque[np.ndarray] = deque(maxlen=rma_window or 1)
    # Gated memory boots at zeros (the training-form zero-evidence state).
    z_state = np.zeros((1, gated_z_dim), dtype=np.float32)
    evidence = np.zeros((1, 1), dtype=np.float32)
    # GRU memory boots at zeros (the trained per-reset hidden state).
    h_state = np.zeros((1, rnn_h_dim), dtype=np.float32)

    for _ in range(steps):
      obs = build_actor_obs(
        mj_model,
        mj_data,
        indices,
        command=command,
        last_action=last_action,
        sim_time=sim_time,
        gait_period=GAIT_PERIOD,
        policy_phase=policy_phase if phase_cfg is not None else None,
      )
      if rma_window:
        if not frames:
          # Seed with the first frame (matches training-side CircularBuffer
          # backfill on reset).
          frames.extend([obs] * rma_window)
        else:
          frames.append(obs)
        net_input = np.concatenate(list(frames), axis=0)[None, :]
      else:
        net_input = obs[None, :]
      feeds: dict[str, np.ndarray] = {input_name: net_input}
      if gated:
        feeds["z_state"] = z_state
        feeds["evidence"] = evidence
      if recurrent:
        feeds["h"] = h_state
      outputs = session.run(None, feeds)
      action = np.asarray(outputs[0])[0]
      if gated:
        z_state = np.asarray(outputs[gated_out_idx["z_state_out"]], dtype=np.float32)
        evidence = np.asarray(outputs[gated_out_idx["evidence_out"]], dtype=np.float32)
      if recurrent:
        if cfg.freeze_memory:
          h_state = np.zeros_like(h_state)
        else:
          h_state = np.asarray(outputs[rnn_out_idx["h_out"]], dtype=np.float32)
      last_action = action.astype(np.float32)
      if phase_cfg is not None and action_dim > n_joints:
        policy_phase = advance_policy_phase(
          policy_phase,
          float(last_action[n_joints]),
          command,
          step_dt=CONTROL_DT,
          period=phase_cfg["period"],
          raw_min=phase_cfg["raw_min"],
          raw_max=phase_cfg["raw_max"],
          command_threshold=phase_cfg["command_threshold"],
        )
      # Position targets (JointPositionAction semantics: default pose +
      # scale * action). The compiled actuators are TORQUE motors — the
      # servo PD runs outside MuJoCo (Dynamixel firmware on hardware), so
      # each physics substep computes the DC-motor torque from the held
      # target and writes it to the joint's own ctrl slot (spec order).
      target = indices.default_joint_pos + indices.action_scale * last_action[:n_joints]
      target[head_slots] = indices.default_joint_pos[head_slots]

      lin_b, ang_z = _root_twist_body(mj_data)
      cmd_lin = np.asarray(command[:2], dtype=np.float64)
      lin_err = np.linalg.norm(cmd_lin - lin_b)
      ang_err = abs(command[2] - ang_z)
      acc.lin_sq_sum += lin_err**2
      acc.lin_count += 1
      acc.ang_sq_sum += ang_err**2
      acc.ang_count += 1

      for _ in range(int(CONTROL_DT / PHYSICS_DT)):
        tau = dc_servo_torque(
          mj_data.qpos[indices.motor_qpos_adr],
          mj_data.qvel[indices.motor_qvel_adr],
          target,
          servo,
        )
        mj_data.ctrl[indices.ctrl_adr] = tau
        mujoco.mj_step(mj_model, mj_data)
        sim_time += PHYSICS_DT
        if _fallen(mj_data):
          fell = True
          break
      ep_len_s = sim_time
      if fell:
        break

    acc.ep_len_sum_s += ep_len_s
    acc.ep_count += 1
    if fell:
      acc.fall_count += 1

  setup_env.close()

  payload = {
    "task": TASK_ID,
    "backend": "vanilla_mujoco",
    "onnx": str(onnx_path),
    "seed": cfg.seed,
    "episode_length_s": cfg.episode_length_s,
    "command_grid": [list(cmd) for cmd in COMMAND_GRID],
    **metrics.to_dict(),
  }

  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
      json.dump(payload, handle, indent=2)

  return payload


def main() -> None:
  import mjlab
  import mjlab.tasks  # noqa: F401

  args = tyro.cli(Sim2SimEvalConfig, config=mjlab.TYRO_FLAGS)
  if args.onnx_file is None and args.checkpoint_file is None:
    print("Error: provide --onnx-file or --checkpoint-file.", file=sys.stderr)
    sys.exit(1)

  payload = run_sim2sim_eval(args)
  if args.output_file is None:
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
