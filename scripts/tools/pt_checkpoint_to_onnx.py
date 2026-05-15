import torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl.runner import MjlabOnPolicyRunner

TASK_ID = "Mjlab-Velocity-Flat-Nubots-Nugus"  # your task
CHECKPOINT = "logs/rsl_rl/nugus_velocity/2026-05-13_17-56-22/model_8500.pt"

env_cfg = load_env_cfg(TASK_ID)
agent_cfg = load_rl_cfg(TASK_ID)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True)

# Export to ONNX (saved next to the .pt file)
from pathlib import Path
ckpt_path = Path(CHECKPOINT)
runner.export_policy_to_onnx(str(ckpt_path.parent), f"{ckpt_path.stem}.onnx")
env.close()
