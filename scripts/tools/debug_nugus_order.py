"""Print the NUgus joint and action ordering used by mjlab.

Run with `uv run python scripts/tools/debug_nugus_order.py` to inspect the
ordering at any commit. Useful for git-bisecting a deployment-side
left/right swap: compare the printed lists between pre- and post-merge
snapshots to see whether the joint or action joint order shifted.

Prints:
  - entity.joint_names: MuJoCo joint order (what joint_pos / joint_vel
    observations index into when asset_cfg uses the default slice(None)).
  - action joint order: the joint sequence that action[i] targets for the
    velocity task's ``joint_pos`` action term (actuator_names=".*").
  - actuator order: declaration order of mjlab actuators (model.ctrl
    order when sort_actuators=False, which is the default).
"""

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ARTICULATION,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity


def main() -> None:
  entity = Entity(get_nugus_robot_cfg())

  joint_names = list(entity.joint_names)
  print(f"num_joints = {len(joint_names)}")
  print("entity.joint_names (joint_pos / joint_vel observation order):")
  for i, name in enumerate(joint_names):
    print(f"  [{i:2d}] {name}")

  action_joint_ids, action_joint_names = entity.find_joints_by_actuator_names(".*")
  print()
  print(
    "Action joint order (action[i] -> joint, for JointPositionActionCfg"
    ' with actuator_names=(".*",)):'
  )
  for i, (jid, name) in enumerate(zip(action_joint_ids, action_joint_names)):
    print(f"  [{i:2d}] joint_id={jid:2d} {name}")

  print()
  print("Actuator declaration order (model.ctrl order with sort_actuators=False):")
  ctrl_idx = 0
  for group in NUGUS_ARTICULATION.actuators:
    for name in group.target_names_expr:
      print(f"  [{ctrl_idx:2d}] {name}")
      ctrl_idx += 1

  print()
  print("Debugging actuator damping dynamics values")
  for group in NUGUS_ARTICULATION.actuators:
    for name in group.target_names_expr:
      print(f" {name}: armature={group.armature}, stiffness={group.stiffness}, damping={group.damping}, effort_limit={group.effort_limit}, friction_loss={group.frictionloss}")


if __name__ == "__main__":
  main()
