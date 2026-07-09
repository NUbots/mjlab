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
  NUGUS_ACTION_SCALE,
  NUGUS_ARTICULATION,
  STAND_BENT_KNEES_KEYFRAME,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


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
  for i, (jid, name) in enumerate(
    zip(action_joint_ids, action_joint_names, strict=True)
  ):
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
      print(
        f" {name}: armature={group.armature}, stiffness={group.stiffness}, damping={group.damping}, effort_limit={group.effort_limit}, frictionloss={group.frictionloss}"
      )
      print(
        f" -> delay_hold_prob={group.delay_hold_prob}, delay_min_lag={group.delay_min_lag}, delay_max_lag={group.delay_max_lag}, delay_per_env_phase={group.delay_per_env_phase}"
      )

  print_action_term_offset_and_scale(entity)


def print_action_term_offset_and_scale(entity: Entity) -> None:
  """Print the per-joint offset and scale used by JointPositionAction.

  Reproduces what ``JointPositionAction`` would compute at env init time
  without needing a full ManagerBasedRlEnv. The action term applies:

      target_pos[i] = action[i] * scale[i] + offset[i]

  where:
    - ``offset[i] = default_joint_pos[target_id[i]]`` when
      ``use_default_offset=True`` (the velocity task default). The
      keyframe dict is resolved against ``entity.joint_names`` via
      ``resolve_expr`` exactly as ``Entity._add_initial_state_keyframe``
      does.
    - ``scale[i]`` is the per-joint value from ``NUGUS_ACTION_SCALE``
      resolved against ``entity.joint_names`` the same way.

  If pre- and post-merge values differ for any joint, the deployment
  side will see a shifted commanded joint target for the same policy
  output — a likely cause of "splits" / overshoot symptoms.
  """
  action_joint_ids, action_joint_names = entity.find_joints_by_actuator_names(".*")

  # default_joint_pos in entity.joint_names order — what env init produces.
  joint_pos_resolved = resolve_expr(
    STAND_BENT_KNEES_KEYFRAME.joint_pos, entity.joint_names, 0.0
  )
  scale_resolved = resolve_expr(NUGUS_ACTION_SCALE, entity.joint_names, 0.0)

  print()
  print(
    "Action term offset / scale (target = action * scale + offset; "
    "offset = default_joint_pos via keyframe, scale = NUGUS_ACTION_SCALE):"
  )
  for i, (jid, name) in enumerate(
    zip(action_joint_ids, action_joint_names, strict=True)
  ):
    offset = joint_pos_resolved[jid]
    scale = scale_resolved[jid]
    print(
      f"  [{i:2d}] joint_id={jid:2d} {name:<24s}"
      f" offset={offset:+.6f}  scale={scale:.6f}"
    )


if __name__ == "__main__":
  main()
