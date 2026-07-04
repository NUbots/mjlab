"""MDP components for the path tracking task.

Re-exports the velocity-task MDP (rewards, terminations, curriculums,
observations) — the reward structure is shared; only the command term and
the actor's command observation differ.
"""

from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403

from .observations import path_waypoints  # noqa: F401
from .path_command import PathCommand, PathCommandCfg  # noqa: F401
