# The goalkeeper reuses the velocity task's MDP terms (posture, foot shaping,
# terminations, domain randomisation) and adds the shot command and its rewards.
from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F403
from .handoff import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .shot_command import *  # noqa: F403
