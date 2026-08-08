"""The training loop: distributional Q-learning over the discrete action space.

The architecture is Linesight's, which is Ape-X's: several collector processes
each driving their own copy of the physics module, one learner doing gradient
steps against a shared prioritised replay buffer. That split is not a scaling
nicety here -- the module runs at roughly half realtime per instance, so a single
collector produces about eleven transitions a second and the only way to feed a
learner is to run several.

The algorithm is IQN (Dabney et al. 2018): learn the *distribution* of returns
rather than its mean. It is the choice the Trackmania literature converged on,
and the reason is specific to racing -- outcomes here are genuinely bimodal.
Committing to a corner either works or puts you into a wall, and a mean-only
Q-value averages those into a number describing neither. A distributional critic
can represent "usually 90, occasionally 5", which is what the situation is.

Modules:

* :mod:`~polytrack.learning.replay` -- n-step returns and prioritised replay
* :mod:`~polytrack.learning.iqn` -- the network, the quantile loss, the agent
* :mod:`~polytrack.learning.collector` -- worker processes and the actor pool
"""

from __future__ import annotations
