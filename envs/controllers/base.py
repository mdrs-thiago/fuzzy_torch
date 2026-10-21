"""Controller interface shared by the PID, fuzzy and RL baselines.

Every controller consumes the *same* observation -- the output of
:class:`~fuzzy_torch.envs.core.wrappers.ErrorFeatures` -- which is what makes the
comparison between them fair by construction rather than by convention.
"""
from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Maps an observation to an action."""

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return the action for this observation."""

    def reset(self) -> None:
        """Clear any internal state (integrators, filters) between episodes."""

    @property
    def name(self) -> str:
        return type(self).__name__
