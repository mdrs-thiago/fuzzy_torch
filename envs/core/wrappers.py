"""Observation wrappers.

:class:`ErrorFeatures` is the mechanism that makes "the RL agent and the fuzzy
controller see identical inputs" structural rather than a convention someone has to
remember. The plant emits raw physical state; this wrapper derives the engineered
control features, and *both* controllers consume the wrapper's output.
"""
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class ErrorFeatures(gym.ObservationWrapper):
    """
    Replace the raw state with ``[e_i, e_dot_i, ...]`` for each tracked state.

    ``e = setpoint - value``, and ``e_dot`` is the negated derivative state where the
    plant provides one (the derivative of the error, since the setpoint is piecewise
    constant), otherwise a finite difference across the last step.

    Args:
        env: a :class:`~fuzzy_torch.envs.core.gym_env.FuzzyControlEnv`.
        derivative_states: state name supplying each tracked state's rate, or ``None``
            to finite-difference it. Defaults to the plant's declared pairing.
    """

    def __init__(self, env: gym.Env, derivative_states: Optional[List[Optional[str]]] = None):
        super().__init__(env)
        plant = env.unwrapped.plant
        self.tracked = list(plant.tracked_states)

        if derivative_states is None:
            derivative_states = [self._guess_derivative(plant, n) for n in self.tracked]
        self.derivative_states = derivative_states
        self._deriv_idx = [None if d is None else plant.state_index(d)
                           for d in derivative_states]

        self.feature_names: List[str] = []
        for name in self.tracked:
            self.feature_names += [f'{name}_error', f'{name}_error_dot']

        n = len(self.feature_names)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(n,), dtype=np.float32)
        self._prev_error: Optional[np.ndarray] = None

    @staticmethod
    def _guess_derivative(plant, name: str) -> Optional[str]:
        """Pair a tracked state with its rate by the plant's own naming."""
        for candidate in (f'{name}_dot', 'omega' if name == 'theta' else None):
            if candidate and candidate in plant.state_names:
                return candidate
        return None

    def observation(self, observation: np.ndarray) -> np.ndarray:
        env = self.env.unwrapped
        setpoint = env.setpoint().detach().cpu().numpy()
        idx = env.plant.tracked_indices
        error = setpoint - observation[idx]

        feats = []
        for k, d in enumerate(self._deriv_idx):
            if d is not None:
                # d(setpoint - x)/dt = -x_dot for a piecewise-constant setpoint.
                rate = -float(observation[d])
            elif self._prev_error is not None:
                rate = float((error[k] - self._prev_error[k]) / env.dt)
            else:
                rate = 0.0
            feats += [float(error[k]), rate]

        self._prev_error = error.copy()
        return np.asarray(feats, dtype=np.float32)

    def reset(self, **kwargs):
        self._prev_error = None
        return super().reset(**kwargs)
