"""Classical baselines: PID and linear state feedback.

These exist to answer two questions:

1. *Does the plant behave?* A controller that drives the plant to its setpoint with
   the response theory predicts is evidence the dynamics are wired correctly --
   complementary to the analytical invariants, which test the open-loop physics.
2. *Is the fuzzy controller any good?* This is the reference it has to beat.

Both operate on the ``ErrorFeatures`` observation ``[e_0, e_dot_0, e_1, e_dot_1, ...]``.
"""
from typing import Optional, Sequence

import numpy as np

from .base import Controller


class PID(Controller):
    """
    Single-output PID on one error channel.

    Uses the *measured* error derivative supplied by ``ErrorFeatures`` rather than
    differencing the error itself, which avoids the derivative kick when the
    setpoint steps.

    Args:
        kp, ki, kd: gains.
        channel: which tracked state to act on (index into ``tracked_states``).
        dt: timestep, needed for the integral term.
        output_limit: symmetric actuator limit, used for clamping and anti-windup.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 channel: int = 0, dt: float = 0.01,
                 output_limit: Optional[float] = None):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.channel = int(channel)
        self.dt = float(dt)
        self.output_limit = output_limit
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        error = float(obs[2 * self.channel])
        error_dot = float(obs[2 * self.channel + 1])

        unsaturated = self.kp * error + self.ki * (self._integral + error * self.dt) \
            + self.kd * error_dot
        u = unsaturated
        if self.output_limit is not None:
            u = float(np.clip(u, -self.output_limit, self.output_limit))

        # Conditional integration: stop accumulating when saturated and the error
        # would push further into the limit, so the integrator cannot wind up.
        if self.output_limit is None or u == unsaturated or (u * error) <= 0.0:
            self._integral += error * self.dt

        return np.array([u], dtype=np.float32)


class StateFeedback(Controller):
    """
    Linear feedback on the whole error-feature vector: ``u = -K e``, one row per actuator.

    A plain PID on position cannot damp crane sway -- the sway states are not in its
    error channel. This is the honest classical baseline for an underactuated plant:
    it uses every measurement the fuzzy controller sees, so neither is handicapped.

    Args:
        gains: ``(n_actuators, n_features)`` or a flat sequence for a single actuator.
        output_limits: per-actuator symmetric limit.
    """

    def __init__(self, gains: Sequence, output_limits: Optional[Sequence[float]] = None):
        K = np.atleast_2d(np.asarray(gains, dtype=np.float64))
        self.K = K
        self.output_limits = None if output_limits is None else np.asarray(
            output_limits, dtype=np.float64)
        if self.output_limits is not None and self.output_limits.shape[0] != K.shape[0]:
            raise ValueError(
                f"gains has {K.shape[0]} actuator rows but output_limits has "
                f"{self.output_limits.shape[0]} entries")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape[0] != self.K.shape[1]:
            raise ValueError(
                f"controller expects {self.K.shape[1]} features, got {obs.shape[0]}")
        u = self.K @ obs
        if self.output_limits is not None:
            u = np.clip(u, -self.output_limits, self.output_limits)
        return u.astype(np.float32)


def crane_baseline(max_force: float = 30.0) -> StateFeedback:
    """
    Hand-tuned baseline for :class:`~fuzzy_torch.envs.plants.crane.CranePlant`.

    Features are ``[x_err, x_err_dot, theta_err, theta_err_dot]``. Position terms
    drive the trolley towards the target; the sway terms carry the *opposite* sign,
    because damping a pendulum means accelerating the trolley toward the swing, not
    away from it.
    """
    return StateFeedback(gains=[[6.0, 8.0, -40.0, -25.0]], output_limits=[max_force])
