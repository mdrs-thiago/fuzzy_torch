"""Control benchmark environments for fuzzy_torch.

Batched, differentiable torch dynamics behind the standard Gymnasium API, so the
same plant serves RL baselines, GPU parameter sweeps and gradient-based controller
tuning.

Install with ``pip install 'torch-fis[envs]'``.
"""
from .core.integrate import INTEGRATORS, ORDERS, euler_step, rk4_step, rollout
from .core.metrics import (
    StepResponseMetrics, disturbance_metrics, integral_metrics, step_response_metrics,
)
from .core.modes import ControlMode
from .core.plant import Plant
from .core.scenarios import Disturbance, Scenario, disturbance_rejection, step_response
from .core.video import Video, compare, record
from .plants.crane import CranePlant
from .plants.drone import QuadrotorPlant

__all__ = [
    'INTEGRATORS', 'ORDERS', 'euler_step', 'rk4_step', 'rollout',
    'StepResponseMetrics', 'step_response_metrics', 'disturbance_metrics',
    'integral_metrics',
    'ControlMode', 'Plant',
    'Scenario', 'Disturbance', 'step_response', 'disturbance_rejection',
    'CranePlant', 'QuadrotorPlant',
    'Video', 'compare', 'record',
    'RaceTrack', 'RacingEnv', 'make_track', 'SuperTuxKartEnv', 'KartRace',
]


def __getattr__(name):
    # gymnasium-dependent pieces are imported lazily so `import fuzzy_torch.envs`
    # works for pure-torch rollouts even without the [envs] extra installed.
    if name in ('FuzzyControlEnv', 'ErrorFeatures'):
        from .core.gym_env import FuzzyControlEnv
        from .core.wrappers import ErrorFeatures
        return {'FuzzyControlEnv': FuzzyControlEnv, 'ErrorFeatures': ErrorFeatures}[name]
    if name in ('RaceTrack', 'RacingEnv', 'make_track'):
        from .racing import RaceTrack, RacingEnv, make_track
        return {'RaceTrack': RaceTrack, 'RacingEnv': RacingEnv, 'make_track': make_track}[name]
    if name in ('SuperTuxKartEnv', 'KartRace'):
        from . import supertuxkart
        return getattr(supertuxkart, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
