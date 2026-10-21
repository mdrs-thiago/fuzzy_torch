"""Student-editable fuzzy controllers for the crane, kart, and drone labs.

Every ``# STUDENT`` marker is intentionally a design decision rather than missing
simulation plumbing.  The classes have fixed observation/action contracts so an
instructor can swap controllers without changing the experiment.
"""
from itertools import product
import math

import numpy as np
import torch

from fuzzy_torch import FuzzyModule, FuzzyRule, FuzzyVariable, GaussianMF, TSKConsequent


def make_tsk_grid(specs, output_name, output_range):
    """Create a differentiable, complete three-term TSK grid.

    ``specs`` is ``[(name, low, centre, high), ...]``.  Gaussian terms ensure
    every state activates a rule; zero initial consequents make the supplied
    policies safe neutral starting points.  Students can replace this helper with
    explicit Mamdani rules while preserving each class's ``__call__`` signature.
    """
    variables = []
    for name, low, centre, high in specs:
        variable = FuzzyVariable(name, low, high)
        spread = max((high - low) / 3.0, 1e-4)
        for label, mean in (('negative', low), ('zero', centre), ('positive', high)):
            variable.add_term(label, GaussianMF(mean, spread))
        variables.append(variable)
    output = FuzzyVariable(output_name, *output_range)
    rules = []
    for terms in product(('negative', 'zero', 'positive'), repeat=len(variables)):
        antecedent = list(zip(variables, terms))
        consequent = TSKConsequent(order=1, normalize=True)
        rules.append(FuzzyRule(antecedent, consequent))
    system = FuzzyModule(variables, output, rules, no_rule_value=0.0)
    # STUDENT: initialise selected TSK consequents from hand-written rules here.
    return system


def _scalar(system, values):
    inputs = {name: torch.as_tensor(value, dtype=torch.float32)
              for name, value in values.items()}
    return float(system(inputs).detach())


class CraneStarter(torch.nn.Module):
    """One fuzzy force output from the fixed ErrorFeatures observation."""

    def __init__(self):
        super().__init__()
        self.force = make_tsk_grid([
            ('x_error', -10.0, 0.0, 10.0),
            ('x_error_dot', -6.0, 0.0, 6.0),
            ('theta_error', -0.6, 0.0, 0.6),
            ('theta_error_dot', -3.0, 0.0, 3.0),
        ], 'force', (-30.0, 30.0))

    def forward(self, features):
        return self.force({'x_error': features[..., 0], 'x_error_dot': features[..., 1],
                           'theta_error': features[..., 2], 'theta_error_dot': features[..., 3]})

    def __call__(self, observation):
        if torch.is_tensor(observation):
            return super().__call__(observation)
        return np.array([_scalar(self.force, dict(zip(
            ('x_error', 'x_error_dot', 'theta_error', 'theta_error_dot'), observation)))], dtype=np.float32)


class RaceStarter(torch.nn.Module):
    """Separate fuzzy systems for steering and throttle, matching RacingEnv."""

    def __init__(self):
        super().__init__()
        self.steering = make_tsk_grid([
            ('cross_track_error', -1.0, 0.0, 1.0),
            ('heading_error', -math.pi, 0.0, math.pi),
        ], 'steering', (-1.0, 1.0))
        self.throttle = make_tsk_grid([
            ('curve', 0.0, 0.35, 1.5),
            ('speed', 0.0, 2.4, 4.8),
        ], 'throttle', (0.0, 1.0))

    def forward(self, features):
        steer = self.steering({'cross_track_error': features[..., 0], 'heading_error': features[..., 1]})
        throttle = self.throttle({'curve': features[..., 4].abs(), 'speed': features[..., 2]})
        return torch.stack([throttle, steer], dim=-1)

    def __call__(self, observation):
        if torch.is_tensor(observation):
            return super().__call__(observation)
        action = self.forward(torch.as_tensor(observation, dtype=torch.float32))
        return action.detach().cpu().numpy().astype(np.float32)


def _euler(quaternion):
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


class DroneStarter(torch.nn.Module):
    """Four small fuzzy systems producing the rate-and-thrust action.

    This factorised structure is purposeful: students can start with altitude
    control, then tune the lateral roll/pitch systems, without facing an
    impractical 12-input Cartesian rule grid.
    """

    def __init__(self, mass=0.65, gravity=9.81, max_thrust=20.0, max_rate=8.0):
        super().__init__()
        self.hover_thrust, self.max_thrust, self.max_rate = mass * gravity, max_thrust, max_rate
        self.roll = make_tsk_grid([('y_error', -4., 0., 4.), ('y_velocity', -5., 0., 5.)],
                                  'roll_rate', (-max_rate, max_rate))
        self.pitch = make_tsk_grid([('x_error', -4., 0., 4.), ('x_velocity', -5., 0., 5.)],
                                   'pitch_rate', (-max_rate, max_rate))
        self.vertical = make_tsk_grid([('z_error', -3., 0., 3.), ('z_velocity', -4., 0., 4.)],
                                      'thrust_delta', (-max_thrust / 2, max_thrust / 2))
        self.yaw = make_tsk_grid([('yaw_error', -math.pi, 0., math.pi), ('yaw_rate', -max_rate, 0., max_rate)],
                                 'yaw_rate_cmd', (-max_rate, max_rate))

    def forward(self, state, target):
        error, velocity = target[..., :3] - state[..., :3], state[..., 3:6]
        attitude, rates = _euler(state[..., 6:10]), state[..., 10:13]
        roll = self.roll({'y_error': error[..., 1], 'y_velocity': -velocity[..., 1]})
        pitch = self.pitch({'x_error': error[..., 0], 'x_velocity': -velocity[..., 0]})
        vertical = self.vertical({'z_error': error[..., 2], 'z_velocity': -velocity[..., 2]})
        yaw = self.yaw({'yaw_error': -attitude[..., 2], 'yaw_rate': -rates[..., 2]})
        return torch.stack([(self.hover_thrust + vertical).clamp(0., self.max_thrust),
                            roll.clamp(-self.max_rate, self.max_rate),
                            pitch.clamp(-self.max_rate, self.max_rate),
                            yaw.clamp(-self.max_rate, self.max_rate)], dim=-1)

    def act(self, state, target):
        """NumPy action adapter for ``FuzzyControlEnv(QuadrotorPlant(...))``."""
        state = torch.as_tensor(state, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        return self.forward(state, target).detach().cpu().numpy().astype(np.float32)
