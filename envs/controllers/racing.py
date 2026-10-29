"""Fuzzy drivers for :class:`fuzzy_torch.envs.racing.RacingEnv`.

:class:`FuzzyRaceDriver` is a compact hand-written Mamdani baseline;
:class:`ConfigRaceDriver` builds the same kind of driver from a JSON spec, which is
what the Race Control leaderboard accepts.
"""
from typing import Optional

import numpy as np
import torch

from ... import FuzzyModule, FuzzyRule, FuzzyVariable, TrapezoidalMF, TriangularMF
from ...config import ConfigError, build_system, system_to_config
from .base import Controller

#: Inputs a racing fuzzy system may use: the observation entries by name, plus
#: ``curve``, the magnitude of the look-ahead curvature (direction-free "how sharp").
RACE_INPUTS = ('cross_track_error', 'heading_error', 'speed', 'yaw_rate',
               'lookahead_curvature', 'progress', 'curve')
#: The two fuzzy systems of a race driver, in action order.
RACE_OUTPUTS = ('throttle', 'steering')


def race_inputs(obs) -> dict:
    """Name every quantity a racing fuzzy system may read from one observation."""
    obs = np.asarray(obs, dtype=np.float32)
    values = dict(zip(RACE_INPUTS[:6], obs.tolist()))
    values['curve'] = abs(values['lookahead_curvature'])
    return values


def system_inputs(system: FuzzyModule, values: dict) -> dict:
    """The inputs of one system, saturated at the edges of each variable's range.

    Beyond its range a variable's terms are usually all zero, so no rule would fire:
    a kart pushed far off the road would stop steering instead of steering back.
    """
    inputs = {}
    for name in system.input_order:
        var = system.input_variables[name]
        inputs[name] = torch.tensor(min(max(values[name], var.range_min), var.range_max))
    return inputs


def _act(steering: FuzzyModule, throttle: FuzzyModule, obs) -> np.ndarray:
    values = race_inputs(obs)
    with torch.no_grad():
        steer = steering(system_inputs(steering, values))
        gas = throttle(system_inputs(throttle, values))
    return np.array([float(gas), float(steer)], dtype=np.float32)


def _three_terms(variable, scale: float, limit: float):
    """Three terms with *true* shoulders, so recovery rules never disappear."""
    variable.add_term('right', TrapezoidalMF(-limit, -limit, -scale, 0.0))
    variable.add_term('centre', TriangularMF(-scale, 0.0, scale))
    variable.add_term('left', TrapezoidalMF(0.0, scale, limit, limit))


class FuzzyRaceDriver(Controller):
    """Mamdani steering and speed controllers using the six published race features.

    Steering observes cross-track and heading error.  Throttle observes the magnitude
    of upcoming curvature and current speed.  The remaining observation entries
    (yaw rate and lap progress) are intentionally logged but omitted from this small,
    explainable baseline; they are natural extensions for student rule bases.
    """
    def __init__(self):
        cross = FuzzyVariable('cross_track_error', -1.0, 1.0)
        heading = FuzzyVariable('heading_error', -math_pi, math_pi)
        turn = FuzzyVariable('steering', -1.0, 1.0)
        # Narrow centre terms make the kart start correcting before it drifts far.
        _three_terms(cross, 0.28, 1.0)
        _three_terms(heading, 0.38, math_pi)
        _three_terms(turn, 0.72, 1.0)
        terms = ('right', 'centre', 'left')
        rules = []
        for i, c in enumerate(terms):
            for j, h in enumerate(terms):
                # Heading wants the same turn sign; cross-track error wants its opposite.
                out = terms[int(np.clip(j - i + 1, 0, 2))]
                rules.append(FuzzyRule([(cross, c), (heading, h)], (turn, out)))
        self.steering = FuzzyModule([cross, heading], turn, rules, no_rule_value=0.0).eval()

        curve = FuzzyVariable('curve', 0.0, 1.5)
        speed = FuzzyVariable('speed', 0.0, 4.8)
        throttle = FuzzyVariable('throttle', 0.0, 1.0)
        curve.add_term('straight', TriangularMF(0.0, 0.0, 0.18))
        curve.add_term('medium', TriangularMF(0.08, 0.32, 0.65))
        curve.add_term('tight', TriangularMF(0.35, 1.5, 1.5))
        speed.add_term('slow', TriangularMF(0.0, 0.0, 1.8))
        speed.add_term('medium', TriangularMF(0.8, 2.2, 3.6))
        speed.add_term('fast', TriangularMF(2.5, 4.8, 4.8))
        throttle.add_term('brake', TriangularMF(0.0, 0.0, 0.45))
        throttle.add_term('cruise', TriangularMF(0.2, 0.55, 0.85))
        throttle.add_term('full', TriangularMF(0.6, 1.0, 1.0))
        speed_rules = [
            ('straight', 'slow', 'full'), ('straight', 'medium', 'full'), ('straight', 'fast', 'cruise'),
            ('medium', 'slow', 'full'), ('medium', 'medium', 'cruise'), ('medium', 'fast', 'brake'),
            ('tight', 'slow', 'cruise'), ('tight', 'medium', 'brake'), ('tight', 'fast', 'brake'),
        ]
        self.throttle = FuzzyModule(
            [curve, speed], throttle,
            [FuzzyRule([(curve, c), (speed, s)], (throttle, out)) for c, s, out in speed_rules],
            no_rule_value=0.0).eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return _act(self.steering, self.throttle, obs)

    def to_config(self) -> dict:
        """This driver as a Race Control JSON spec."""
        return {'throttle': system_to_config(self.throttle, digits=4),
                'steering': system_to_config(self.steering, digits=4)}


class ConfigRaceDriver(Controller):
    """A race driver built from ``{"throttle": <fis spec>, "steering": <fis spec>}``.

    Each system follows :func:`fuzzy_torch.config.build_system` and may read any of
    :data:`RACE_INPUTS`.  The action is ``[throttle, steering]``; ``RacingEnv``
    clips both to ``[-1, 1]``.  When no rule fires a system outputs its ``default``,
    or 0 if none is given, so a sparse rule base coasts rather than crashing.

    Pass size limits when the spec comes from an untrusted source.
    """
    def __init__(self, spec: dict, *, max_rules: Optional[int] = None,
                 max_inputs: Optional[int] = None, max_terms: Optional[int] = None):
        if not isinstance(spec, dict):
            raise ConfigError(f"expected an object with {' and '.join(RACE_OUTPUTS)}")
        allowed = set(RACE_OUTPUTS) | {'name', 'description'}
        unknown = set(spec) - allowed
        if unknown:
            raise ConfigError(f"unknown key(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
        systems = {}
        for output in RACE_OUTPUTS:
            if output not in spec:
                raise ConfigError(f"{output}: missing; a driver needs both "
                                  f"{' and '.join(RACE_OUTPUTS)} systems")
            sub = spec[output]
            if isinstance(sub, dict) and 'default' not in sub:
                sub = {**sub, 'default': 0.0}
            systems[output] = build_system(
                sub, allowed_inputs=RACE_INPUTS, output_name=output, path=output,
                max_rules=max_rules, max_inputs=max_inputs, max_terms=max_terms).eval()
        self.throttle, self.steering = systems['throttle'], systems['steering']
        self.spec = spec

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return _act(self.steering, self.throttle, obs)


# Avoid making pi look like a configurable controller parameter in generated docs.
math_pi = float(np.pi)
