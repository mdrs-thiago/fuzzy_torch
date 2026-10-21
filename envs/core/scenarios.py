"""Reference profiles and disturbances.

The benchmark protocol is deliberately deterministic: every controller on a plant
faces the same reference and the same disturbance, injected at the same instant with
the same magnitude. Without that, "IAE after a disturbance" compares controllers
that never met the same disturbance.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class Disturbance:
    """A scripted force added to the plant's disturbance channel.

    Args:
        time: when it starts [s].
        magnitude: force [N] (or the channel's native unit).
        duration: how long it lasts [s]. ``None`` means an impulse of one timestep.
        channel: index into ``u``; defaults to the plant's disturbance channel.
    """
    time: float
    magnitude: float
    duration: Optional[float] = None
    channel: Optional[int] = None

    def active(self, t: float, dt: float) -> bool:
        if t < self.time - 1e-12:
            return False
        span = self.duration if self.duration is not None else dt
        return t < self.time + span - 1e-12


@dataclass
class Scenario:
    """One benchmark episode: how long, what to track, what gets thrown at it."""

    name: str
    episode_steps: int
    setpoint: Optional[Sequence[float]] = None
    #: Time at which the setpoint steps to ``setpoint``. Before it, ``initial_setpoint``
    #: applies -- this is what makes the step-response metrics well defined.
    step_time: float = 0.0
    initial_setpoint: Optional[Sequence[float]] = None
    disturbances: List[Disturbance] = field(default_factory=list)
    randomize_initial_state: bool = False

    def setpoint_at(self, t: float, default: torch.Tensor) -> torch.Tensor:
        target = default if self.setpoint is None else torch.as_tensor(
            self.setpoint, dtype=default.dtype, device=default.device)
        if t < self.step_time - 1e-12:
            if self.initial_setpoint is None:
                return torch.zeros_like(target)
            return torch.as_tensor(self.initial_setpoint, dtype=default.dtype,
                                   device=default.device)
        return target

    def disturbance_at(self, t: float, dt: float, n_actions: int,
                       default_channel: int) -> Optional[torch.Tensor]:
        """Total scripted disturbance vector at time ``t``, or ``None`` if quiet."""
        active = [d for d in self.disturbances if d.active(t, dt)]
        if not active:
            return None
        vec = torch.zeros(n_actions)
        for d in active:
            vec[default_channel if d.channel is None else d.channel] += d.magnitude
        return vec


def step_response(episode_steps: int, setpoint: Sequence[float],
                  step_time: float = 0.0) -> Scenario:
    """A clean step change in the reference, with no disturbance."""
    return Scenario(name='step_response', episode_steps=episode_steps,
                    setpoint=setpoint, step_time=step_time)


def disturbance_rejection(episode_steps: int, setpoint: Sequence[float],
                          time: float, magnitude: float,
                          duration: Optional[float] = None) -> Scenario:
    """Hold the setpoint, then take a scripted hit at a fixed time and magnitude."""
    return Scenario(name='disturbance_rejection', episode_steps=episode_steps,
                    setpoint=setpoint, step_time=0.0,
                    disturbances=[Disturbance(time=time, magnitude=magnitude,
                                              duration=duration)])
