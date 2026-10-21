"""Classical control metrics.

Gymnasium reports ``reward``; a control engineer wants IAE, overshoot and settling
time. No maintained Gym suite reports these, so we compute them from the recorded
trajectory. Everything here takes a ``(steps,)`` or ``(batch, steps)`` error signal
and is pure torch, so it works on a batched GPU rollout as well as one episode.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch


@dataclass
class StepResponseMetrics:
    """Classical step-response report card. Times in seconds, NaN when undefined."""
    iae: float          # integral of |e|
    itae: float         # integral of t*|e|
    ise: float          # integral of e^2
    overshoot: float    # percent beyond the commanded step, 0 if none
    rise_time: float    # 10% -> 90% of the step
    settling_time: float  # last exit from the +/- tolerance band
    steady_state_error: float
    control_effort: float  # integral of |u|
    peak_control: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _trapz(y: torch.Tensor, dt: float) -> torch.Tensor:
    """Trapezoidal integral along the last axis. More accurate than a plain sum,
    which biases every integral metric by half a sample at each end."""
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)
    return dt * (y[..., :-1] + y[..., 1:]).sum(dim=-1) / 2.0


def integral_metrics(error: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    """IAE, ITAE and ISE of an error signal, integrated by trapezoid."""
    abs_e = error.abs()
    t = torch.arange(error.shape[-1], dtype=error.dtype, device=error.device) * dt
    return {
        'iae': _trapz(abs_e, dt),
        'itae': _trapz(abs_e * t, dt),
        'ise': _trapz(error ** 2, dt),
    }


def step_response_metrics(response: torch.Tensor,
                          target: float,
                          dt: float,
                          initial: float = 0.0,
                          control: Optional[torch.Tensor] = None,
                          settle_tol: float = 0.02) -> StepResponseMetrics:
    """
    Classical metrics for a single (non-batched) step response.

    Args:
        response: tracked output over time, ``(steps,)``.
        target: commanded value after the step.
        dt: timestep.
        initial: value the output started from.
        control: actuator signal, ``(steps,)``, for effort metrics.
        settle_tol: settling band as a fraction of the step size.
    """
    response = response.detach().reshape(-1).double()
    span = target - initial
    error = target - response
    integrals = integral_metrics(error, dt)

    nan = float('nan')
    if abs(span) < 1e-12:
        # A zero-size step has no overshoot or rise time to speak of.
        overshoot, rise_time = 0.0, nan
    else:
        peak = response.max().item() if span > 0 else response.min().item()
        overshoot = max(0.0, (peak - target) / span * 100.0)

        lo = initial + 0.1 * span
        hi = initial + 0.9 * span
        crossed_lo = (response >= lo) if span > 0 else (response <= lo)
        crossed_hi = (response >= hi) if span > 0 else (response <= hi)
        i_lo = int(crossed_lo.nonzero()[0]) if crossed_lo.any() else None
        i_hi = int(crossed_hi.nonzero()[0]) if crossed_hi.any() else None
        rise_time = (i_hi - i_lo) * dt if (i_lo is not None and i_hi is not None
                                           and i_hi >= i_lo) else nan

    band = settle_tol * abs(span) if abs(span) > 1e-12 else settle_tol
    outside = (error.abs() > band)
    # Settling time is the last time it *leaves* the band, not the first time it enters:
    # a response that re-exits later has not settled.
    settling_time = (int(outside.nonzero()[-1]) + 1) * dt if outside.any() else 0.0

    tail = max(1, response.shape[-1] // 20)
    steady_state_error = float(error[-tail:].mean().abs())

    if control is None:
        effort, peak_control = nan, nan
    else:
        c = control.detach().reshape(-1).double()
        effort = float(_trapz(c.abs(), dt))
        peak_control = float(c.abs().max())

    return StepResponseMetrics(
        iae=float(integrals['iae']), itae=float(integrals['itae']),
        ise=float(integrals['ise']), overshoot=float(overshoot),
        rise_time=float(rise_time), settling_time=float(settling_time),
        steady_state_error=steady_state_error,
        control_effort=effort, peak_control=peak_control,
    )


def disturbance_metrics(response: torch.Tensor, target: float, dt: float,
                        disturbance_step: int) -> Dict[str, float]:
    """Peak deviation and recovery behaviour after a scripted disturbance."""
    response = response.detach().reshape(-1).double()
    after = response[disturbance_step:]
    error = target - after
    integrals = integral_metrics(error, dt)
    peak_dev = float(error.abs().max()) if error.numel() else float('nan')

    band = 0.02 * max(abs(target), 1.0)
    outside = error.abs() > band
    recovery = (int(outside.nonzero()[-1]) + 1) * dt if outside.any() else 0.0
    return {
        'peak_deviation': peak_dev,
        'recovery_time': recovery,
        'iae_after': float(integrals['iae']),
        'itae_after': float(integrals['itae']),
    }
