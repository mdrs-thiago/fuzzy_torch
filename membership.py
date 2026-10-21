import torch
import torch.nn as nn

# Guard for degenerate denominators (b == a, c == b, sigma == 0, ...).
# Small enough not to perturb legitimate values, large enough to keep autograd finite.
EPS = 1e-12


def _ramp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, rising: bool) -> torch.Tensor:
    """
    Linear ramp between `lo` and `hi`, degenerating to a step when hi == lo.

    Uses the double-``torch.where`` idiom: the division is always evaluated (both
    branches are), so the denominator is sanitised *before* dividing. Otherwise the
    unused branch produces inf/NaN and poisons the backward pass.
    """
    den = hi - lo
    ok = den.abs() > EPS
    safe_den = torch.where(ok, den, torch.ones_like(den))
    slope = (x - lo) / safe_den if rising else (hi - x) / safe_den
    step = (x >= hi).to(x.dtype) if rising else (x <= lo).to(x.dtype)
    return torch.where(ok, slope, step)


class MembershipFunction(nn.Module):
    """
    Base class for Membership Functions.

    Contract: ``forward`` maps a tensor of any shape to a tensor of the same shape
    with every element in [0, 1], and stays finite under autograd for all inputs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.as_tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.as_tensor(c, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # trimf(x; a, b, c) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
        # Degenerate shoulders (a == b or b == c) collapse to a step, so
        # trimf(0; 0, 0, 5) is 1 at x = 0 rather than 0.
        left = _ramp(x, self.a, self.b, rising=True)
        right = _ramp(x, self.b, self.c, rising=False)
        return torch.clamp(torch.min(left, right), 0.0, 1.0)


class TrapezoidalMF(MembershipFunction):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.as_tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.as_tensor(c, dtype=torch.float32))
        self.d = nn.Parameter(torch.as_tensor(d, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # trapmf(x; a, b, c, d) = max(min((x-a)/(b-a), 1, (d-x)/(d-c)), 0)
        left = _ramp(x, self.a, self.b, rising=True)
        right = _ramp(x, self.c, self.d, rising=False)
        return torch.clamp(torch.min(left, right), 0.0, 1.0)


class GaussianMF(MembershipFunction):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = nn.Parameter(torch.as_tensor(mu, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.as_tensor(sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # |sigma| so a training step that walks sigma through zero cannot flip the
        # curve inside out; clamp keeps the division finite at sigma == 0.
        sigma = self.sigma.abs().clamp_min(EPS)
        return torch.exp(-0.5 * ((x - self.mu) / sigma) ** 2)


class BellMF(MembershipFunction):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.as_tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.as_tensor(c, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gbellmf(x; a, b, c) = 1 / (1 + |(x-c)/a|^(2b))
        # |z| is clamped away from 0: pow(0, p) has an infinite derivative for p < 1,
        # which makes the gradient NaN at x == c whenever 2b < 1.
        a = self.a.abs().clamp_min(EPS)
        z = ((x - self.c) / a).abs().clamp_min(EPS)
        return 1.0 / (1.0 + torch.pow(z, 2 * self.b))


class SigmoidMF(MembershipFunction):
    def __init__(self, b, c):
        super().__init__()
        self.b = nn.Parameter(torch.as_tensor(b, dtype=torch.float32))  # slope
        self.c = nn.Parameter(torch.as_tensor(c, dtype=torch.float32))  # center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.sigmoid rather than 1/(1+exp(-.)): it does not overflow for large |b*(x-c)|.
        return torch.sigmoid(self.b * (x - self.c))
