import itertools
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

EPS = 1e-12


def polynomial_exponents(n_inputs: int, order: int) -> torch.Tensor:
    """
    Exponent table for the full multivariate polynomial of total degree <= `order`.

    Row j is the exponent vector of one monomial, so the table has
    ``comb(n_inputs + order, order)`` rows. Cross terms are included:
    for two inputs at order 2 the monomials are
    ``1, x0, x1, x0^2, x0*x1, x1^2``.
    """
    if order < 0:
        raise ValueError(f"TSK order must be >= 0, got {order}")
    rows: List[List[int]] = []
    for degree in range(order + 1):
        for combo in itertools.combinations_with_replacement(range(n_inputs), degree):
            exps = [0] * n_inputs
            for i in combo:
                exps[i] += 1
            rows.append(exps)
    return torch.tensor(rows, dtype=torch.long)


class Consequent(nn.Module):
    """
    Base class for functional (Sugeno/TSK) rule consequents.

    A consequent maps the crisp input vector to a crisp scalar per batch item:
    ``forward(X: (batch, n_inputs)) -> (batch,)``. Columns of ``X`` follow
    ``self.input_names``, which :class:`~fuzzy_torch.system.FuzzyModule` binds at
    construction time so a rule cannot silently permute its inputs.

    Subclass this to plug in an arbitrary f(x) (an MLP, a spline, ...).
    """

    def __init__(self):
        super().__init__()
        self.input_names: Optional[List[str]] = None

    @property
    def is_bound(self) -> bool:
        return self.input_names is not None

    def bind(self, input_names: Sequence[str], input_ranges=None) -> None:
        """Called by the owning system to fix the input column order."""
        self.input_names = list(input_names)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TSKConsequent(Consequent):
    """
    Takagi-Sugeno-Kang consequent of arbitrary polynomial order.

    - order 0: ``f(x) = c0``                              (Sugeno constant)
    - order 1: ``f(x) = c0 + c1*x1 + ... + cn*xn``        (classic first-order TSK / ANFIS)
    - order p: every monomial of total degree <= p, cross terms included

    Coefficients are an ``nn.Parameter`` and train by autograd, unlike the
    weighted-least-squares ``TSKConsequent`` in the sibling ``eautomfis`` package.

    Args:
        order: polynomial order p (>= 0).
        input_names: input variable order. May be omitted; ``FuzzyModule`` binds it.
        coefficients: initial coefficients, shape ``(comb(n+p, p),)``. Defaults to zeros.
        normalize: map each input from its variable range onto [-1, 1] before raising
            powers. Strongly recommended for order >= 2: on a [0, 30] universe a raw
            cubic term reaches 27000 and dominates the fit. Requires known ranges.
        input_ranges: sequence of ``(min, max)`` per input. ``FuzzyModule`` supplies
            these from the input ``FuzzyVariable``s when it binds.
    """

    def __init__(self,
                 order: int,
                 input_names: Optional[Sequence[str]] = None,
                 coefficients=None,
                 normalize: bool = True,
                 input_ranges: Optional[Sequence[Sequence[float]]] = None):
        super().__init__()
        if order < 0:
            raise ValueError(f"TSK order must be >= 0, got {order}")
        self.order = int(order)
        self.normalize = bool(normalize)
        self._init_coefficients = coefficients
        self.register_buffer('exponents', torch.zeros((0, 0), dtype=torch.long))
        self.register_buffer('_center', torch.zeros(0))
        self.register_buffer('_half_range', torch.ones(0))
        if input_names is not None:
            self.bind(input_names, input_ranges)

    @staticmethod
    def n_terms_for(n_inputs: int, order: int) -> int:
        """Number of coefficients for `n_inputs` inputs at polynomial `order`."""
        return math.comb(n_inputs + order, order)

    @property
    def n_terms(self) -> int:
        return self.exponents.shape[0]

    def bind(self, input_names: Sequence[str], input_ranges=None) -> None:
        input_names = list(input_names)
        if self.is_bound:
            if input_names != self.input_names:
                raise ValueError(
                    f"TSKConsequent already bound to {self.input_names}, "
                    f"cannot rebind to {input_names}"
                )
            return
        self.input_names = input_names
        n = len(input_names)
        # Bind onto whatever device/dtype the module already lives on, so binding
        # after a .to(...) does not silently drop new tensors back onto the CPU.
        device = self.exponents.device
        dtype = self._half_range.dtype

        self.exponents = polynomial_exponents(n, self.order).to(device)

        if self.normalize:
            if input_ranges is None:
                raise ValueError(
                    "TSKConsequent(normalize=True) needs input_ranges; pass them "
                    "explicitly or let FuzzyModule bind the consequent."
                )
            ranges = torch.as_tensor(
                [[float(lo), float(hi)] for lo, hi in input_ranges],
                dtype=dtype, device=device,
            )
            if ranges.shape[0] != n:
                raise ValueError(
                    f"input_ranges has {ranges.shape[0]} entries but there are {n} inputs"
                )
            self._center = (ranges[:, 1] + ranges[:, 0]) / 2.0
            self._half_range = ((ranges[:, 1] - ranges[:, 0]) / 2.0).abs().clamp_min(EPS)
        else:
            self._center = torch.zeros(n, dtype=dtype, device=device)
            self._half_range = torch.ones(n, dtype=dtype, device=device)

        init = self._init_coefficients
        if init is None:
            coeffs = torch.zeros(self.n_terms, dtype=dtype, device=device)
        else:
            coeffs = torch.as_tensor(init, dtype=dtype, device=device).reshape(-1)
            if coeffs.numel() != self.n_terms:
                raise ValueError(
                    f"order-{self.order} consequent over {n} inputs needs "
                    f"{self.n_terms} coefficients, got {coeffs.numel()}"
                )
        self.coefficients = nn.Parameter(coeffs)
        self._init_coefficients = None

    def basis(self, X: torch.Tensor) -> torch.Tensor:
        """Monomial design matrix, shape (batch, n_terms)."""
        z = (X - self._center) / self._half_range if self.normalize else X
        # Integer exponents keep pow exact and finite for negative bases;
        # exponent 0 yields 1 with a zero gradient, including at x == 0.
        return torch.pow(z.unsqueeze(1), self.exponents).prod(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.is_bound:
            raise RuntimeError(
                "TSKConsequent is not bound to any inputs; construct it with "
                "input_names or add it to a FuzzyModule first."
            )
        if X.shape[-1] != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} input columns, got {X.shape[-1]}"
            )
        return self.basis(X) @ self.coefficients

    def monomial_names(self) -> List[str]:
        """Human-readable label per coefficient, e.g. ['1', 'Service', 'Service*Food']."""
        names = []
        for row in self.exponents.tolist():
            factors = []
            for name, e in zip(self.input_names, row):
                if e == 1:
                    factors.append(name)
                elif e > 1:
                    factors.append(f"{name}^{e}")
            names.append("*".join(factors) if factors else "1")
        return names

    def __repr__(self):
        if not self.is_bound:
            return f"TSKConsequent(order={self.order}, unbound)"
        terms = " + ".join(
            f"{c:.4g}*{m}" if m != "1" else f"{c:.4g}"
            for c, m in zip(self.coefficients.detach().tolist(), self.monomial_names())
        )
        # The coefficients multiply the *scaled* inputs when normalize=True, so say so
        # rather than let the expression be read as acting on raw values.
        scale = " [inputs rescaled to [-1, 1]]" if self.normalize else ""
        return f"f(x) = {terms}{scale}"
