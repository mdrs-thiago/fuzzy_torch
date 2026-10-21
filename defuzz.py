import torch
import torch.nn as nn

# Below this total activation mass a fuzzy set is treated as empty: no rule fired.
ACTIVATION_TOL = 1e-9


class Defuzzification(nn.Module):
    """
    Base class for Defuzzification methods.

    Subclasses take the aggregated fuzzy set already evaluated on ``self.universe``
    -- a tensor of shape ``(batch, resolution)`` -- and return one crisp value per
    batch item, shape ``(batch,)``.

    When the aggregated set is empty (no rule fired) there is no meaningful crisp
    value, so ``fill_value`` is returned instead. It defaults to NaN so an empty
    rule base cannot masquerade as a real answer; :class:`FuzzyModule` overrides it
    from its ``no_rule_value`` argument.
    """

    #: whether gradients flow through this method
    differentiable = True

    def __init__(self, universe_min: float, universe_max: float, resolution: int = 100,
                 fill_value: float = float('nan')):
        super().__init__()
        self.universe_min = universe_min
        self.universe_max = universe_max
        self.resolution = resolution
        self.fill_value = fill_value
        # Static grid for numerical integration/operations
        self.register_buffer('universe', torch.linspace(universe_min, universe_max, resolution))

    def mass(self, membership_values: torch.Tensor) -> torch.Tensor:
        """Total activation of the aggregated set, shape ``(batch,)``."""
        return torch.sum(membership_values, dim=-1)

    def _fill_empty(self, value: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """Replace results for empty aggregated sets with ``fill_value``.

        ``value`` must already be finite everywhere -- feeding a NaN into
        ``torch.where`` would make the backward pass NaN even on the live branch.
        """
        return torch.where(mass > ACTIVATION_TOL,
                           value,
                           torch.full_like(value, self.fill_value))

    def forward(self, membership_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Centroid(Defuzzification):
    """Centre of gravity: ``sum(x * u(x)) / sum(u(x))``. Differentiable."""

    def forward(self, membership_values: torch.Tensor) -> torch.Tensor:
        numerator = torch.sum(self.universe * membership_values, dim=-1)
        denominator = self.mass(membership_values)
        # Sanitise before dividing, not after: a NaN produced here would survive
        # into the gradient of the non-empty entries.
        safe = torch.where(denominator > ACTIVATION_TOL,
                           denominator, torch.ones_like(denominator))
        return self._fill_empty(numerator / safe, denominator)


class Bisector(Defuzzification):
    """
    The x that splits the area under the aggregated set into two equal halves.

    Not differentiable: the answer comes from an ``argmin`` over the grid, so no
    gradient reaches the membership values.
    """

    differentiable = False

    def forward(self, membership_values: torch.Tensor) -> torch.Tensor:
        total = self.mass(membership_values)
        cumulative_area = torch.cumsum(membership_values, dim=-1)
        half_area = total.unsqueeze(-1) / 2.0

        # First grid point at or past half the area. argmin over |cum - half| would
        # tie on a symmetric set and take the earlier index, biasing the answer one
        # grid step low.
        idx = (cumulative_area < half_area).sum(dim=-1).clamp_max(self.resolution - 1)
        return self._fill_empty(self.universe[idx], total)


class MeanOfMaximum(Defuzzification):
    """
    Average of the x values holding the maximal membership.

    Not differentiable: the max-plateau mask is a hard threshold.
    """

    differentiable = False

    def forward(self, membership_values: torch.Tensor) -> torch.Tensor:
        total = self.mass(membership_values)
        max_val, _ = torch.max(membership_values, dim=-1, keepdim=True)
        # Mask of the max plateau, tolerant of float noise.
        mask = (membership_values >= (max_val - 1e-5)).to(membership_values.dtype)

        sum_x_max = torch.sum(self.universe * mask, dim=-1)
        count_max = torch.sum(mask, dim=-1).clamp_min(1.0)

        # Without the mass check an all-zero set has every point "maximal" and this
        # would quietly return the midpoint of the universe.
        return self._fill_empty(sum_x_max / count_max, total)
