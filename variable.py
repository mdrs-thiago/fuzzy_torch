import torch
import torch.nn as nn
from typing import Dict
from .membership import MembershipFunction

class FuzzyVariable(nn.Module):
    """
    Represents a linguistic variable (e.g., 'Temperature') and its associated fuzzy sets.
    """
    def __init__(self, name: str, range_min: float, range_max: float):
        super().__init__()
        self.name = name
        self.range_min = range_min
        self.range_max = range_max
        self.terms = nn.ModuleDict()  # Holds MembershipFunction objects

    def add_term(self, name: str, mf: MembershipFunction):
        """Adds a linguistic term (e.g., 'Low') with a membership function."""
        self.terms[name] = mf

    def fuzzify(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the membership degree for all terms given an input x.
        Returns a dictionary {term_name: membership_degree_tensor}.
        """
        memberships = {}
        for name, mf in self.terms.items():
            memberships[name] = mf(x)
        return memberships

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.fuzzify(x)

    def plot(self, value=None, ax=None, **kwargs):
        """Draw every term; optionally mark a crisp ``value`` and its degrees.

        See :func:`fuzzy_torch.viz.plot_variable`. Needs matplotlib.
        """
        from .viz import plot_variable
        return plot_variable(self, value=value, ax=ax, **kwargs)

    def __repr__(self):
        return f"FuzzyVariable(name='{self.name}', range=[{self.range_min}, {self.range_max}], terms={list(self.terms.keys())})"
