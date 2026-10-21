import torch
import torch.nn as nn
from typing import Dict, List, Sequence, Tuple, Union

from .norm import TNorm, Min
from .variable import FuzzyVariable
from .consequent import Consequent

Proposition = Tuple[FuzzyVariable, str]


class FuzzyRule(nn.Module):
    """
    A fuzzy rule.

    Mamdani form:
        IF (v1 IS t1) [AND (v2 IS t2) ...] THEN (output IS t_out)
    TSK / Sugeno form:
        IF (v1 IS t1) [AND (v2 IS t2) ...] THEN output = f(x)

    The antecedent variables are held in an ``nn.ModuleList``, so a rule owns its
    membership-function parameters: ``rule.parameters()``, ``rule.to(device)`` and
    ``rule.state_dict()`` all work on a standalone rule, not just inside a
    :class:`~fuzzy_torch.system.FuzzyModule`.
    """

    def __init__(self,
                 antecedent: Union[Proposition, Sequence[Proposition]],
                 consequent: Union[Proposition, Consequent],
                 t_norm: TNorm = None):
        """
        Args:
            antecedent: ``(FuzzyVariable, term_name)`` or a sequence of them,
                connected by ``t_norm`` (AND).
            consequent: ``(FuzzyVariable, term_name)`` for a Mamdani rule, or a
                :class:`~fuzzy_torch.consequent.Consequent` for a TSK rule.
            t_norm: T-Norm used for the AND connections. Defaults to :class:`Min`.
        """
        super().__init__()

        if isinstance(antecedent, tuple) and len(antecedent) == 2 \
                and isinstance(antecedent[0], FuzzyVariable):
            antecedent = [antecedent]
        antecedent = list(antecedent)
        if not antecedent:
            raise ValueError("a rule needs at least one antecedent proposition")

        ant_vars, ant_terms = [], []
        for prop in antecedent:
            var, term = prop
            if not isinstance(var, FuzzyVariable):
                raise TypeError(f"antecedent expects a FuzzyVariable, got {type(var).__name__}")
            if term not in var.terms:
                raise KeyError(
                    f"variable '{var.name}' has no term '{term}' "
                    f"(available: {list(var.terms.keys())})"
                )
            ant_vars.append(var)
            ant_terms.append(term)

        # ModuleList registers the variables as submodules of this rule.
        self._ant_vars = nn.ModuleList(ant_vars)
        self._ant_terms = ant_terms

        if isinstance(consequent, Consequent):
            self.consequent_fn = consequent
            self._cons_var = None
            self._cons_term = None
        else:
            var, term = consequent
            if not isinstance(var, FuzzyVariable):
                raise TypeError(
                    f"consequent expects a FuzzyVariable or a Consequent, got {type(var).__name__}"
                )
            if term not in var.terms:
                raise KeyError(
                    f"output variable '{var.name}' has no term '{term}' "
                    f"(available: {list(var.terms.keys())})"
                )
            self.consequent_fn = None
            self._cons_var = var
            self._cons_term = term

        self.t_norm = t_norm if t_norm else Min()

    @property
    def antecedent(self) -> List[Proposition]:
        """The antecedent as the original ``[(FuzzyVariable, term_name), ...]`` list."""
        return list(zip(self._ant_vars, self._ant_terms))

    @property
    def consequent(self) -> Union[Proposition, Consequent]:
        """``(FuzzyVariable, term_name)`` for a Mamdani rule, else the ``Consequent``."""
        if self.is_tsk:
            return self.consequent_fn
        return (self._cons_var, self._cons_term)

    @property
    def is_tsk(self) -> bool:
        return self.consequent_fn is not None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Firing strength of the rule.

        inputs: {variable name: input tensor}. The returned tensor has the shape of
        the input tensors.
        """
        firing_strength = None
        for var, term in zip(self._ant_vars, self._ant_terms):
            if var.name not in inputs:
                raise ValueError(f"Input for variable '{var.name}' not provided.")

            membership = var.terms[term](inputs[var.name])

            if firing_strength is None:
                firing_strength = membership
            else:
                firing_strength = self.t_norm(firing_strength, membership)

        return firing_strength

    def __repr__(self):
        ant_str = " AND ".join(f"({var.name} IS '{term}')"
                               for var, term in zip(self._ant_vars, self._ant_terms))
        if self.is_tsk:
            cons_str = repr(self.consequent_fn)
        else:
            cons_str = f"({self._cons_var.name} IS '{self._cons_term}')"
        return f"IF {ant_str} THEN {cons_str}"
