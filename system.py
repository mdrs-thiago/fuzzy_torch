import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .variable import FuzzyVariable
from .rule import FuzzyRule
from .norm import TNorm, SNorm, Min, Max, ProbabilisticSum
from .defuzz import Defuzzification, Centroid, ACTIVATION_TOL

MAMDANI = 'mamdani'
TSK = 'tsk'


class FuzzyModule(nn.Module):
    """
    Fuzzy Inference System, Mamdani or Takagi-Sugeno-Kang.

    The mode is inferred from the rules: rules with a ``(FuzzyVariable, term)``
    consequent make a Mamdani system, rules with a
    :class:`~fuzzy_torch.consequent.Consequent` make a TSK system. A rule base may
    not mix the two.

    Mamdani inference
        firing strength -> implication against each consequent fuzzy set ->
        aggregation across rules -> defuzzification.

    TSK inference
        firing strength -> per-rule crisp ``f_i(x)`` -> weighted average
        ``sum(w_i * f_i) / sum(w_i)``. No defuzzifier is involved, and the output
        variable needs no terms.

    Args:
        inputs: input :class:`FuzzyVariable`s. Their order fixes the column order
            handed to TSK consequents.
        output: the single output variable (MISO).
        rules: the rule base.
        defuzz_method: Mamdani only; defaults to :class:`Centroid` over the output
            variable's range. Ignored in TSK mode.
        implication: Mamdani implication T-Norm, default :class:`Min` (clipping).
            Pass :class:`Product` for scaling implication.
        aggregation: Mamdani aggregation S-Norm across rules, default :class:`Max`.
        tsk_normalize: TSK only. ``True`` (default) divides by the total firing
            strength (weighted average); ``False`` gives the unnormalised weighted sum.
        no_rule_value: returned for batch items where no rule fires. Defaults to
            NaN so a silent, plausible-looking wrong answer is impossible; set it to
            e.g. the midpoint of the output range if you prefer a fallback.
        chunk_rules: Mamdani only. Number of rules whose implication is evaluated as
            one ``(batch, chunk, resolution)`` block. ``None`` (default) folds them one
            at a time, which measured fastest on CPU at every size tried: the wide
            form is memory-bound, and 400 rules over a batch of 4096 would materialise
            a 1.3 GB intermediate. Raise it on a GPU with memory headroom.
    """

    def __init__(self,
                 inputs: List[FuzzyVariable],
                 output: FuzzyVariable,
                 rules: List[FuzzyRule],
                 defuzz_method: Optional[Defuzzification] = None,
                 implication: Optional[TNorm] = None,
                 aggregation: Optional[SNorm] = None,
                 tsk_normalize: bool = True,
                 no_rule_value: float = float('nan'),
                 chunk_rules: Optional[int] = None):
        super().__init__()
        self._cons_cache: Optional[torch.Tensor] = None

        inputs = list(inputs)
        self.input_variables = nn.ModuleDict({v.name: v for v in inputs})
        if len(self.input_variables) != len(inputs):
            raise ValueError("input variable names must be unique")
        #: input variable names, in the order given -- the TSK consequent column order
        self.input_order: List[str] = [v.name for v in inputs]

        self.output_variable = output
        self.rules = nn.ModuleList(rules)

        self.mode = self._detect_mode(self.rules)
        self.tsk_normalize = bool(tsk_normalize)
        self.no_rule_value = float(no_rule_value)
        self.chunk_rules = chunk_rules

        self.implication = implication if implication is not None else Min()
        self.aggregation = aggregation if aggregation is not None else Max()

        if self.mode == TSK:
            if defuzz_method is not None:
                warnings.warn(
                    "defuzz_method is ignored for a TSK system: TSK consequents are "
                    "already crisp.", stacklevel=2)
            self.defuzz = None
            self._bind_tsk_consequents(inputs)
        else:
            self._validate_mamdani_rules()
            self.defuzz = defuzz_method if defuzz_method else Centroid(
                output.range_min, output.range_max)
            self.defuzz.fill_value = self.no_rule_value
            if not getattr(self.defuzz, 'differentiable', True):
                warnings.warn(
                    f"{type(self.defuzz).__name__} is not differentiable; gradients "
                    "will not reach the membership functions. Use Centroid for training.",
                    stacklevel=2)

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _detect_mode(rules) -> str:
        if len(rules) == 0:
            return MAMDANI
        flags = {bool(r.is_tsk) for r in rules}
        if len(flags) > 1:
            raise ValueError(
                "rule base mixes Mamdani and TSK consequents; a system must be "
                "entirely one or the other")
        return TSK if flags.pop() else MAMDANI

    def _validate_mamdani_rules(self) -> None:
        for i, rule in enumerate(self.rules):
            var, term = rule.consequent
            if var is not self.output_variable:
                raise ValueError(
                    f"rule {i + 1} has consequent variable '{var.name}' but the "
                    f"system's output variable is '{self.output_variable.name}'; "
                    "only single-output (MISO) systems are supported")
            if term not in self.output_variable.terms:
                raise KeyError(
                    f"rule {i + 1}: output variable '{var.name}' has no term '{term}'")

    def _bind_tsk_consequents(self, inputs: List[FuzzyVariable]) -> None:
        ranges = [(v.range_min, v.range_max) for v in inputs]
        for i, rule in enumerate(self.rules):
            cons = rule.consequent_fn
            if cons.is_bound:
                if list(cons.input_names) != self.input_order:
                    raise ValueError(
                        f"rule {i + 1}: consequent is bound to inputs "
                        f"{list(cons.input_names)} but the system's input order is "
                        f"{self.input_order}")
            else:
                cons.bind(self.input_order, ranges)

    # ------------------------------------------------------------- input prep

    def _prepare_inputs(self, x: Dict[str, torch.Tensor]):
        """
        Normalise the input dict to 1-D tensors of a common batch size.

        Accepts 0-dim scalars, ``(B,)`` and ``(B, 1)`` tensors, and broadcasts
        length-1 entries against the batch. Returns
        ``(prepared, batch_size, was_scalar)``; ``was_scalar`` is True only when
        every input was 0-dim, in which case the caller squeezes the output back.
        """
        if not x:
            raise ValueError("no inputs provided")

        prepared: Dict[str, torch.Tensor] = {}
        was_scalar = True
        for name, value in x.items():
            t = value if torch.is_tensor(value) else torch.as_tensor(
                value, dtype=torch.float32, device=self._device())
            if t.dim() == 0:
                t = t.reshape(1)
            else:
                was_scalar = False
                if t.dim() > 1:
                    if t.shape[1:].numel() != 1:
                        raise ValueError(
                            f"input '{name}' has shape {tuple(t.shape)}; expected a "
                            "scalar, (batch,) or (batch, 1) tensor")
                    t = t.reshape(-1)
            prepared[name] = t

        sizes = {t.shape[0] for t in prepared.values()}
        batch_size = max(sizes)
        if sizes - {1, batch_size}:
            raise ValueError(
                f"inconsistent batch sizes across inputs: "
                f"{ {k: v.shape[0] for k, v in prepared.items()} }")
        if len(sizes) > 1:
            prepared = {k: (v.expand(batch_size) if v.shape[0] == 1 else v)
                        for k, v in prepared.items()}

        return prepared, batch_size, was_scalar

    def _device(self) -> torch.device:
        for t in self.parameters():
            return t.device
        for t in self.buffers():
            return t.device
        return torch.device('cpu')

    # ---------------------------------------------------------------- forward

    def get_rule_activations(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Firing strength of every rule, shape ``(batch, n_rules)``."""
        prepared, batch_size, _ = self._prepare_inputs(x)
        return self._activations(prepared, batch_size)

    def _activations(self, prepared: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if len(self.rules) == 0:
            return torch.zeros((batch_size, 0), device=self._device())
        return torch.stack([rule(prepared) for rule in self.rules], dim=1)

    def _consequent_matrix(self) -> torch.Tensor:
        """Consequent MFs evaluated on the defuzzification grid, shape ``(n_rules, resolution)``."""
        if self._cons_cache is not None:
            return self._cons_cache
        universe = self.defuzz.universe
        terms = self.output_variable.terms
        matrix = torch.stack([terms[rule.consequent[1]](universe) for rule in self.rules])
        # Only cache in eval mode: in train mode the output MF parameters move under us.
        # train()/eval() and any .to()/.float() clear the cache.
        if not self.training:
            self._cons_cache = matrix
        return matrix

    def _aggregate(self, clipped: torch.Tensor) -> torch.Tensor:
        """Aggregate ``(batch, n_rules, resolution)`` across rules -> ``(batch, resolution)``."""
        if isinstance(self.aggregation, Max):
            return clipped.amax(dim=1)
        if isinstance(self.aggregation, ProbabilisticSum):
            return 1.0 - torch.prod(1.0 - clipped, dim=1)
        acc = clipped[:, 0]
        for i in range(1, clipped.shape[1]):
            acc = self.aggregation(acc, clipped[:, i])
        return acc

    def _infer_mamdani(self, strengths: torch.Tensor) -> torch.Tensor:
        cons = self._consequent_matrix()               # (R, res)
        n_rules = cons.shape[0]
        chunk = max(1, int(self.chunk_rules)) if self.chunk_rules is not None else 1

        aggregated = None
        for start in range(0, n_rules, chunk):
            block = cons[start:start + chunk]                              # (r, res)
            w = strengths[:, start:start + chunk]                          # (B, r)
            if block.shape[0] == 1:
                # Broadcast (B, 1) against (1, res) straight to (B, res). Going through
                # the 3-D path would add an amax over a singleton axis, i.e. a full
                # extra copy of the intermediate per rule.
                part = self.implication(w, block)                          # (B, res)
            else:
                clipped = self.implication(w.unsqueeze(-1), block.unsqueeze(0))  # (B, r, res)
                part = self._aggregate(clipped)                            # (B, res)
            aggregated = part if aggregated is None else self.aggregation(aggregated, part)

        return self.defuzz(aggregated)

    def _infer_tsk(self, prepared: Dict[str, torch.Tensor],
                   strengths: torch.Tensor) -> torch.Tensor:
        X = torch.stack([prepared[name] for name in self.input_order], dim=1)  # (B, n)
        f = torch.stack([rule.consequent_fn(X) for rule in self.rules], dim=1)  # (B, R)

        weighted = (strengths * f).sum(dim=1)
        mass = strengths.sum(dim=1)
        ok = mass > ACTIVATION_TOL

        if self.tsk_normalize:
            # Sanitise the denominator before dividing: a NaN here would propagate
            # into the gradients of the entries that did fire.
            safe = torch.where(ok, mass, torch.ones_like(mass))
            value = weighted / safe
        else:
            value = weighted

        return torch.where(ok, value, torch.full_like(value, self.no_rule_value))

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference.

        x: ``{variable name: tensor}``. Scalars, ``(B,)`` and ``(B, 1)`` are accepted.
        Returns a ``(B,)`` tensor, or a 0-dim tensor if every input was 0-dim.
        """
        prepared, batch_size, was_scalar = self._prepare_inputs(x)

        if len(self.rules) == 0:
            out = torch.full((batch_size,), self.no_rule_value, device=self._device())
            return out.squeeze(0) if was_scalar else out

        strengths = self._activations(prepared, batch_size)

        if self.mode == TSK:
            out = self._infer_tsk(prepared, strengths)
        else:
            out = self._infer_mamdani(strengths)

        return out.squeeze(0) if was_scalar else out

    # ------------------------------------------------------------ cache hooks

    def train(self, mode: bool = True):
        self._cons_cache = None
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        self._cons_cache = None
        return super()._apply(*args, **kwargs)

    # -------------------------------------------------------- introspection

    def print_rules(self):
        """Prints all rules in a human-readable format."""
        print(f"Fuzzy System Rules ({self.mode}):")
        for i, rule in enumerate(self.rules):
            print(f"Rule {i+1}: {rule}")

    def explain(self, x: Dict[str, torch.Tensor], threshold: float = 0.01):
        """Prints a detailed explanation of the inference for the first batch item."""
        prepared, batch_size, _ = self._prepare_inputs(x)

        print("\n--- Inference Explanation ---")
        print(f"Mode: {self.mode}")

        print("Inputs:")
        for k, v in prepared.items():
            print(f"  {k}: {v[0].item():.4f}")

        activations = self._activations(prepared, batch_size)[0]  # (n_rules,)

        if self.mode == TSK and len(self.rules):
            X = torch.stack([prepared[n][:1] for n in self.input_order], dim=1)
            consequents = [rule.consequent_fn(X)[0].item() for rule in self.rules]
            total = activations.sum().item()
        else:
            consequents, total = None, None

        print("\nRule Activations:")
        triggered_rules = 0
        for i, rule in enumerate(self.rules):
            act = activations[i].item()
            if act <= threshold:
                continue
            triggered_rules += 1
            print(f"  Rule {i+1}: {rule} [Strength: {act:.4f}]")
            for var, term in rule.antecedent:
                val = prepared[var.name][:1]
                mem = var.terms[term](val)[0].item()
                print(f"    - {var.name} is {term} (val={val.item():.2f}) -> {mem:.4f}")
            if consequents is not None:
                share = act / total if total > ACTIVATION_TOL else 0.0
                print(f"    -> f(x) = {consequents[i]:.4f} (weight share {share:.2%})")

        if triggered_rules == 0:
            print("  No rules triggered significantly.")

        out = self.forward(x)
        out_val = out.item() if out.numel() == 1 else out[0].item()
        print(f"\nFinal Output ({self.output_variable.name}): {out_val:.4f}")

    # -------------------------------------------------------- visualisation

    def plot_inference(self, x: Dict[str, torch.Tensor], **kwargs):
        """Rule-viewer figure of the inference for the first batch item.

        The graphical counterpart of :meth:`explain`. See
        :func:`fuzzy_torch.viz.plot_inference`. Needs matplotlib.
        """
        from .viz import plot_inference
        return plot_inference(self, x, **kwargs)

    def plot_surface(self, x: Optional[str] = None, y: Optional[str] = None, **kwargs):
        """Output over one or two inputs (curve or contour), others held fixed.

        See :func:`fuzzy_torch.viz.plot_surface`. Needs matplotlib.
        """
        from .viz import plot_surface
        return plot_surface(self, x, y, **kwargs)


class MamdaniSystem(FuzzyModule):
    """:class:`FuzzyModule` restricted to Mamdani rules; fails loudly otherwise."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != MAMDANI:
            raise ValueError("MamdaniSystem requires Mamdani rules; got TSK consequents")


class TSKSystem(FuzzyModule):
    """:class:`FuzzyModule` restricted to TSK rules; fails loudly otherwise."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != TSK:
            raise ValueError("TSKSystem requires TSK consequents; got Mamdani rules")
