"""Build fuzzy systems from plain data (a dict, usually parsed from JSON), and back.

The format is deliberately small enough to write by hand::

    {
      "inputs": {
        "error": {"range": [-1, 1],
                  "terms": {"neg": ["trap", -1, -1, -0.5, 0],
                            "zero": ["tri", -0.5, 0, 0.5],
                            "pos": ["trap", 0, 0.5, 1, 1]}}
      },
      "output": {"name": "u", "range": [-1, 1],
                 "terms": {"left": ["tri", -1, -1, 0], "none": ["tri", -1, 0, 1],
                           "right": ["tri", 0, 1, 1]}},
      "rules": [
        {"if": {"error": "neg"},  "then": "right"},
        {"if": {"error": "zero"}, "then": "none"},
        {"if": {"error": "pos"},  "then": "left"}
      ]
    }

Membership functions are ``[shape, *params]`` with shape one of ``tri`` (a, b, c),
``trap`` (a, b, c, d), ``gauss`` (mu, sigma), ``bell`` (a, b, c) or ``sigmoid`` (b, c).

A rule's ``then`` selects the inference mode:

- a term name (string) -> Mamdani;
- a number -> zero-order TSK, ``f(x) = c``;
- a list ``[c0, c1, ..., cn]`` -> first-order TSK in raw input units, columns in
  the order the inputs are listed;
- ``{"order": p, "coefficients": [...], "normalize": bool}`` -> TSK of any order.

A rule may also carry ``"weight"``, a number in [0, 1] (default 1) that scales its
firing strength.

Optional top-level keys: ``and`` (``min``/``product``), ``implication``
(``min``/``product``), ``aggregation`` (``max``/``probsum``), ``defuzz``
(``centroid``/``bisector``/``mom``), ``resolution`` and ``default`` (the value
returned when no rule fires; NaN if omitted).

Every validation failure raises :class:`ConfigError` whose message starts with the
path of the offending field (``rules[3].if.error: ...``), so it can be shown to the
author of the file as-is.
"""
import math
import warnings
from typing import Any, Dict, List, Optional

from .consequent import TSKConsequent
from .defuzz import Bisector, Centroid, MeanOfMaximum
from .membership import BellMF, GaussianMF, SigmoidMF, TrapezoidalMF, TriangularMF
from .norm import Max, Min, ProbabilisticSum, Product
from .rule import FuzzyRule
from .system import FuzzyModule, TSK
from .variable import FuzzyVariable


class ConfigError(ValueError):
    """A fuzzy-system spec is malformed. The message names the offending field."""


# shape -> (class, parameter names). Parameter order matches the constructors.
SHAPES = {
    'tri': (TriangularMF, ('a', 'b', 'c')),
    'trap': (TrapezoidalMF, ('a', 'b', 'c', 'd')),
    'gauss': (GaussianMF, ('mu', 'sigma')),
    'bell': (BellMF, ('a', 'b', 'c')),
    'sigmoid': (SigmoidMF, ('b', 'c')),
}
_T_NORMS = {'min': Min, 'product': Product}
_S_NORMS = {'max': Max, 'probsum': ProbabilisticSum}
_DEFUZZ = {'centroid': Centroid, 'bisector': Bisector, 'mom': MeanOfMaximum}


def _fail(path: str, message: str):
    raise ConfigError(f"{path}: {message}" if path else message)


def _number(value, path: str) -> float:
    # bool is an int subclass; `true` in a JSON file is never meant as 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        _fail(path, "must be finite")
    return value


def _mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(path, f"expected an object, got {type(value).__name__}")
    return value


def _choice(spec: dict, key: str, table: dict, default: str, path: str):
    name = spec.get(key, default)
    if name not in table:
        _fail(f"{path}{key}", f"unknown value {name!r} (choose from {', '.join(table)})")
    return table[name]()


def build_membership(spec, path: str = ''):
    """``["tri", a, b, c]`` -> :class:`TriangularMF`, and so on for every shape."""
    if not isinstance(spec, (list, tuple)) or not spec or not isinstance(spec[0], str):
        _fail(path, 'expected ["shape", params...], e.g. ["tri", 0, 0.5, 1]')
    shape, params = spec[0], spec[1:]
    if shape not in SHAPES:
        _fail(path, f"unknown shape {shape!r} (choose from {', '.join(SHAPES)})")
    cls, names = SHAPES[shape]
    if len(params) != len(names):
        _fail(path, f"{shape} takes {len(names)} parameters ({', '.join(names)}), "
                    f"got {len(params)}")
    values = [_number(p, f"{path}[{i + 1}]") for i, p in enumerate(params)]
    if shape in ('tri', 'trap') and any(x > y for x, y in zip(values, values[1:])):
        _fail(path, f"{shape} parameters must be non-decreasing, got {values}")
    if shape == 'gauss' and values[1] <= 0:
        _fail(path, "gauss sigma must be positive")
    if shape == 'bell' and values[0] == 0:
        _fail(path, "bell width a must be non-zero")
    return cls(*values)


def build_variable(name: str, spec, path: str = '', need_terms: bool = True,
                   max_terms: Optional[int] = None) -> FuzzyVariable:
    spec = _mapping(spec, path)
    rng = spec.get('range')
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        _fail(f"{path}.range", "expected [min, max]")
    lo, hi = (_number(v, f"{path}.range[{i}]") for i, v in enumerate(rng))
    if not lo < hi:
        _fail(f"{path}.range", f"min must be below max, got [{lo}, {hi}]")
    variable = FuzzyVariable(name, lo, hi)
    terms = spec.get('terms', {})
    terms = _mapping(terms, f"{path}.terms")
    if need_terms and not terms:
        _fail(f"{path}.terms", "needs at least one term")
    if max_terms is not None and len(terms) > max_terms:
        _fail(f"{path}.terms", f"at most {max_terms} terms allowed, got {len(terms)}")
    for term, mf in terms.items():
        variable.add_term(str(term), build_membership(mf, f"{path}.terms.{term}"))
    return variable


def _consequent(then, n_inputs: int, path: str):
    """A TSK consequent from a number, a coefficient list, or an explicit object."""
    if isinstance(then, (int, float)) and not isinstance(then, bool):
        return TSKConsequent(order=0, coefficients=[_number(then, path)], normalize=False)
    if isinstance(then, list):
        if len(then) != n_inputs + 1:
            _fail(path, f"a first-order TSK rule needs {n_inputs + 1} coefficients "
                        f"[c0, one per input], got {len(then)}")
        return TSKConsequent(order=1, normalize=False,
                             coefficients=[_number(c, f"{path}[{i}]") for i, c in enumerate(then)])
    spec = _mapping(then, path)
    order = spec.get('order')
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 3:
        _fail(f"{path}.order", "expected an integer from 0 to 3")
    coeffs = spec.get('coefficients')
    need = TSKConsequent.n_terms_for(n_inputs, order)
    if not isinstance(coeffs, list) or len(coeffs) != need:
        _fail(f"{path}.coefficients", f"order {order} over {n_inputs} inputs needs "
                                      f"{need} coefficients")
    normalize = spec.get('normalize', False)
    if not isinstance(normalize, bool):
        _fail(f"{path}.normalize", "expected true or false")
    return TSKConsequent(order=order, normalize=normalize,
                         coefficients=[_number(c, f"{path}.coefficients[{i}]")
                                       for i, c in enumerate(coeffs)])


def build_system(spec, *, allowed_inputs=None, output_name: Optional[str] = None,
                 max_rules: Optional[int] = None, max_inputs: Optional[int] = None,
                 max_terms: Optional[int] = None, path: str = '') -> FuzzyModule:
    """Build a :class:`FuzzyModule` from a spec dict (see the module docstring).

    Args:
        allowed_inputs: if given, input names must come from this collection --
            useful when the inputs are fixed by an environment's observation.
        output_name: overrides ``output.name``.
        max_rules, max_inputs, max_terms: size limits, for specs from untrusted sources.
    """
    spec = _mapping(spec, path)
    prefix = f"{path}." if path else ''
    known = {'type', 'inputs', 'output', 'rules', 'and', 'implication', 'aggregation',
             'defuzz', 'resolution', 'default', 'description'}
    unknown = set(spec) - known
    if unknown:
        _fail(path, f"unknown key(s) {sorted(unknown)} (allowed: {sorted(known)})")

    inputs_spec = _mapping(spec.get('inputs'), f"{prefix}inputs")
    if not inputs_spec:
        _fail(f"{prefix}inputs", "needs at least one input")
    if max_inputs is not None and len(inputs_spec) > max_inputs:
        _fail(f"{prefix}inputs", f"at most {max_inputs} inputs allowed, got {len(inputs_spec)}")
    inputs = {}
    for name, var_spec in inputs_spec.items():
        if allowed_inputs is not None and name not in allowed_inputs:
            _fail(f"{prefix}inputs.{name}",
                  f"unknown input (choose from {', '.join(allowed_inputs)})")
        inputs[name] = build_variable(name, var_spec, f"{prefix}inputs.{name}",
                                      max_terms=max_terms)

    rules_spec = spec.get('rules')
    if not isinstance(rules_spec, list) or not rules_spec:
        _fail(f"{prefix}rules", "expected a non-empty list of rules")
    if max_rules is not None and len(rules_spec) > max_rules:
        _fail(f"{prefix}rules", f"at most {max_rules} rules allowed, got {len(rules_spec)}")

    # The mode follows from the first rule's `then`; mixing is rejected per rule below.
    first_then = _mapping(rules_spec[0], f"{prefix}rules[0]").get('then')
    mamdani = isinstance(first_then, str)
    declared = spec.get('type')
    if declared is not None and declared not in ('mamdani', 'tsk'):
        _fail(f"{prefix}type", "expected 'mamdani' or 'tsk'")
    if declared is not None and (declared == 'mamdani') != mamdani:
        _fail(f"{prefix}type", f"declared {declared!r}, but the rules' 'then' values are "
                               f"{'term names (Mamdani)' if mamdani else 'numbers (TSK)'}")

    out_spec = _mapping(spec.get('output'), f"{prefix}output")
    out_name = output_name or str(out_spec.get('name', 'output'))
    output = build_variable(out_name, out_spec, f"{prefix}output", need_terms=mamdani,
                            max_terms=max_terms)

    t_norm = _choice(spec, 'and', _T_NORMS, 'min', prefix)
    rules: List[FuzzyRule] = []
    for i, rule_spec in enumerate(rules_spec):
        rpath = f"{prefix}rules[{i}]"
        rule_spec = _mapping(rule_spec, rpath)
        extra = set(rule_spec) - {'if', 'then', 'weight'}
        if extra:
            _fail(rpath, f"unknown key(s) {sorted(extra)}; a rule has only 'if', 'then' "
                         "and an optional 'weight'")
        antecedent_spec = _mapping(rule_spec.get('if'), f"{rpath}.if")
        if not antecedent_spec:
            _fail(f"{rpath}.if", "needs at least one condition")
        antecedent = []
        for var_name, term in antecedent_spec.items():
            if var_name not in inputs:
                _fail(f"{rpath}.if.{var_name}",
                      f"not one of this system's inputs ({', '.join(inputs)})")
            if term not in inputs[var_name].terms:
                _fail(f"{rpath}.if.{var_name}",
                      f"unknown term {term!r} (available: {', '.join(inputs[var_name].terms)})")
            antecedent.append((inputs[var_name], term))
        then = rule_spec.get('then')
        if isinstance(then, str) != mamdani:
            _fail(f"{rpath}.then", "every rule must be Mamdani (a term name) or every rule "
                                   "TSK (a number/coefficients); this one differs from rule 0")
        if mamdani:
            if then not in output.terms:
                _fail(f"{rpath}.then",
                      f"unknown output term {then!r} (available: {', '.join(output.terms)})")
            consequent = (output, then)
        else:
            consequent = _consequent(then, len(inputs), f"{rpath}.then")
        weight = _number(rule_spec.get('weight', 1.0), f"{rpath}.weight")
        if not 0.0 <= weight <= 1.0:
            _fail(f"{rpath}.weight", f"must be in [0, 1], got {weight}")
        rules.append(FuzzyRule(antecedent, consequent, t_norm=t_norm, weight=weight))

    default = spec.get('default')
    no_rule_value = float('nan') if default is None else _number(default, f"{prefix}default")
    kwargs: Dict[str, Any] = {'no_rule_value': no_rule_value}
    if mamdani:
        resolution = spec.get('resolution', 201)
        if isinstance(resolution, bool) or not isinstance(resolution, int) \
                or not 11 <= resolution <= 2001:
            _fail(f"{prefix}resolution", "expected an integer from 11 to 2001")
        defuzz_cls = _DEFUZZ.get(spec.get('defuzz', 'centroid'))
        if defuzz_cls is None:
            _fail(f"{prefix}defuzz", f"unknown method (choose from {', '.join(_DEFUZZ)})")
        kwargs.update(
            defuzz_method=defuzz_cls(output.range_min, output.range_max, resolution),
            implication=_choice(spec, 'implication', _T_NORMS, 'min', prefix),
            aggregation=_choice(spec, 'aggregation', _S_NORMS, 'max', prefix),
        )
    with warnings.catch_warnings():
        # Non-differentiable defuzzifiers are a legitimate choice for a hand-written spec.
        warnings.simplefilter('ignore')
        return FuzzyModule(list(inputs.values()), output, rules, **kwargs)


# ------------------------------------------------------------------- export

def _round(x: float, digits: int) -> float:
    return float(round(float(x), digits))


def membership_to_config(mf, digits: int = 6) -> list:
    for shape, (cls, names) in SHAPES.items():
        if type(mf) is cls:
            return [shape] + [_round(getattr(mf, n).detach(), digits) for n in names]
    raise ConfigError(f"cannot export membership function {type(mf).__name__}")


def _norm_name(norm, table) -> str:
    for name, cls in table.items():
        if type(norm) is cls:
            return name
    raise ConfigError(f"cannot export {type(norm).__name__}")


def system_to_config(system: FuzzyModule, digits: int = 6) -> dict:
    """Inverse of :func:`build_system`: a JSON-ready dict of the system's *current*
    parameters, so a controller tuned by gradient descent can be saved and shared."""
    def variable(v, with_terms=True):
        out = {'range': [_round(v.range_min, digits), _round(v.range_max, digits)]}
        if with_terms:
            out['terms'] = {n: membership_to_config(mf, digits) for n, mf in v.terms.items()}
        return out

    tsk = system.mode == TSK
    spec: Dict[str, Any] = {
        'type': 'tsk' if tsk else 'mamdani',
        'inputs': {name: variable(system.input_variables[name]) for name in system.input_order},
        'output': {'name': system.output_variable.name,
                   **variable(system.output_variable, with_terms=not tsk)},
        'rules': [],
    }
    t_norms = {_norm_name(r.t_norm, _T_NORMS) for r in system.rules}
    if len(t_norms) > 1:
        raise ConfigError("rules use different AND operators; the format has one per system")
    if t_norms and t_norms != {'min'}:
        spec['and'] = t_norms.pop()

    for rule in system.rules:
        entry: Dict[str, Any] = {'if': {v.name: t for v, t in rule.antecedent}}
        if tsk:
            cons = rule.consequent_fn
            if not isinstance(cons, TSKConsequent):
                raise ConfigError(f"cannot export consequent {type(cons).__name__}")
            coeffs = [_round(c, digits) for c in cons.coefficients.detach().tolist()]
            if cons.order == 0 and not cons.normalize:
                entry['then'] = coeffs[0]
            elif cons.order == 1 and not cons.normalize:
                entry['then'] = coeffs
            else:
                entry['then'] = {'order': cons.order, 'coefficients': coeffs,
                                 'normalize': cons.normalize}
        else:
            entry['then'] = rule.consequent[1]
        if rule.weight != 1.0:
            entry['weight'] = _round(rule.weight, digits)
        spec['rules'].append(entry)

    if not tsk:
        spec['implication'] = _norm_name(system.implication, _T_NORMS)
        spec['aggregation'] = _norm_name(system.aggregation, _S_NORMS)
        spec['defuzz'] = next(n for n, c in _DEFUZZ.items() if type(system.defuzz) is c)
        spec['resolution'] = int(system.defuzz.resolution)
    if not math.isnan(system.no_rule_value):
        spec['default'] = system.no_rule_value
    return spec
