"""Plotting helpers: they draw what the system actually computes, and never crash."""
import math

import pytest
import torch

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from fuzzy_torch import (  # noqa: E402
    BellMF, FuzzyModule, FuzzyRule, FuzzyVariable, GaussianMF, Product,
    ProbabilisticSum, SigmoidMF, TrapezoidalMF, TriangularMF, TSKConsequent, TSKSystem,
)
from fuzzy_torch.viz import default_span  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('mf', [
    TriangularMF(0, 0, 5), TrapezoidalMF(1, 3, 6, 9), GaussianMF(5, 1.5),
    BellMF(2, 3, 5), SigmoidMF(2, 5),
])
def test_membership_plot_covers_the_function(mf):
    ax = mf.plot()
    x, y = ax.lines[0].get_data()
    assert len(x) > 10
    assert y.max() > 0.95          # the default span reaches the peak / shoulder
    lo, hi = default_span(mf)
    assert lo < hi


def test_membership_plot_accepts_range_and_axes():
    fig, ax = plt.subplots()
    out = GaussianMF(0, 1).plot((-1, 1), ax=ax, color='r')
    assert out is ax
    x, _ = ax.lines[0].get_data()
    assert x.min() == pytest.approx(-1) and x.max() == pytest.approx(1)


def test_variable_plot_reports_degrees(service):
    ax = service.plot(value=6.5)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['Poor (0.00)', 'Good (0.70)', 'Excellent (0.30)']


def test_variable_plot_without_value(service):
    ax = service.plot()
    assert len(ax.lines) == len(service.terms)


def _crisp_line(fig):
    """The red output marker in the bottom-right cell."""
    ax = fig.axes[-1]
    xs = [line.get_xdata()[0] for line in ax.lines if line.get_color() == 'r']
    return xs[0] if xs else None


def test_mamdani_inference_matches_forward(tipping):
    x = {'Service': 6.5, 'Food': 9.8}
    fig = tipping.plot_inference(x)
    rows = len(tipping.rules) + 1
    cols = len(tipping.input_order) + 1
    assert len(fig.axes) == rows * cols
    assert _crisp_line(fig) == pytest.approx(tipping(x).item(), abs=1e-4)


@pytest.mark.parametrize('kw', [dict(implication=Product()),
                                dict(aggregation=ProbabilisticSum())])
def test_mamdani_inference_with_other_operators(service, food, tip, kw):
    rules = [FuzzyRule((service, 'Good'), (tip, 'Medium')),
             FuzzyRule((food, 'Delicious'), (tip, 'High'))]
    fis = FuzzyModule([service, food], tip, rules, **kw)
    fig = fis.plot_inference({'Service': 6.0, 'Food': 9.0})
    assert _crisp_line(fig) == pytest.approx(fis({'Service': 6.0, 'Food': 9.0}).item(),
                                             abs=1e-4)


def test_inference_when_no_rule_fires(tipping):
    x = {'Service': 0.0, 'Food': 5.0}
    assert math.isnan(tipping(x).item())
    fig = tipping.plot_inference(x)          # draws, without a crisp marker
    assert _crisp_line(fig) is None


def test_tsk_inference_matches_forward():
    x = FuzzyVariable('X', 0, 10)
    x.add_term('Low', GaussianMF(2, 2))
    x.add_term('High', GaussianMF(8, 2))
    fis = TSKSystem([x], FuzzyVariable('Y', 0, 100), [
        FuzzyRule((x, 'Low'), TSKConsequent(1, coefficients=[10.0, 5.0])),
        FuzzyRule((x, 'High'), TSKConsequent(0, coefficients=[80.0])),
    ])
    fig = fis.plot_inference({'X': 6.0})
    assert _crisp_line(fig) == pytest.approx(fis({'X': 6.0}).item(), abs=1e-4)


def test_large_rule_base_hides_inactive_rules():
    variables = []
    for name in 'abc':
        v = FuzzyVariable(name, -1, 1)
        v.add_term('N', TriangularMF(-1, -1, 0))
        v.add_term('Z', TriangularMF(-1, 0, 1))
        v.add_term('P', TriangularMF(0, 1, 1))
        variables.append(v)
    out = FuzzyVariable('u', -1, 1)
    out.add_term('Z', TriangularMF(-1, 0, 1))
    rules = [FuzzyRule([(variables[0], t0), (variables[1], t1), (variables[2], t2)], (out, 'Z'))
             for t0 in 'NZP' for t1 in 'NZP' for t2 in 'NZP']
    fis = FuzzyModule(variables, out, rules)
    x = {'a': 1.0, 'b': 1.0, 'c': 1.0}       # exactly one rule (P, P, P) fires
    fig = fis.plot_inference(x)
    assert len(fig.axes) == (1 + 1) * 4
    fig = fis.plot_inference(x, active_only=False)
    assert len(fig.axes) == (27 + 1) * 4


def test_surface_2d_and_1d(tipping):
    ax = tipping.plot_surface()
    assert ax.get_xlabel() == 'Service' and ax.get_ylabel() == 'Food'
    ax = tipping.plot_surface('Food', fixed={'Service': 9.0})
    x, y = ax.lines[0].get_data()
    expected = tipping({'Food': torch.as_tensor(x, dtype=torch.float32),
                        'Service': torch.full((len(x),), 9.0)})
    assert torch.allclose(torch.as_tensor(y, dtype=torch.float32), expected,
                          equal_nan=True, atol=1e-5)
