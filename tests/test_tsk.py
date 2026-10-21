"""Takagi-Sugeno-Kang inference at arbitrary polynomial order."""
import math

import pytest
import torch

from fuzzy_torch import (
    Centroid, FuzzyRule, FuzzyVariable, GaussianMF, TSKConsequent,
    TSKSystem, TriangularMF, polynomial_exponents,
)


def _system(order, n_rules=3, normalize=True, tsk_normalize=True):
    s = FuzzyVariable('Service', 0, 10)
    s.add_term('Poor', TriangularMF(0, 0, 5))
    s.add_term('Good', TriangularMF(0, 5, 10))
    s.add_term('Excellent', TriangularMF(5, 10, 10))
    out = FuzzyVariable('Tip', 0, 30)  # no terms needed in TSK mode
    terms = ['Poor', 'Good', 'Excellent'][:n_rules]
    rules = [FuzzyRule((s, t), TSKConsequent(order, normalize=normalize)) for t in terms]
    return TSKSystem([s], out, rules, tsk_normalize=tsk_normalize), s, rules


# ------------------------------------------------------------ polynomial basis

@pytest.mark.parametrize("n, p", [(1, 0), (1, 1), (1, 5), (2, 0), (2, 1), (2, 2),
                                  (2, 3), (3, 2), (4, 3)])
def test_coefficient_count_is_n_choose_p(n, p):
    exps = polynomial_exponents(n, p)
    assert exps.shape == (math.comb(n + p, p), n)
    assert TSKConsequent.n_terms_for(n, p) == math.comb(n + p, p)


def test_basis_includes_cross_terms():
    cons = TSKConsequent(2, input_names=['a', 'b'], input_ranges=[(0, 1), (0, 1)],
                         normalize=False)
    assert cons.monomial_names() == ['1', 'a', 'b', 'a^2', 'a*b', 'b^2']


def test_order_three_two_inputs_has_ten_terms():
    cons = TSKConsequent(3, input_names=['a', 'b'], input_ranges=[(0, 1), (0, 1)])
    assert cons.n_terms == 10
    assert 'a^2*b' in cons.monomial_names()


def test_negative_order_is_rejected():
    with pytest.raises(ValueError, match="order must be >= 0"):
        TSKConsequent(-1, input_names=['a'], normalize=False)


def test_wrong_coefficient_count_is_rejected():
    with pytest.raises(ValueError, match="needs 3 coefficients"):
        TSKConsequent(1, input_names=['a', 'b'], input_ranges=[(0, 1), (0, 1)],
                      coefficients=[1.0, 2.0])


def test_basis_is_exact_for_negative_and_zero_inputs():
    cons = TSKConsequent(3, input_names=['a'], normalize=False)
    x = torch.tensor([[-2.0], [0.0], [3.0]])
    assert torch.allclose(cons.basis(x),
                          torch.tensor([[1., -2., 4., -8.],
                                        [1., 0., 0., 0.],
                                        [1., 3., 9., 27.]]))


def test_basis_gradient_is_finite_at_zero():
    """pow(0, 0) for the constant term must not produce a NaN gradient."""
    cons = TSKConsequent(3, input_names=['a'], normalize=False)
    x = torch.tensor([[0.0]], requires_grad=True)
    cons.basis(x).sum().backward()
    assert torch.isfinite(x.grad).all()


# ------------------------------------------------------------------- inference

def test_order_zero_is_a_weighted_average_of_constants():
    fis, s, rules = _system(order=0)
    constants = [5.0, 15.0, 25.0]
    with torch.no_grad():
        for rule, c in zip(rules, constants):
            rule.consequent_fn.coefficients.fill_(c)

    x = torch.tensor([2.5, 5.0, 7.5])
    w = fis.get_rule_activations({'Service': x})
    expected = (w * torch.tensor(constants)).sum(1) / w.sum(1)
    assert torch.allclose(fis({'Service': x}), expected, atol=1e-6)


def test_order_one_single_rule_reproduces_the_linear_function():
    """One always-firing rule: the system output is exactly f(x)."""
    s = FuzzyVariable('X', -1, 1)
    s.add_term('Any', TriangularMF(-1, 0, 1))
    out = FuzzyVariable('Y', -10, 10)
    cons = TSKConsequent(1, normalize=False, coefficients=[2.0, 3.0])  # 2 + 3x
    fis = TSKSystem([s], out, [FuzzyRule((s, 'Any'), cons)])

    x = torch.tensor([-0.5, 0.0, 0.25, 0.5])
    assert torch.allclose(fis({'X': x}), 2.0 + 3.0 * x, atol=1e-6)


def test_order_two_single_rule_reproduces_the_quadratic():
    s = FuzzyVariable('X', -1, 1)
    s.add_term('Any', TriangularMF(-1, 0, 1))
    out = FuzzyVariable('Y', -10, 10)
    cons = TSKConsequent(2, normalize=False, coefficients=[1.0, -2.0, 0.5])
    fis = TSKSystem([s], out, [FuzzyRule((s, 'Any'), cons)])

    x = torch.tensor([-0.75, -0.2, 0.3, 0.9])
    assert torch.allclose(fis({'X': x}), 1.0 - 2.0 * x + 0.5 * x ** 2, atol=1e-6)


def test_cross_term_is_actually_evaluated():
    a = FuzzyVariable('A', -1, 1); a.add_term('Any', TriangularMF(-1, 0, 1))
    b = FuzzyVariable('B', -1, 1); b.add_term('Any', TriangularMF(-1, 0, 1))
    out = FuzzyVariable('Y', -10, 10)
    # monomials: 1, A, B, A^2, A*B, B^2
    cons = TSKConsequent(2, normalize=False, coefficients=[0, 0, 0, 0, 4.0, 0])
    fis = TSKSystem([a, b], out, [FuzzyRule([(a, 'Any'), (b, 'Any')], cons)])

    xa = torch.tensor([0.5, -0.5]); xb = torch.tensor([0.5, 0.5])
    assert torch.allclose(fis({'A': xa, 'B': xb}), 4.0 * xa * xb, atol=1e-6)


def test_normalization_rescales_to_the_variable_range():
    """With normalize=True the coefficients act on inputs mapped onto [-1, 1]."""
    s = FuzzyVariable('X', 0, 10)
    s.add_term('Any', TriangularMF(0, 5, 10))
    out = FuzzyVariable('Y', 0, 100)
    cons = TSKConsequent(1, normalize=True, coefficients=[10.0, 4.0])  # 10 + 4*z
    fis = TSKSystem([s], out, [FuzzyRule((s, 'Any'), cons)])

    x = torch.tensor([2.5, 5.0, 7.5])
    z = (x - 5.0) / 5.0
    assert torch.allclose(fis({'X': x}), 10.0 + 4.0 * z, atol=1e-6)


def test_unnormalized_tsk_is_the_weighted_sum():
    fis, s, rules = _system(order=0, tsk_normalize=False)
    with torch.no_grad():
        for rule in rules:
            rule.consequent_fn.coefficients.fill_(10.0)

    x = torch.tensor([2.5, 5.0, 7.5])
    w = fis.get_rule_activations({'Service': x})
    assert torch.allclose(fis({'Service': x}), 10.0 * w.sum(1), atol=1e-6)


def test_no_rule_fires_gives_nan_by_default():
    s = FuzzyVariable('X', 0, 10)
    s.add_term('Mid', TriangularMF(4, 5, 6))
    out = FuzzyVariable('Y', 0, 10)
    fis = TSKSystem([s], out, [FuzzyRule((s, 'Mid'), TSKConsequent(0))])
    result = fis({'X': torch.tensor([0.0, 5.0])})
    assert torch.isnan(result[0]) and torch.isfinite(result[1])


def test_shapes_match_mamdani(service):
    out = FuzzyVariable('Tip', 0, 30)
    fis = TSKSystem([service], out, [FuzzyRule((service, 'Good'), TSKConsequent(1))])
    assert fis({'Service': torch.tensor(5.0)}).shape == ()
    assert fis({'Service': torch.tensor([1.0, 2.0, 3.0])}).shape == (3,)


# -------------------------------------------------------------------- training

def test_first_order_tsk_fits_a_linear_target():
    """A short Adam run on a rule base spanning the range must fit an affine target.

    Only the *predictions* are asserted: a weighted average of overlapping local
    models is not identifiable, so individual coefficients have no unique solution.
    """
    torch.manual_seed(0)
    x_var = FuzzyVariable('X', 0, 10)
    for name, mu in [('Low', 0.0), ('Mid', 5.0), ('High', 10.0)]:
        x_var.add_term(name, GaussianMF(mu, 3.0))
    out = FuzzyVariable('Y', 0, 100)
    rules = [FuzzyRule((x_var, n), TSKConsequent(1, normalize=False))
             for n in ['Low', 'Mid', 'High']]
    fis = TSKSystem([x_var], out, rules)

    x = torch.linspace(0, 10, 200)
    y = 3.0 + 2.0 * x

    opt = torch.optim.Adam([p for r in rules for p in r.consequent_fn.parameters()], lr=0.1)
    for _ in range(1500):
        opt.zero_grad()
        loss = ((fis({'X': x}) - y) ** 2).mean()
        loss.backward()
        opt.step()

    assert loss.item() < 1e-3
    assert torch.allclose(fis({'X': x}), y, atol=0.1)


def test_single_rule_tsk_recovers_exact_coefficients():
    """With one always-firing rule the model is identifiable, so the fit must land
    on the true coefficients, not merely on the true predictions."""
    torch.manual_seed(0)
    x_var = FuzzyVariable('X', 0, 10)
    x_var.add_term('Any', TriangularMF(0, 5, 10))
    out = FuzzyVariable('Y', 0, 100)
    cons = TSKConsequent(1, normalize=False)
    fis = TSKSystem([x_var], out, [FuzzyRule((x_var, 'Any'), cons)])

    x = torch.linspace(0.5, 9.5, 200)
    y = 3.0 + 2.0 * x

    opt = torch.optim.Adam(cons.parameters(), lr=0.1)
    for _ in range(3000):
        opt.zero_grad()
        loss = ((fis({'X': x}) - y) ** 2).mean()
        loss.backward()
        opt.step()

    c0, c1 = cons.coefficients.tolist()
    assert c0 == pytest.approx(3.0, abs=1e-2)
    assert c1 == pytest.approx(2.0, abs=1e-3)


def test_gradients_reach_both_coefficients_and_membership_functions():
    fis, s, rules = _system(order=1)
    with torch.no_grad():
        for i, rule in enumerate(rules):
            rule.consequent_fn.coefficients.copy_(torch.tensor([5.0 * (i + 1), 1.0]))

    loss = ((fis({'Service': torch.tensor([2.0, 5.0, 8.0])}) - 12.0) ** 2).mean()
    loss.backward()

    coeff_grads = [r.consequent_fn.coefficients.grad for r in rules]
    assert all(g is not None and torch.isfinite(g).all() for g in coeff_grads)
    assert any(g.abs().sum() > 0 for g in coeff_grads)

    mf_grads = [p.grad for p in s.parameters() if p.grad is not None]
    assert mf_grads and all(torch.isfinite(g).all() for g in mf_grads)
    assert any(g.abs().sum() > 0 for g in mf_grads)


# ------------------------------------------------------------------ validation

def test_consequent_binding_is_checked(service):
    out = FuzzyVariable('Tip', 0, 30)
    cons = TSKConsequent(1, input_names=['Wrong'], input_ranges=[(0, 1)])
    with pytest.raises(ValueError, match="bound to inputs"):
        TSKSystem([service], out, [FuzzyRule((service, 'Good'), cons)])


def test_unbound_consequent_cannot_be_called():
    with pytest.raises(RuntimeError, match="not bound"):
        TSKConsequent(1)(torch.rand(2, 1))


def test_normalize_without_ranges_is_rejected():
    with pytest.raises(ValueError, match="needs input_ranges"):
        TSKConsequent(1, input_names=['a'], normalize=True)


def test_tsk_system_rejects_mamdani_rules(service, tip):
    with pytest.raises(ValueError, match="TSKSystem requires TSK"):
        TSKSystem([service], tip, [FuzzyRule((service, 'Good'), (tip, 'Medium'))])


def test_defuzz_method_is_ignored_with_a_warning(service):
    out = FuzzyVariable('Tip', 0, 30)
    with pytest.warns(UserWarning, match="ignored for a TSK system"):
        fis = TSKSystem([service], out, [FuzzyRule((service, 'Good'), TSKConsequent(0))],
                        defuzz_method=Centroid(0, 30))
    assert fis.defuzz is None
