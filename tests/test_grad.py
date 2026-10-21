"""End-to-end differentiability: gradients must reach inputs and every parameter."""
import pytest
import torch

from fuzzy_torch import (
    BellMF, Centroid, FuzzyModule, FuzzyRule, FuzzyVariable, GaussianMF, Min,
    Product, TSKConsequent, TSKSystem, TrapezoidalMF, TriangularMF,
)


def test_mamdani_gradients_reach_inputs_and_parameters(tipping):
    x = {'Service': torch.tensor([6.5], requires_grad=True),
         'Food': torch.tensor([9.8], requires_grad=True)}
    loss = ((tipping(x) - 25.0) ** 2).sum()
    loss.backward()

    for name, t in x.items():
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    assert any(t.grad.abs().sum() > 0 for t in x.values())

    grads = [(n, p.grad) for n, p in tipping.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for _, g in grads)
    assert any(g.abs().sum() > 0 for _, g in grads)


@pytest.mark.parametrize("implication", [Min(), Product()], ids=["min", "product"])
def test_gradients_are_finite_across_the_whole_input_range(service, food, tip, implication):
    rules = [
        FuzzyRule([(service, 'Poor'), (food, 'Rancid')], (tip, 'Low')),
        FuzzyRule((service, 'Good'), (tip, 'Medium')),
        FuzzyRule([(service, 'Excellent'), (food, 'Delicious')], (tip, 'High')),
    ]
    fis = FuzzyModule([service, food], tip, rules, defuzz_method=Centroid(0, 30, 201),
                      implication=implication, no_rule_value=15.0)
    x = torch.linspace(0, 10, 101, requires_grad=True)
    fis({'Service': x, 'Food': 10.0 - x.detach()}).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_mamdani_training_recovers_a_reachable_target():
    """Teacher/student: fit a target the rule base can actually represent.

    Fitting an arbitrary curve (say y = 3x) with two Mamdani rules has no good
    solution, so a "loss goes down" assertion there would be testing the optimiser's
    luck rather than the gradients. Generating the target from the same architecture
    makes the optimum reachable and the assertion meaningful.
    """
    torch.manual_seed(0)

    def build(mu_low, mu_high):
        s = FuzzyVariable('X', 0, 10)
        s.add_term('Low', GaussianMF(mu_low, 2.0))
        s.add_term('High', GaussianMF(mu_high, 2.0))
        t = FuzzyVariable('Y', 0, 30)
        t.add_term('Small', TriangularMF(0, 0, 15))
        t.add_term('Large', TriangularMF(15, 30, 30))
        fis = FuzzyModule([s], t, [FuzzyRule((s, 'Low'), (t, 'Small')),
                                   FuzzyRule((s, 'High'), (t, 'Large'))],
                          defuzz_method=Centroid(0, 30, 201), no_rule_value=15.0)
        return s, fis

    x = torch.linspace(0, 10, 64)
    _, teacher = build(2.0, 8.0)
    with torch.no_grad():
        y = teacher({'X': x}).clone()

    student_var, student = build(4.0, 6.0)
    opt = torch.optim.Adam(student_var.parameters(), lr=0.05)
    losses = []
    for _ in range(400):
        opt.zero_grad()
        loss = ((student({'X': x}) - y) ** 2).mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 1e-2
    assert losses[-1] < losses[0] * 0.01
    mus = sorted(p.item() for n, p in student_var.named_parameters() if n.endswith('mu'))
    assert mus[0] == pytest.approx(2.0, abs=0.3)
    assert mus[1] == pytest.approx(8.0, abs=0.3)


@pytest.mark.parametrize("mf_factory", [
    lambda: TriangularMF(0, 0, 5),      # degenerate left shoulder
    lambda: TriangularMF(5, 10, 10),    # degenerate right shoulder
    lambda: TrapezoidalMF(0, 0, 1, 3),
    lambda: GaussianMF(5, 2),
    lambda: BellMF(2, 0.4, 5),          # 2b < 1: used to make NaN gradients
], ids=["tri-left", "tri-right", "trap", "gauss", "bell-small-b"])
def test_every_membership_function_trains_inside_a_system(mf_factory):
    s = FuzzyVariable('X', 0, 10)
    s.add_term('A', mf_factory())
    t = FuzzyVariable('Y', 0, 30)
    t.add_term('B', TriangularMF(0, 15, 30))
    fis = FuzzyModule([s], t, [FuzzyRule((s, 'A'), (t, 'B'))],
                      defuzz_method=Centroid(0, 30, 201), no_rule_value=15.0)

    x = torch.linspace(0, 10, 41, requires_grad=True)
    fis({'X': x}).sum().backward()

    assert torch.isfinite(x.grad).all()
    for name, p in fis.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_tsk_gradients_reach_inputs_and_coefficients():
    s = FuzzyVariable('X', 0, 10)
    s.add_term('Low', GaussianMF(2.0, 2.0))
    s.add_term('High', GaussianMF(8.0, 2.0))
    out = FuzzyVariable('Y', 0, 30)
    rules = [FuzzyRule((s, 'Low'), TSKConsequent(2)),
             FuzzyRule((s, 'High'), TSKConsequent(2))]
    fis = TSKSystem([s], out, rules)

    x = torch.linspace(0, 10, 41, requires_grad=True)
    ((fis({'X': x}) - 12.0) ** 2).mean().backward()

    assert torch.isfinite(x.grad).all()
    for rule in rules:
        g = rule.consequent_fn.coefficients.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    for name, p in s.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
