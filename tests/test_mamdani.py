"""Mamdani inference: numerical correctness, configurable norms, empty rule bases.

The vectorized (batch, n_rules, resolution) path is checked against a naive
rule-by-rule reference implementation, which is what the library used to do.
"""
import pytest
import torch

from fuzzy_torch import (
    Bisector, Centroid, FuzzyModule, FuzzyRule, FuzzyVariable, MamdaniSystem,
    MeanOfMaximum, Max, Min, ProbabilisticSum, Product, TSKConsequent, TriangularMF,
)


def naive_mamdani(fis, x):
    """Straightforward rule-by-rule Mamdani, used only as a test oracle."""
    universe = fis.defuzz.universe
    strengths = fis.get_rule_activations(x)
    aggregated = None
    for i, rule in enumerate(fis.rules):
        cons = fis.output_variable.terms[rule.consequent[1]](universe)
        clipped = fis.implication(strengths[:, i].unsqueeze(-1), cons.unsqueeze(0))
        aggregated = clipped if aggregated is None else fis.aggregation(aggregated, clipped)
    return fis.defuzz(aggregated)


@pytest.mark.parametrize("implication", [Min(), Product()], ids=["min", "product"])
@pytest.mark.parametrize("aggregation", [Max(), ProbabilisticSum()], ids=["max", "probsum"])
def test_vectorized_matches_naive(service, food, tip, implication, aggregation):
    rules = [
        FuzzyRule([(service, 'Poor'), (food, 'Rancid')], (tip, 'Low')),
        FuzzyRule((service, 'Good'), (tip, 'Medium')),
        FuzzyRule([(service, 'Excellent'), (food, 'Delicious')], (tip, 'High')),
    ]
    fis = FuzzyModule([service, food], tip, rules, defuzz_method=Centroid(0, 30, 201),
                      implication=implication, aggregation=aggregation)
    x = {'Service': torch.linspace(0, 10, 25), 'Food': torch.linspace(10, 0, 25)}
    got, want = fis(x), naive_mamdani(fis, x)
    # The sweep includes ends where nothing fires, so NaNs must line up too.
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    assert torch.allclose(got, want, atol=1e-5, equal_nan=True)


def test_rule_chunking_matches_unchunked(service, food, tip):
    rules = [
        FuzzyRule([(service, 'Poor'), (food, 'Rancid')], (tip, 'Low')),
        FuzzyRule((service, 'Good'), (tip, 'Medium')),
        FuzzyRule([(service, 'Excellent'), (food, 'Delicious')], (tip, 'High')),
    ]
    x = {'Service': torch.linspace(0, 10, 17), 'Food': torch.linspace(0, 10, 17)}
    full = FuzzyModule([service, food], tip, rules, defuzz_method=Centroid(0, 30, 201))
    chunked = FuzzyModule([service, food], tip, rules, defuzz_method=Centroid(0, 30, 201),
                          chunk_rules=2)
    assert torch.allclose(full(x), chunked(x), atol=1e-6, equal_nan=True)


def test_single_fully_fired_symmetric_rule_gives_its_centre(service, tip):
    """Strength 1 on a symmetric consequent must defuzzify to its centre."""
    fis = FuzzyModule([service], tip, [FuzzyRule((service, 'Good'), (tip, 'Medium'))],
                      defuzz_method=Centroid(0, 30, 601))
    # 'Good' is TriangularMF(0, 5, 10), so Service = 5 fires it fully.
    out = fis({'Service': torch.tensor(5.0)})
    assert out.item() == pytest.approx(15.0, abs=1e-3)


def test_symmetric_rule_base_is_symmetric(tipping):
    """A symmetric system at the midpoint must land on the midpoint of the output."""
    out = tipping({'Service': torch.tensor(5.0), 'Food': torch.tensor(5.0)})
    assert out.item() == pytest.approx(15.0, abs=1e-3)


def test_output_is_monotonic_in_service(tipping):
    xs = torch.linspace(0.5, 9.5, 40)
    out = tipping({'Service': xs, 'Food': xs * 0 + 5.0})
    assert (out[1:] - out[:-1] >= -1e-4).all()


def test_output_stays_inside_the_universe(tipping):
    out = tipping({'Service': torch.linspace(0, 10, 101),
                   'Food': torch.linspace(10, 0, 101)})
    live = out[~torch.isnan(out)]
    assert live.min() >= 0.0 and live.max() <= 30.0


def test_shoulder_sets_fire_at_the_range_ends(service, food, tip):
    """Service=0 must fire 'Poor' fully; before the MF fix its strength was 0."""
    rules = [FuzzyRule([(service, 'Poor'), (food, 'Rancid')], (tip, 'Low'))]
    fis = FuzzyModule([service, food], tip, rules, defuzz_method=Centroid(0, 30, 201))
    strengths = fis.get_rule_activations({'Service': torch.tensor(0.0),
                                          'Food': torch.tensor(0.0)})
    assert strengths.item() == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(fis({'Service': torch.tensor(0.0), 'Food': torch.tensor(0.0)}))


# --------------------------------------------------------------- empty rule base

def _single_narrow_rule():
    s = FuzzyVariable('S', 0, 10)
    s.add_term('Mid', TriangularMF(4, 5, 6))
    t = FuzzyVariable('T', 0, 30)
    t.add_term('Med', TriangularMF(0, 15, 30))
    return s, t, [FuzzyRule((s, 'Mid'), (t, 'Med'))]


def test_no_rule_fires_gives_nan_by_default():
    """It used to return 0.0, which is not even inside a [0, 30] universe."""
    s, t, rules = _single_narrow_rule()
    fis = FuzzyModule([s], t, rules, defuzz_method=Centroid(0, 30, 201))
    out = fis({'S': torch.tensor([0.0, 5.0])})
    assert torch.isnan(out[0])
    assert out[1].item() == pytest.approx(15.0, abs=1e-3)


def test_no_rule_value_override():
    s, t, rules = _single_narrow_rule()
    fis = FuzzyModule([s], t, rules, defuzz_method=Centroid(0, 30, 201), no_rule_value=15.0)
    assert fis({'S': torch.tensor([0.0])}).item() == 15.0


def test_gradients_survive_a_non_firing_batch_item():
    """torch.where over a NaN branch would poison the gradients of the live items."""
    s, t, rules = _single_narrow_rule()
    fis = FuzzyModule([s], t, rules, defuzz_method=Centroid(0, 30, 201), no_rule_value=15.0)
    x = torch.tensor([0.0, 4.5], requires_grad=True)
    fis({'S': x}).sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in fis.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


# ------------------------------------------------------------------- validation

def test_consequent_must_target_the_system_output(service, tip):
    other = FuzzyVariable('Other', 0, 30)
    other.add_term('Medium', TriangularMF(0, 15, 30))
    with pytest.raises(ValueError, match="only single-output"):
        FuzzyModule([service], tip, [FuzzyRule((service, 'Good'), (other, 'Medium'))])


def test_mixed_rule_base_is_rejected(service, tip):
    with pytest.raises(ValueError, match="mixes Mamdani and TSK"):
        FuzzyModule([service], tip, [
            FuzzyRule((service, 'Good'), (tip, 'Medium')),
            FuzzyRule((service, 'Poor'), TSKConsequent(0)),
        ])


def test_unknown_term_is_caught_at_construction(service, tip):
    with pytest.raises(KeyError, match="Nonexistent"):
        FuzzyRule((service, 'Nonexistent'), (tip, 'Medium'))


def test_mamdani_system_rejects_tsk_rules(service, tip):
    with pytest.raises(ValueError, match="MamdaniSystem requires Mamdani"):
        MamdaniSystem([service], tip, [FuzzyRule((service, 'Good'), TSKConsequent(0))])


def test_non_differentiable_defuzzifier_warns(service, tip):
    with pytest.warns(UserWarning, match="not differentiable"):
        FuzzyModule([service], tip, [FuzzyRule((service, 'Good'), (tip, 'Medium'))],
                    defuzz_method=Bisector(0, 30))
    with pytest.warns(UserWarning, match="not differentiable"):
        FuzzyModule([service], tip, [FuzzyRule((service, 'Good'), (tip, 'Medium'))],
                    defuzz_method=MeanOfMaximum(0, 30))
