"""Rule weights: a weight in [0, 1] scales a rule's firing strength."""
import math

import pytest
import torch

from fuzzy_torch import (
    ConfigError, FuzzyModule, FuzzyRule, FuzzyVariable, TSKConsequent, TriangularMF,
    build_system, system_to_config,
)


def _system(mode, weight=1.0):
    """Two inputs whose 'on' degree equals the input itself, one rule per input.

    Since mu_on(b) == b, a rule of weight w at input b fires exactly like an
    unweighted rule at input w * b -- the equivalence the tests below rely on.
    """
    a = FuzzyVariable('a', 0, 1)
    a.add_term('on', TriangularMF(0, 1, 1))
    b = FuzzyVariable('b', 0, 1)
    b.add_term('on', TriangularMF(0, 1, 1))
    y = FuzzyVariable('y', 0, 10)
    if mode == 'mamdani':
        y.add_term('low', TriangularMF(0, 0, 10))
        y.add_term('high', TriangularMF(0, 10, 10))
        cons_a, cons_b = (y, 'low'), (y, 'high')
    else:
        cons_a = TSKConsequent(0, coefficients=[2.0], normalize=False)
        cons_b = TSKConsequent(0, coefficients=[8.0], normalize=False)
    rules = [FuzzyRule((a, 'on'), cons_a), FuzzyRule((b, 'on'), cons_b, weight=weight)]
    return FuzzyModule([a, b], y, rules)


@pytest.mark.parametrize("mode", ["mamdani", "tsk"])
@pytest.mark.parametrize("w", [0.0, 0.3, 0.75, 1.0])
def test_weight_equals_scaled_firing_strength(mode, w):
    a = torch.tensor([0.9, 0.5, 0.2])
    b = torch.tensor([0.4, 1.0, 0.7])
    weighted = _system(mode, weight=w)({'a': a, 'b': b})
    reference = _system(mode)({'a': a, 'b': w * b})
    torch.testing.assert_close(weighted, reference)


def test_tsk_weighted_average_formula():
    fis = _system('tsk', weight=0.5)
    out = fis({'a': torch.tensor(1.0), 'b': torch.tensor(1.0)})
    assert out.item() == pytest.approx((1.0 * 2.0 + 0.5 * 8.0) / 1.5)


def test_zero_weight_rule_alone_counts_as_no_rule_fired():
    fis = _system('tsk', weight=0.0)
    out = fis({'a': torch.tensor(0.0), 'b': torch.tensor(1.0)})
    assert math.isnan(out.item())


def test_get_rule_activations_weighted_and_raw():
    fis = _system('mamdani', weight=0.25)
    x = {'a': torch.tensor([0.5]), 'b': torch.tensor([0.8])}
    torch.testing.assert_close(fis.get_rule_activations(x, weighted=False),
                               torch.tensor([[0.5, 0.8]]))
    torch.testing.assert_close(fis.get_rule_activations(x), torch.tensor([[0.5, 0.2]]))
    torch.testing.assert_close(fis.rule_weights(), torch.tensor([1.0, 0.25]))


def test_default_weight_is_one_and_hidden_from_repr():
    fis = _system('mamdani')
    assert fis.rules[0].weight == 1.0
    assert 'weight' not in repr(fis.rules[0])
    fis.rules[0].weight = 0.5
    assert repr(fis.rules[0]).endswith('(weight 0.5)')


@pytest.mark.parametrize("bad", [-0.1, 1.5, float('nan'), float('inf')])
def test_weight_outside_unit_interval_is_rejected(bad):
    v = FuzzyVariable('v', 0, 1)
    v.add_term('t', TriangularMF(0, 0.5, 1))
    out = FuzzyVariable('o', 0, 1)
    out.add_term('t', TriangularMF(0, 0.5, 1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FuzzyRule((v, 't'), (out, 't'), weight=bad)
    rule = FuzzyRule((v, 't'), (out, 't'))
    with pytest.raises(ValueError):
        rule.weight = bad
    assert rule.weight == 1.0


def test_weight_is_a_buffer_in_state_dict():
    fis = _system('tsk', weight=0.3)
    state = fis.state_dict()
    assert state['rules.1._weight'].item() == pytest.approx(0.3)
    assert not any('_weight' in name for name, _ in fis.named_parameters())

    other = _system('tsk', weight=1.0)
    other.load_state_dict(state)
    assert other.rules[1].weight == pytest.approx(0.3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_weight_follows_device():
    fis = _system('mamdani', weight=0.4).to('cuda')
    assert fis.rules[1]._weight.device.type == 'cuda'
    out = fis({'a': torch.tensor([0.5], device='cuda'), 'b': torch.tensor([0.5], device='cuda')})
    assert out.device.type == 'cuda'


def test_gradients_still_reach_membership_parameters():
    fis = _system('mamdani', weight=0.5)
    out = fis({'a': torch.tensor([0.6]), 'b': torch.tensor([0.7])})
    out.sum().backward()
    grads = [p.grad for p in fis.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ------------------------------------------------------------------ config

def _spec(weight=None):
    rule = {"if": {"x": "lo"}, "then": "down"}
    if weight is not None:
        rule["weight"] = weight
    return {
        "inputs": {"x": {"range": [0, 1], "terms": {"lo": ["tri", 0, 0, 1],
                                                    "hi": ["tri", 0, 1, 1]}}},
        "output": {"name": "u", "range": [-1, 1],
                   "terms": {"down": ["tri", -1, -1, 0], "up": ["tri", 0, 1, 1]}},
        "rules": [rule, {"if": {"x": "hi"}, "then": "up"}],
    }


def test_config_weight_round_trip():
    fis = build_system(_spec(weight=0.25))
    assert [r.weight for r in fis.rules] == [0.25, 1.0]
    exported = system_to_config(fis)
    assert exported['rules'][0]['weight'] == 0.25
    assert 'weight' not in exported['rules'][1]
    assert [r.weight for r in build_system(exported).rules] == [0.25, 1.0]


@pytest.mark.parametrize("bad", [1.5, -1, "high", True])
def test_config_rejects_bad_weight(bad):
    with pytest.raises(ConfigError, match=r"^rules\[0\]\.weight"):
        build_system(_spec(weight=bad))
