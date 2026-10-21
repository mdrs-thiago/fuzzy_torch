"""Building fuzzy systems from JSON-style specs, and exporting them back."""
import json
import math

import pytest
import torch

from fuzzy_torch import (
    Centroid, ConfigError, FuzzyModule, FuzzyRule, FuzzyVariable, GaussianMF, TrapezoidalMF,
    TriangularMF, TSKConsequent, build_system, system_to_config,
)
from fuzzy_torch.system import MAMDANI, TSK

TIP = {
    'inputs': {
        'service': {'range': [0, 10], 'terms': {'poor': ['tri', 0, 0, 5], 'good': ['tri', 0, 5, 10],
                                                'excellent': ['tri', 5, 10, 10]}},
        'food': {'range': [0, 10], 'terms': {'rancid': ['trap', 0, 0, 1, 3],
                                             'delicious': ['trap', 7, 9, 10, 10]}},
    },
    'output': {'name': 'tip', 'range': [0, 30],
               'terms': {'low': ['tri', 0, 0, 15], 'medium': ['tri', 0, 15, 30], 'high': ['tri', 15, 30, 30]}},
    'rules': [
        {'if': {'service': 'poor', 'food': 'rancid'}, 'then': 'low'},
        {'if': {'service': 'good'}, 'then': 'medium'},
        {'if': {'service': 'excellent', 'food': 'delicious'}, 'then': 'high'},
    ],
}


def test_mamdani_spec_matches_the_hand_built_system():
    service = FuzzyVariable('service', 0, 10)
    service.add_term('poor', TriangularMF(0, 0, 5))
    service.add_term('good', TriangularMF(0, 5, 10))
    service.add_term('excellent', TriangularMF(5, 10, 10))
    food = FuzzyVariable('food', 0, 10)
    food.add_term('rancid', TrapezoidalMF(0, 0, 1, 3))
    food.add_term('delicious', TrapezoidalMF(7, 9, 10, 10))
    tip = FuzzyVariable('tip', 0, 30)
    tip.add_term('low', TriangularMF(0, 0, 15))
    tip.add_term('medium', TriangularMF(0, 15, 30))
    tip.add_term('high', TriangularMF(15, 30, 30))
    by_hand = FuzzyModule([service, food], tip, [
        FuzzyRule([(service, 'poor'), (food, 'rancid')], (tip, 'low')),
        FuzzyRule((service, 'good'), (tip, 'medium')),
        FuzzyRule([(service, 'excellent'), (food, 'delicious')], (tip, 'high')),
    ], defuzz_method=Centroid(0, 30, 201))

    fis = build_system(TIP)
    assert fis.mode == MAMDANI
    inputs = {'service': torch.linspace(0, 10, 21), 'food': torch.linspace(10, 0, 21)}
    assert torch.allclose(fis(inputs), by_hand(inputs), equal_nan=True)


def test_number_and_list_consequents_build_tsk_rules():
    spec = {
        'inputs': {'x': {'range': [0, 10], 'terms': {'lo': ['gauss', 2, 2], 'hi': ['gauss', 8, 2]}}},
        'output': {'range': [0, 100]},
        'rules': [{'if': {'x': 'lo'}, 'then': 5.0}, {'if': {'x': 'hi'}, 'then': [1.0, 2.0]}],
    }
    fis = build_system(spec)
    assert fis.mode == TSK
    # Far into 'hi' the output approaches 1 + 2x in raw units.
    assert fis({'x': torch.tensor(10.0)}).item() == pytest.approx(21.0, rel=0.05)


def test_export_round_trips_through_json():
    x = FuzzyVariable('x', -1, 1)
    x.add_term('n', GaussianMF(-1, .5))
    x.add_term('p', GaussianMF(1, .5))
    y = FuzzyVariable('y', -5, 5)
    rules = [FuzzyRule((x, 'n'), TSKConsequent(order=2, coefficients=[1, 2, 3])),
             FuzzyRule((x, 'p'), TSKConsequent(order=1, coefficients=[-1, .5], normalize=False))]
    original = FuzzyModule([x], y, rules, no_rule_value=0.0)
    rebuilt = build_system(json.loads(json.dumps(system_to_config(original))))
    probe = torch.linspace(-1, 1, 41)
    assert torch.allclose(original({'x': probe}), rebuilt({'x': probe}), atol=1e-5)

    mamdani = build_system(TIP)
    again = build_system(json.loads(json.dumps(system_to_config(mamdani))))
    s, f = torch.meshgrid(torch.linspace(0, 10, 11), torch.linspace(0, 10, 11), indexing='ij')
    inputs = {'service': s.reshape(-1), 'food': f.reshape(-1)}
    assert torch.allclose(mamdani(inputs), again(inputs), equal_nan=True)


def test_default_replaces_nan_when_no_rule_fires():
    spec = json.loads(json.dumps(TIP))
    spec['rules'] = [spec['rules'][2]]
    assert math.isnan(build_system(spec)({'service': 0.0, 'food': 0.0}).item())
    spec['default'] = 15.0
    assert build_system(spec)({'service': 0.0, 'food': 0.0}).item() == 15.0


@pytest.mark.parametrize('mutate, where', [
    (lambda s: s['rules'][1]['if'].update(service='goood'), "rules[1].if.service: unknown term 'goood'"),
    (lambda s: s['rules'][0].update(then='huge'), "rules[0].then: unknown output term 'huge'"),
    (lambda s: s['rules'][0]['if'].update(price='low'), 'rules[0].if.price: not one of'),
    (lambda s: s['inputs']['food']['terms'].update(ok=['tri', 5, 3, 8]), 'non-decreasing'),
    (lambda s: s['inputs']['food']['terms'].update(ok=['blob', 1]), "unknown shape 'blob'"),
    (lambda s: s['inputs']['food']['terms'].update(ok=['tri', 1, 2]), 'tri takes 3 parameters'),
    (lambda s: s['inputs']['food'].update(range=[10, 0]), 'inputs.food.range: min must be below max'),
    (lambda s: s['rules'][2].update(then=3.0), 'every rule must be Mamdani'),
    (lambda s: s.update(defuzz='magic'), 'defuzz: unknown method'),
    (lambda s: s.update(weight=1), "unknown key(s) ['weight']"),
    (lambda s: s['inputs']['food']['terms'].update(ok=['gauss', 5, True]), 'expected a number'),
])
def test_errors_name_the_offending_field(mutate, where):
    spec = json.loads(json.dumps(TIP))
    mutate(spec)
    with pytest.raises(ConfigError) as info:
        build_system(spec)
    assert where in str(info.value)


def test_limits_and_allowed_inputs():
    with pytest.raises(ConfigError, match='at most 2 rules'):
        build_system(TIP, max_rules=2)
    with pytest.raises(ConfigError, match='unknown input'):
        build_system(TIP, allowed_inputs=('service',))
    with pytest.raises(ConfigError, match='first-order TSK rule needs 3'):
        build_system({**TIP, 'output': {'range': [0, 30]},
                      'rules': [{'if': {'service': 'good'}, 'then': [1, 2]}]})
