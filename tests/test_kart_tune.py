"""kart_tune's parameter handling (the search itself needs SuperTuxKart)."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kart_tune():
    spec = importlib.util.spec_from_file_location('kart_tune', ROOT / 'examples/kart_tune.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec(name):
    return json.loads((ROOT / 'examples/race_control' / name).read_text())


@pytest.mark.parametrize('name', ['example_driver.json', 'kart_starter.json'])
def test_random_moves_keep_rules_shoulders_and_valid_shapes(kart_tune, name):
    from fuzzy_torch.envs.controllers import ConfigRaceDriver
    spec = _spec(name)
    params = kart_tune.Parameters(spec)
    rng = np.random.default_rng(0)
    for _ in range(20):
        moved = params.with_vector(params.vector + rng.normal(0, 0.3, params.vector.shape) * params.scales)
        ConfigRaceDriver(moved, **kart_tune.LIMITS)          # still a valid driver
        for system in ('steering', 'throttle'):
            assert [r['if'] for r in moved[system]['rules']] == [r['if'] for r in spec[system]['rules']]
            for var_name, var in moved[system]['inputs'].items():
                lo, hi = var['range']
                original = spec[system]['inputs'][var_name]['terms']
                for term, mf in var['terms'].items():
                    if mf[0] in ('tri', 'trap'):
                        assert mf[1:] == sorted(mf[1:]) and lo <= mf[1] and mf[-1] <= hi
                        # Anchors on the range edge stay, so shoulders stay shoulders.
                        for old, new in zip(original[term][1:], mf[1:]):
                            if old in (lo, hi):
                                assert new == old


def test_tsk_rule_outputs_are_tuned_too(kart_tune):
    spec = _spec('kart_starter.json')
    params = kart_tune.Parameters(spec)
    thens = [r['then'] for r in spec['throttle']['rules']]
    moved = params.with_vector(params.vector + 0.2 * params.scales)
    assert [r['then'] for r in moved['throttle']['rules']] != thens
    assert [r['then'] for r in moved['steering']['rules']] == [r['then'] for r in spec['steering']['rules']]
