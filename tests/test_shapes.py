"""Input shape handling, module registration and state round-trips."""
import pytest
import torch

from fuzzy_torch import (
    Centroid, FuzzyModule, FuzzyRule, FuzzyVariable, TSKConsequent,
)


def test_batched_input(tipping):
    out = tipping({'Service': torch.tensor([2.5, 5.0, 7.5]),
                   'Food': torch.tensor([1.0, 5.0, 9.5])})
    assert out.shape == (3,)


def test_scalar_input_returns_scalar(tipping):
    """0-dim tensors used to raise IndexError: tuple index out of range."""
    out = tipping({'Service': torch.tensor(5.0), 'Food': torch.tensor(5.0)})
    assert out.shape == ()
    assert torch.isfinite(out)


def test_column_vector_input(tipping):
    col = tipping({'Service': torch.tensor([[5.0], [7.5]]),
                   'Food': torch.tensor([[5.0], [9.5]])})
    flat = tipping({'Service': torch.tensor([5.0, 7.5]),
                    'Food': torch.tensor([5.0, 9.5])})
    assert col.shape == (2,)
    assert torch.allclose(col, flat)


def test_python_floats_are_accepted(tipping):
    assert tipping({'Service': 5.0, 'Food': 5.0}).shape == ()


def test_scalar_broadcasts_against_batch(tipping):
    out = tipping({'Service': torch.tensor([2.5, 5.0, 7.5]), 'Food': torch.tensor(5.0)})
    assert out.shape == (3,)


def test_inconsistent_batch_sizes_are_rejected(tipping):
    with pytest.raises(ValueError, match="inconsistent batch sizes"):
        tipping({'Service': torch.rand(3), 'Food': torch.rand(4)})


def test_multi_column_input_is_rejected(tipping):
    with pytest.raises(ValueError, match="expected a scalar"):
        tipping({'Service': torch.rand(3, 4), 'Food': torch.rand(3, 4)})


def test_missing_input_is_reported(tipping):
    with pytest.raises(ValueError, match="Food"):
        tipping({'Service': torch.tensor([5.0])})


def test_explain_accepts_scalars(tipping, capsys):
    """explain() used to raise IndexError on 0-dim inputs."""
    tipping.explain({'Service': torch.tensor(6.5), 'Food': torch.tensor(9.8)})
    assert "Final Output (Tip)" in capsys.readouterr().out


def test_rule_owns_its_membership_parameters(service, tip):
    """A standalone rule used to report zero parameters."""
    rule = FuzzyRule((service, 'Good'), (tip, 'Medium'))
    names = [n for n, _ in rule.named_parameters()]
    assert names, "rule registered no parameters"
    assert any('Good' in n for n in names)
    assert any('Medium' in n for n in names)


def test_rule_moves_with_to(service, tip):
    rule = FuzzyRule((service, 'Good'), (tip, 'Medium')).double()
    assert all(p.dtype == torch.float64 for p in rule.parameters())


def test_state_dict_round_trip(tipping):
    query = {'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])}
    reference = tipping(query).clone()
    state = {k: v.clone() for k, v in tipping.state_dict().items()}

    with torch.no_grad():
        for p in tipping.parameters():
            p.add_(1.0)
    assert not torch.allclose(tipping(query), reference)

    # Rules and the module share the variable objects, so restoring the saved state
    # must bring both views back to the original values.
    tipping.load_state_dict(state)
    assert torch.allclose(tipping(query), reference)


def test_dtype_cast_is_consistent(tipping):
    out32 = tipping({'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])})
    tipping.double()
    out64 = tipping({'Service': torch.tensor([6.5], dtype=torch.float64),
                     'Food': torch.tensor([9.8], dtype=torch.float64)})
    assert out64.dtype == torch.float64
    assert torch.allclose(out64.float(), out32, atol=1e-4)


def test_consequent_cache_is_invalidated_by_parameter_updates(tipping):
    """eval() caches the consequent grid; train() must drop it."""
    tipping.eval()
    before = tipping({'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])})
    tipping.train()
    with torch.no_grad():
        tipping.output_variable.terms['Medium'].b.add_(5.0)
    tipping.eval()
    after = tipping({'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])})
    assert not torch.allclose(before, after)


def test_empty_rule_base(service, tip):
    fis = FuzzyModule([service], tip, [], defuzz_method=Centroid(0, 30))
    out = fis({'Service': torch.tensor([1.0, 2.0])})
    assert out.shape == (2,)
    assert torch.isnan(out).all()


def test_tsk_consequent_moves_with_the_module(service):
    out = FuzzyVariable('Tip', 0, 30)
    fis = FuzzyModule([service], out, [FuzzyRule((service, 'Good'), TSKConsequent(1))])
    fis.double()
    cons = fis.rules[0].consequent_fn
    assert cons.coefficients.dtype == torch.float64
    assert cons.exponents.dtype == torch.long  # integer buffers must not be cast
