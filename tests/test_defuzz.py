"""Defuzzification methods, including the empty-set contract."""
import pytest
import torch

from fuzzy_torch import Bisector, Centroid, MeanOfMaximum

METHODS = [Centroid, Bisector, MeanOfMaximum]


def _symmetric_set(resolution=201):
    """A symmetric triangle over [0, 30], peaking at 15."""
    universe = torch.linspace(0, 30, resolution)
    return (1.0 - (universe - 15.0).abs() / 15.0).clamp_min(0.0).unsqueeze(0)


@pytest.mark.parametrize("cls", METHODS, ids=lambda c: c.__name__)
def test_symmetric_set_defuzzifies_to_its_centre(cls):
    d = cls(0, 30, 201)
    assert d(_symmetric_set()).item() == pytest.approx(15.0, abs=0.1)


@pytest.mark.parametrize("cls", METHODS, ids=lambda c: c.__name__)
def test_empty_set_is_nan_not_a_plausible_number(cls):
    """Centroid used to return 0.0 and MeanOfMaximum the universe midpoint."""
    d = cls(0, 30, 201)
    assert torch.isnan(d(torch.zeros(1, 201))).all()


@pytest.mark.parametrize("cls", METHODS, ids=lambda c: c.__name__)
def test_fill_value_is_configurable(cls):
    d = cls(0, 30, 201, fill_value=-1.0)
    assert d(torch.zeros(1, 201)).item() == -1.0


@pytest.mark.parametrize("cls", METHODS, ids=lambda c: c.__name__)
def test_empty_and_live_items_coexist_in_one_batch(cls):
    d = cls(0, 30, 201)
    values = torch.cat([torch.zeros(1, 201), _symmetric_set()], dim=0)
    out = d(values)
    assert torch.isnan(out[0]) and torch.isfinite(out[1])


@pytest.mark.parametrize("cls", METHODS, ids=lambda c: c.__name__)
def test_output_stays_inside_the_universe(cls):
    d = cls(0, 30, 201)
    values = torch.rand(16, 201)
    out = d(values)
    assert (out >= 0.0).all() and (out <= 30.0).all()


def test_centroid_is_differentiable():
    d = Centroid(0, 30, 201)
    values = _symmetric_set().clone().requires_grad_(True)
    out = d(values)
    assert out.requires_grad
    out.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_centroid_gradient_is_clean_when_part_of_the_batch_is_empty():
    """The NaN branch must not leak into the gradient of the live entry."""
    d = Centroid(0, 30, 201, fill_value=15.0)
    values = torch.cat([torch.zeros(1, 201), _symmetric_set()], dim=0).requires_grad_(True)
    d(values).sum().backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("cls", [Bisector, MeanOfMaximum], ids=lambda c: c.__name__)
def test_non_differentiable_methods_declare_it(cls):
    assert cls.differentiable is False
    out = cls(0, 30, 201)(_symmetric_set().clone().requires_grad_(True))
    assert not out.requires_grad


def test_centroid_declares_itself_differentiable():
    assert Centroid.differentiable is True


def test_centroid_matches_a_hand_computed_value():
    """Two spikes of equal height at 10 and 20 have their centre of gravity at 15."""
    d = Centroid(0, 30, 31)
    values = torch.zeros(1, 31)
    values[0, 10] = 1.0
    values[0, 20] = 1.0
    assert d(values).item() == pytest.approx(15.0, abs=1e-5)

    values[0, 20] = 3.0     # weighted: (10*1 + 20*3) / 4 = 17.5
    assert d(values).item() == pytest.approx(17.5, abs=1e-5)


def test_universe_moves_with_the_module():
    d = Centroid(0, 30, 11).double()
    assert d.universe.dtype == torch.float64
