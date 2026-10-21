"""Membership function correctness, including the degenerate shoulders that used to
return 0 instead of 1 because of a `+1e-6` denominator guard."""
import pytest
import torch

from fuzzy_torch import (
    BellMF, GaussianMF, SigmoidMF, TrapezoidalMF, TriangularMF,
)

ALL_MFS = [
    TriangularMF(0, 0, 5),
    TriangularMF(0, 5, 10),
    TriangularMF(5, 10, 10),
    TrapezoidalMF(0, 0, 1, 3),
    TrapezoidalMF(2, 4, 6, 8),
    TrapezoidalMF(7, 9, 10, 10),
    GaussianMF(5, 2),
    BellMF(2, 2, 5),
    SigmoidMF(1, 5),
]


@pytest.mark.parametrize("mf, x, expected", [
    # Left shoulder: a == b. Used to return 0.0 at the peak.
    (TriangularMF(0, 0, 5), 0.0, 1.0),
    (TriangularMF(0, 0, 5), 2.5, 0.5),
    (TriangularMF(0, 0, 5), 5.0, 0.0),
    (TriangularMF(0, 0, 5), -1.0, 0.0),
    # Right shoulder: b == c. Used to return 0.0 at the peak.
    (TriangularMF(5, 10, 10), 10.0, 1.0),
    (TriangularMF(5, 10, 10), 7.5, 0.5),
    (TriangularMF(5, 10, 10), 11.0, 0.0),
    # Ordinary triangle.
    (TriangularMF(0, 5, 10), 2.5, 0.5),
    (TriangularMF(0, 5, 10), 5.0, 1.0),
    (TriangularMF(0, 5, 10), 7.5, 0.5),
    # Fully degenerate a == b == c: a spike at the point.
    (TriangularMF(3, 3, 3), 3.0, 1.0),
    (TriangularMF(3, 3, 3), 3.1, 0.0),
])
def test_triangular_values(mf, x, expected):
    assert mf(torch.tensor(x)).item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("mf, x, expected", [
    (TrapezoidalMF(0, 0, 1, 3), 0.0, 1.0),   # left shoulder, used to be 0.0
    (TrapezoidalMF(0, 0, 1, 3), 1.0, 1.0),
    (TrapezoidalMF(0, 0, 1, 3), 2.0, 0.5),
    (TrapezoidalMF(0, 0, 1, 3), 3.0, 0.0),
    (TrapezoidalMF(7, 9, 10, 10), 10.0, 1.0),  # right shoulder, used to be 0.0
    (TrapezoidalMF(7, 9, 10, 10), 8.0, 0.5),
    (TrapezoidalMF(7, 9, 10, 10), 7.0, 0.0),
    (TrapezoidalMF(2, 4, 6, 8), 5.0, 1.0),
    (TrapezoidalMF(2, 4, 6, 8), 3.0, 0.5),
    (TrapezoidalMF(2, 4, 6, 8), 7.0, 0.5),
])
def test_trapezoidal_values(mf, x, expected):
    assert mf(torch.tensor(x)).item() == pytest.approx(expected, abs=1e-6)


def test_no_epsilon_bias():
    """The old `+1e-6` guard perturbed ordinary interior values too."""
    assert TriangularMF(0, 0, 5)(torch.tensor(2.5)).item() == 0.5


@pytest.mark.parametrize("mf", ALL_MFS, ids=lambda m: repr(m).split('(')[0])
def test_range_is_unit_interval(mf):
    x = torch.linspace(-50, 50, 4001)
    y = mf(x)
    assert torch.isfinite(y).all()
    assert y.min() >= 0.0 and y.max() <= 1.0


@pytest.mark.parametrize("mf", ALL_MFS, ids=lambda m: repr(m).split('(')[0])
def test_shape_is_preserved(mf):
    for shape in [(), (7,), (3, 4)]:
        assert mf(torch.rand(shape) * 10).shape == shape


@pytest.mark.parametrize("mf", ALL_MFS, ids=lambda m: repr(m).split('(')[0])
def test_gradients_are_finite_everywhere(mf):
    """Covers the parameter breakpoints, where the piecewise branches meet."""
    x = torch.linspace(-12, 22, 1701, requires_grad=True)
    y = mf(x)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in mf.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_bell_gradient_at_center_with_small_b():
    """pow(0, p) has an infinite derivative for p < 1: this used to be NaN."""
    mf = BellMF(2.0, 0.4, 5.0)
    x = torch.tensor([5.0], requires_grad=True)
    y = mf(x)
    assert y.item() == pytest.approx(1.0, abs=1e-6)
    y.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(mf.a.grad) and torch.isfinite(mf.b.grad) and torch.isfinite(mf.c.grad)


def test_gaussian_ignores_sigma_sign():
    x = torch.linspace(0, 10, 101)
    assert torch.allclose(GaussianMF(5, 2)(x), GaussianMF(5, -2)(x))


@pytest.mark.parametrize("sigma", [0.0, -1e-6, 1e-6, -1e-3])
def test_gaussian_stays_finite_as_sigma_crosses_zero(sigma):
    """The old `sigma + 1e-6` guard only protected sigma from above, leaving a
    singularity at sigma == -1e-6 with a NaN value and a NaN gradient. Training that
    walks sigma through zero must stay finite from either side."""
    mf = GaussianMF(5.0, sigma)
    x = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = mf(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(mf.mu.grad) and torch.isfinite(mf.sigma.grad)


def test_sigmoid_does_not_overflow():
    mf = SigmoidMF(50.0, 5.0)
    y = mf(torch.tensor([-1e4, 1e4]))
    assert torch.isfinite(y).all()
    assert y.tolist() == [0.0, 1.0]


def test_gradcheck_in_double_precision():
    """Analytic gradients match finite differences away from the breakpoints."""
    for mf in [TriangularMF(0., 5., 10.), TrapezoidalMF(2., 4., 6., 8.),
               GaussianMF(5., 2.), BellMF(2., 2., 5.), SigmoidMF(1., 5.)]:
        mf = mf.double()
        x = torch.tensor([1.3, 4.7, 7.1], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(mf, (x,), eps=1e-6, atol=1e-4)
