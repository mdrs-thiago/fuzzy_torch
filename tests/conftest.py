"""Shared fixtures.

The package is consumed as ``fuzzy_torch`` (a git submodule checked out under that
name), so the tests import it by adding the *parent* of the repo to sys.path.
"""
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parent))

from fuzzy_torch import (  # noqa: E402
    Centroid, FuzzyModule, FuzzyRule, FuzzyVariable,
    TrapezoidalMF, TriangularMF,
)


@pytest.fixture
def service():
    v = FuzzyVariable('Service', 0, 10)
    v.add_term('Poor', TriangularMF(0, 0, 5))
    v.add_term('Good', TriangularMF(0, 5, 10))
    v.add_term('Excellent', TriangularMF(5, 10, 10))
    return v


@pytest.fixture
def food():
    v = FuzzyVariable('Food', 0, 10)
    v.add_term('Rancid', TrapezoidalMF(0, 0, 1, 3))
    v.add_term('Delicious', TrapezoidalMF(7, 9, 10, 10))
    return v


@pytest.fixture
def tip():
    v = FuzzyVariable('Tip', 0, 30)
    v.add_term('Low', TriangularMF(0, 0, 15))
    v.add_term('Medium', TriangularMF(0, 15, 30))
    v.add_term('High', TriangularMF(15, 30, 30))
    return v


@pytest.fixture
def tipping(service, food, tip):
    """The canonical tipping system, Mamdani."""
    rules = [
        FuzzyRule([(service, 'Poor'), (food, 'Rancid')], (tip, 'Low')),
        FuzzyRule((service, 'Good'), (tip, 'Medium')),
        FuzzyRule([(service, 'Excellent'), (food, 'Delicious')], (tip, 'High')),
    ]
    return FuzzyModule(inputs=[service, food], output=tip, rules=rules,
                       defuzz_method=Centroid(0, 30, resolution=201))


def approx(value, expected, tol=1e-5):
    return torch.allclose(torch.as_tensor(value, dtype=torch.float64),
                          torch.as_tensor(expected, dtype=torch.float64), atol=tol)
