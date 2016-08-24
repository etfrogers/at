import pytest
import numpy
import at
import elements


@pytest.fixture
def rin():
    rin = numpy.array(numpy.zeros((6,)))
    return rin


def test_drift_offset(rin):
    d = elements.Drift('drift', 1)
    lattice = [d]
    rin[0] = 1e-6
    rin[2] = 2e-6
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)
