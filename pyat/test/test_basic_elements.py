import pytest
import numpy
import at
import elements


@pytest.fixture
def rin():
    rin = numpy.array(numpy.zeros((6,1)))
    return rin


def test_marker(rin):
    m = elements.Marker('marker')
    assert m.name == 'marker'
    assert m.length == 0
    lattice = [m]
    rin = numpy.random.rand(*rin.shape)
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_aperture_inside_limits(rin):
    a = elements.Aperture('aperture', [-1e-3, 1e-3, -1e-4, 1e-4])
    assert a.name == 'aperture'
    assert a.length == 0
    lattice = [a]
    rin[0][0] = 1e-5
    rin[2][0] = -1e-5
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_aperture_outside_limits(rin):
    a = elements.Aperture('aperture', [-1e-3, 1e-3, -1e-4, 1e-4])
    assert a.name == 'aperture'
    assert a.length == 0
    lattice = [a]
    rin[0][0] = 1e-2
    rin[2][0] = -1e-2
    at.atpass(lattice, rin, 1)
    assert numpy.isinf(rin[0][0])
    assert rin[2][0] == -1e-2  # Only the first coordinate is marked as infinity


def test_drift_offset(rin):
    d = elements.Drift('drift', 1)
    lattice = [d]
    rin[0][0] = 1e-6
    rin[2][0] = 2e-6
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_drift_divergence(rin):
    d = elements.Drift('drift', 1.0)
    assert d.name == 'drift'
    assert d.length == 1
    lattice = [d]
    rin[1][0] = 1e-6
    rin[3][0] = -2e-6
    at.atpass(lattice, rin, 1)
    # results from Matlab
    rin_expected = numpy.array([1e-6, 1e-6, -2e-6, -2e-6, 0, 2.5e-12]).reshape(6,1)
    numpy.testing.assert_equal(rin, rin_expected)
