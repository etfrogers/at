import pytest
import numpy
import at
import elements


@pytest.fixture
def rin():
    rin = numpy.array(numpy.zeros((6,)))
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
    rin[0] = 1e-5
    rin[2] = -1e-5
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)


def test_aperture_outside_limits(rin):
    a = elements.Aperture('aperture', [-1e-3, 1e-3, -1e-4, 1e-4])
    assert a.name == 'aperture'
    assert a.length == 0
    lattice = [a]
    rin[0] = 1e-2
    rin[2] = -1e-2
    at.atpass(lattice, rin, 1)
    assert numpy.isinf(rin[0])
    assert rin[2] == -1e-2  # Only the first coordinate is marked as infinity


def test_drift_offset(rin):
    d = elements.Drift('drift', 1)
    lattice = [d]
    rin[0] = 1e-6
    rin[2] = 2e-6
    rin_orig = numpy.array(rin, copy=True)
    at.atpass(lattice, rin, 1)
    numpy.testing.assert_equal(rin, rin_orig)
