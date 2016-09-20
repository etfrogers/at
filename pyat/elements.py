import numpy


class Element(object):
    def __init__(self, name, length=0, **kwargs):
        self.name = name
        self.Length = length
        self.PassMethod = kwargs.pop('PassMethod', 'IdentityPass')
        for k in kwargs:
            setattr(self, k, kwargs[k])


class Marker(Element):
    def __init__(self, name, length, **kwargs):
        kwargs.setdefault('PassMethod', 'IdentityPass')
        super(Marker, self).__init__(name, length, **kwargs)


class Aperture(Element):
    def __init__(self, name, limits=[0,0,0,0], **kwargs):
        assert len(limits) == 4
        kwargs.setdefault('PassMethod', 'AperturePass')
        super(Aperture, self).__init__(name, **kwargs)
        self.Limits = numpy.array(limits)


class Drift(Element):
    def __init__(self, name, length, **kwargs):
        kwargs.setdefault('PassMethod', 'DriftPass')
        super(Drift, self).__init__(name, length, **kwargs)


class Magnet(Element):
    def __init__(self, name, length, **kwargs):
        super(Magnet, self).__init__(name, length, **kwargs)
        poly_a = kwargs.get('PolynomA', numpy.array([0, 0, 0, 0]))
        poly_b = kwargs.get('PolynomB', numpy.array([0, 0, 0, 0]))
        self.PolynomA = numpy.array(poly_a, dtype=numpy.float64)
        self.PolynomB = numpy.array(poly_b, dtype=numpy.float64)
        self.MaxOrder = 3
        self.NumIntSteps = 10


class Dipole(Magnet):
    def __init__(self, name, length, **kwargs):
        kwargs.setdefault('PassMethod', 'BndMPoleSymplectic4E2Pass')
        super(Dipole, self).__init__(name, length, **kwargs)


class Quadrupole(Magnet):
    def __init__(self, name, length, **kwargs):
        kwargs.setdefault('PassMethod', 'QuadLinearPass')
        super(Quadrupole, self).__init__(name, length, **kwargs)


class Sextupole(Magnet):
    def __init__(self, name, length, **kwargs):
        kwargs.setdefault('PassMethod', 'StrMPoleSymplectic4Pass')
        super(Sextupole, self).__init__(name, length, **kwargs)
