import numpy


class Element(object):
    def __init__(self, name, length=0, **kwargs):
        self.name = name
        self.Length = length
        self.PassMethod = kwargs.pop('PassMethod', 'IdentityPass')
        for k in kwargs:
            setattr(self, k, kwargs[k])


class Marker(Element):
    def __init__(self, name, **kwargs):
        kwargs.setdefault('PassMethod', 'IdentityPass')
        super(Marker, self).__init__(name, **kwargs)


class Aperture(Element):
    def __init__(self, name, limits, **kwargs):
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
        self.PolynomA = numpy.array([0, 0, 0, 0])
        self.PolynomB = numpy.array([0, 0, 0, 0])


class Quadrupole(Magnet):
    def __init__(self, name, length, k1, **kwargs):
        kwargs.setdefault('PassMethod', 'QuadLinearPass')
        super(Quadrupole, self).__init__(name, length, **kwargs)
        self.PolynomB[1] = k1


class Sextupole(Magnet):
    def __init__(self, name, length, k2, **kwargs):
        kwargs.setdefault('PassMethod', 'StrMPoleSymplectic4Pass')
        super(Sextupole, self).__init__(name, length, **kwargs)
        self.PolynomB[2] = k2
