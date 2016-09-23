import numpy


class Element(object):
    REQUIRED_ATTRIBUTES = []

    def __init__(self, length=0, **kwargs):
        self.Length = length
        self.PassMethod = kwargs.pop('PassMethod', 'IdentityPass')
        for k in kwargs:
            setattr(self, k, kwargs[k])


class Marker(Element):

    def __init__(self, **kwargs):
        kwargs.setdefault('PassMethod', 'IdentityPass')
        super(Marker, self).__init__(0, **kwargs)


class Aperture(Element):
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Limits']

    def __init__(self, limits, **kwargs):
        assert len(limits) == 4
        kwargs.setdefault('PassMethod', 'AperturePass')
        super(Aperture, self).__init__(0, **kwargs)
        self.Limits = numpy.array(limits, dtype=numpy.float64)


class Drift(Element):
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length']

    def __init__(self, length, **kwargs):
        kwargs.setdefault('PassMethod', 'DriftPass')
        super(Drift, self).__init__(length, **kwargs)


class Magnet(Element):
    REQUIRED_ATTRIBUTES = Element.REQUIRED_ATTRIBUTES + ['Length']

    def __init__(self, length, **kwargs):
        mxo = kwargs.pop('MaxOrder', 3)
        numintsteps = kwargs.pop('NumIntSteps', 10)
        poly_a = kwargs.pop('PolynomA', numpy.zeros(mxo + 1))
        poly_b = kwargs.pop('PolynomB', numpy.zeros(mxo + 1))
        super(Magnet, self).__init__(length, **kwargs)
        self.PolynomA = numpy.array(poly_a, dtype=numpy.float64)
        self.PolynomB = numpy.array(poly_b, dtype=numpy.float64)
        self.MaxOrder = int(mxo)
        self.NumIntSteps = int(numintsteps)


class Dipole(Magnet):
    REQUIRED_ATTRIBUTES = Magnet.REQUIRED_ATTRIBUTES + ['BendingAngle',
                                                        'EntranceAngle',
                                                        'ExitAngle']

    def __init__(self, length, bending_angle, entrance_angle, exit_angle,
                 **kwargs):
        kwargs.setdefault('PassMethod', 'BndMPoleSymplectic4E2Pass')
        super(Dipole, self).__init__(length, **kwargs)
        self.BendingAngle = bending_angle
        self.EntranceAngle = entrance_angle
        self.ExitAngle = exit_angle


class Quadrupole(Magnet):

    def __init__(self, length, **kwargs):
        kwargs.setdefault('PassMethod', 'QuadLinearPass')
        kwargs.setdefault('MaxOrder', 1)
        super(Quadrupole, self).__init__(length, **kwargs)


class Sextupole(Magnet):

    def __init__(self, length, **kwargs):
        kwargs.setdefault('PassMethod', 'StrMPoleSymplectic4Pass')
        kwargs.setdefault('MaxOrder', 2)
        super(Sextupole, self).__init__(length, **kwargs)
