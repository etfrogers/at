import numpy


class Element(object):
    def __init__(self, name, length=0):
        self.name = name
        self.length = length
        self.pass_method = 'IdentityPass'


class Drift(Element):
    def __init__(self, name, length):
        super(Drift, self).__init__(name, length)
        self.pass_method = 'DriftPass'


class Magnet(Element):
    def __init__(self, name, length):
        super(Magnet, self).__init__(name, length)
        self.polynom_a = numpy.array([0,0,0,0])
        self.polynom_b = numpy.array([0,0,0,0])


class Quad(Magnet):
    def __init__(self, name, length, k1):
        super(Quad, self).__init__(name, length)
        self.polynom_b[1] = k1
        self.pass_method = 'QuadLinearPass'


class Sext(Magnet):
    def __init__(self, name, length, k2):
        super(Sext, self).__init__(name, length)
        self.polynom_b[2] = k2
        self.pass_method = 'StrMPoleSymplectic4Pass'
