"""
The python extension method.
"""
import sys
sys.path.append('./build/lib.linux-x86_64-2.7/')
import dls_packages
import numpy
import pml
import aphla as ap
import at
import time

pml.initialise('SRI21')
the_ring = ap.getElements('*')

PASS_METHODS = {'SEXT': 'StrMPoleSymplectic4Pass',
                'QUAD': 'QuadLinearPass',
                'BEND': 'BndMPoleSymplectic4E2Pass',
                'DRIFT': 'DriftPass'}

for element in the_ring:
    family = element.family
    if family == 'BEND':
        element.bending_angle = 0.1309
        element.entrance_angle = 0.0654
        element.exit_angle = 0.0654
        element.gap = 0.0466
        element.fringe_int_1 = 0.6438
        element.fringe_int_2 = 0.6438
        element.max_order = 3
        element.num_int_steps = 10
        element.polynom_a = numpy.array([0,0,0,0])
        element.polynom_b = numpy.array([0,0,0,0])
    elif family == 'SEXT':
        element.max_order = 3
        element.num_int_steps = 10
        element.polynom_a = numpy.array([0,0,0,0])
        element.polynom_b = numpy.array([0,0,element.k2,0])
    try:
        element.pass_method = PASS_METHODS[family]
    except KeyError:
        element.pass_method = 'DriftPass'

rin = numpy.zeros((1, 6))
rin[0,0] = 1e-6
print(rin)
t = time.time()
at.atpass(the_ring, rin, 1000)
print("Time taken: {}".format(time.time() - t))

print(rin)
