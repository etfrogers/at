from distutils.core import setup, Extension
import numpy
import os

macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

mach_home = os.getenv('MACH_HOME')
mach_arch = os.getenv('MACHARCH')
if (mach_home is not None and mach_arch is not None):
    integrator_path = '"' + os.path.join(mach_home, 'lib', mach_arch, 'python2.7', 'atintegrators') + '"'
    macros.append(('INTEGRATOR_PATH', integrator_path))
else:
    integrator_path = '../atintegrators'


print(macros)
at = Extension('at', sources=['at.c'],
               define_macros=macros,
               include_dirs=[numpy.get_include(), integrator_path])

setup(name='at', py_modules=['elements'], ext_modules=[at])
