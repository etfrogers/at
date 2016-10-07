from distutils.core import setup, Extension
import numpy
import sys
import os
import glob

macros=[]

mach_home = os.getenv('MACH_HOME')
mach_arch = os.getenv('MACHARCH')
pyversion = "python{0.major}.{0.minor}".format(sys.version_info)
if (mach_home is not None and mach_arch is not None):
    integrator_path = '"' + os.path.join(mach_home, 'lib', mach_arch, pyversion, 'atintegrators') + '"'
    macros.append(('INTEGRATOR_PATH', integrator_path))
else:
    integrator_path = '../atintegrators'

at = Extension('at', sources=['at.c'],
               define_macros=macros,
               include_dirs=[numpy.get_include(), '../atintegrators'])

setup(name='at', py_modules=['elements','load_mat'], ext_modules=[at])

for pass_method in glob.glob(os.path.join(integrator_path, '*Pass.c')):
    print(pass_method)
    setup(name=pass_method[:-2], ext_modules=[Extension(name=pass_method[:-2], sources=[pass_method], define_macros=[('PYAT', None)], include_dirs=[numpy.get_include(), '../atintegrators'])])
