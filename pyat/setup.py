from distutils.core import setup, Extension
import numpy
import sys
import os
import glob

macros=[]

mach_home = os.getenv('MACH_HOME')
mach_arch = os.getenv('MACHARCH')
pyversion = "python{0.major}.{0.minor}".format(sys.version_info)

integrator_src = os.path.abspath('../atintegrators')
integrator_build = None

for pass_method in glob.glob(os.path.join(integrator_src, '*Pass.c')):
    print(pass_method)
    x = setup(name=pass_method[:-2], ext_modules=[Extension(name=pass_method[:-2], sources=[pass_method], define_macros=[('PYAT', None)], include_dirs=[numpy.get_include(), integrator_src])])
    print(x.command_obj['install'].install_platlib)
    if integrator_build is not None:
        integrator_build = x.command_obj['install'].install_platlib
        macros.append(('INTEGRATOR_PATH', integrator_build))

at = Extension('at', sources=['at.c'],
               define_macros=macros,
               include_dirs=[numpy.get_include(), integrator_src])

setup(name='at', py_modules=['elements','load_mat'], ext_modules=[at])

