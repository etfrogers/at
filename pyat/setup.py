from setuptools import setup, Extension
import numpy
import sys
import os
import glob


macros = [('PYAT', None)]

integrator_src = os.path.abspath('../atintegrators')

cflags = []

if not sys.platform.startswith('win32'):
    cflags += ['-Wno-unused-function']


def integrator_extension(pass_method):
    name = ".".join(('at', 'integrators', os.path.basename(pass_method)[:-2]))
    return Extension(name=name,
                     sources=[pass_method],
                     include_dirs=[numpy.get_include(), integrator_src],
                     define_macros=macros,
                     extra_compile_args=cflags)


integ_list = glob.glob(os.path.join(integrator_src, '*Pass.c'))

at = Extension('at.atpass',
               sources=['at.c'],
               define_macros=macros,
               include_dirs=[numpy.get_include(), integrator_src],
               extra_compile_args=cflags)

setup(name='at',
      version='0.0.1',
      description='Accelerator Toolbox',
      install_requires=['numpy'],
      package_dir={'at': ''},
      packages=['at', 'at.integrators'],
      ext_modules=[at] + [integrator_extension(pm) for pm in integ_list],
      zip_safe=False)
