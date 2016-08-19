# To run with dls-python, you need to use the full path.
from pkg_resources import require
require('numpy')
from distutils.core import setup, Extension
import numpy

at = Extension('at', sources=['at.c'], define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

setup(name='at', ext_modules=[at], include_dirs=[numpy.get_include()])
