import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize("subtraction_utils.pyx"), include_dirs=[numpy.get_include()])
