from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules=cythonize('FunkSVD_sgd.pyx'),
    include_dirs=[numpy.get_include()]
)

#to run in console
#python compile_cython.py build_ext --inplace
