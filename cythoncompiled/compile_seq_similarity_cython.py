#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To compile, run:
    python 'this/file/path' build_ext --inplace
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy

ext_modules = Extension('seq_similarity',
                ['seq_similarity.pyx'],
                extra_compile_args=['-O3'],
                include_dirs=[numpy.get_include(),],
                )

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_modules]
)



"""
from distutils.core import setup
#from distutils.extension import Extension
from Cython.Build import cythonize

if __name__ == "__main__":
    setup(name='seq_similarity',
            ext_modules=cythonize('seq_similarity.pyx',
                                    #annotate=True,          # generate html annotation file
                                    compiler_directives={'cdivision': True}
                            )
    )
"""