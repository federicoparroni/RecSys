"""
To compile, run:
    python 'this/file/path' build_ext --inplace
"""
from distutils.core import setup
from Cython.Build import cythonize

#if __name__ == "__main__":

setup(name='seq_similarity',
        ext_modules=cythonize('seq_similarity.pyx',
                                annotate=True,          # generate html annotation file
                                compiler_directives={'cdivision': True}
                        )
)