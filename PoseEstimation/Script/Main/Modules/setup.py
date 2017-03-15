import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='features_labels',
    ext_modules=cythonize('features_labels.pyx', include_dirs=[numpy.get_include()])
)
