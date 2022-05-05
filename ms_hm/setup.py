import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize


setup(ext_modules=cythonize([
    Extension(
        "utils",
        ["cython/utils.pyx"],
        extra_compile_args=["-O3"],
        extra_link_args=["-fopenmp"],
        language="c",
        include_dirs=[numpy.get_include()]
    )
],
                            language_level=3))
