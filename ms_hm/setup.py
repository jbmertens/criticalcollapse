from setuptools import Extension, setup
from Cython.Build import cythonize


setup(ext_modules=cythonize([
    Extension(
        "utils",
        ["cython/utils.pyx"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp"],
        language="c++",
    )
],
                            language_level=3))
