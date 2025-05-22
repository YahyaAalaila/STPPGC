import numpy
from setuptools import setup, Extension

ext_modules = [
    Extension(
        "lib.neural_stpp.data_utils_fast",
        ["src/lib/neural_stpp/data_utils_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup( 
    # … copy the same metadata from your pyproject …
    ext_modules=ext_modules,
)
