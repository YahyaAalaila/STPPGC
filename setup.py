# setup.py
import numpy
from setuptools import setup, find_packages, Extension

ext_modules = [
  Extension(
    "lib.neural_stpp.data_utils_fast",
    ["src/lib/neural_stpp/data_utils_fast.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
  )
]

setup(
  name="lightning-stpp",
  version="0.0.1",
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  ext_modules=ext_modules,
  install_requires=["torch>=2.2", "lightning>=2.2", "ray[tune]>=2.9"],
)
