import os
from subprocess import check_output
import sys

# Numpy/mpi4py must be installed prior to installing aerodesk
import numpy

# import mpi4py

# Import distutils
from setuptools import setup, find_packages

setup(
    name="ml_buckling",
    version="0.1",
    description="Machine learning surrogate models for buckling",
    long_description_content_type="text/markdown",
    author="Sean P. Engelstad",
    author_email="sengeltad312@gatech.edu",
    install_requires=["numpy", "tacs>3.7.0"],
    packages=find_packages(include=["ml_buckling*"]),
)
