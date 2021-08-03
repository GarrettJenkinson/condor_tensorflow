import setuptools
from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(_here, 'condor_tensorflow', 'version.py')) as f:
    exec(f.read(), version)


with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'condor_tensorflow',
    url = 'https://github.com/GarrettJenkinson/condor_tensorflow',
    author = 'GarrettJenkinson',
    author_email = 'Jenkinson.William@mayo.edu',
    packages = setuptools.find_packages(),
    install_requires = ['scikit-learn', 'numpy', 'tensorflow>=2.2'],
    version = version['__version__'],
    long_description_content_type = "text/markdown",
    license = 'MIT',
    description = 'Tensorflow Keras implementation of CONDOR ordinal regression loss, activation, and metrics',
    long_description = long_description,
    python_requires = '>=3.6',
)
