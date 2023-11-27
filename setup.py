#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='sae',
    version='0.1.0',
    description='An implementation of an SAE systems design problem',
    author='Akash Agrawal',
    author_email='ask-drc@andrew.cmu.edu',
    url='',
    install_requires=["numpy", "scipy", "pandas", "openpyxl"],
    packages=['sae'],
    package_dir={'sae': 'src'},
    include_package_data=True,
)