#!/usr/bin/env python

from setuptools import setup, find_packages, find_namespace_packages

setup(name="NISS",
      version="0.1",
      description="Package to develop neural informed sparse solvers",
      author="Rui Peng Li",
      author_email="li50@llnl.gov",
      packages=find_namespace_packages(where='niss'),
      # namespace_packages=['NISS'],
      package_dir={"": "niss"},
      install_requires=['torch',
                        'pytest',
                        'matplotlib',
                        'numpy'],
      python_requires='>=3.8'
      )
