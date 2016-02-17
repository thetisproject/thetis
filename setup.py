#!/usr/bin/env python

from distutils.core import setup
from glob import glob

setup(name='thetis',
      version='0.1',
      description='Finite element ocean model',
      author='Tuomas Karna',
      author_email='tuomas.karna@gmail.com',
      url='https://github.com/thetisproject/thetis',
      packages=['thetis', 'test', 'examples'],
      scripts=glob('scripts/*'),
     )
