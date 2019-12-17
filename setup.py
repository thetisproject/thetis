#!/usr/bin/env python

from distutils.core import setup
from glob import glob

import versioneer

cmdclass = versioneer.get_cmdclass()

setup(name='thetis',
      cmdclass=cmdclass,
      version=versioneer.get_version(),
      description='Finite element ocean model',
      author='Tuomas Karna',
      author_email='tuomas.karna@gmail.com',
      url='https://github.com/thetisproject/thetis',
      packages=['thetis', 'test', 'examples'],
      scripts=glob('scripts/*'),
     )
