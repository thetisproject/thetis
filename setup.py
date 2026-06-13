#!/usr/bin/env python

from distutils.core import setup
from glob import glob

setup(scripts=glob('scripts/*'))
