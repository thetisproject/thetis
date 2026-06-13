#!/usr/bin/env python

from distutils.core import setup
from glob import glob
from os.path import isfile

setup(scripts=[path for path in glob('scripts/*') if isfile(path)])
