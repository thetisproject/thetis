# Copyright (C) 2015-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections import OrderedDict
from dolfin import compile_cpp_code


def _get_cpp_module(cpp_files, force_recompile=False):
    """
    Use the dolfin machinery to compile, wrap with swig and load a c++ module
    """
    cpp_dir = os.path.dirname(os.path.abspath(__file__))

    cpp_sources = []
    for cpp_filename in cpp_files:
        lines = []
        cpp_path = os.path.join(cpp_dir, cpp_filename)
        with open(cpp_path, 'rt') as cpp_file_obj:
            for line in cpp_file_obj:
                if line.startswith('#include') and 'remove_in_jit' in line:
                    pass
                else:
                    lines.append(line)
        cpp_sources.append(''.join(lines))

    # Force recompilation
    if force_recompile:
        cpp_sources.append('// Force recompile, time is %s \n' % time.time())

    sep = '\n\n// ' + '$' * 77 + '\n\n'
    cpp_code = sep.join(cpp_sources)

    module = compile_cpp_code(cpp_code)
    assert module is not None

    return module


class _ModuleCache(object):
    def __init__(self):
        """
        A registry and cache of available C/C++ extension modules
        """
        self.available_modules = OrderedDict()
        self.module_cache = {}

    def add_module(self, name, cpp_files, test_compile=True):
        """
        Add a module that can be compiled
        """
        self.available_modules[name] = cpp_files

        if test_compile:
            # Compile at once to test the code
            self.get_module(name)

    def get_module(self, name, force_recompile=False):
        """
        Compile and load a module (first time) or use from cache (subsequent requests)
        """
        if force_recompile or name not in self.module_cache:
            cpp_files = self.available_modules[name]
            mod = _get_cpp_module(cpp_files, force_recompile)
            self.module_cache[name] = mod

        return self.module_cache[name]


###############################################################################################
# Functions to be used by other modules

_MODULES = _ModuleCache()
_MODULES.add_module('naive_nodal', ['slope_limiter/naive_nodal.h'])
_MODULES.add_module(
    'hierarchical_taylor', ['slope_limiter/limiter_common.h', 'slope_limiter/hierarchical_taylor.h']
)
_MODULES.add_module('measure_local_maxima', ['slope_limiter/measure_local_maxima.h'])
_MODULES.add_module('linear_convection', ['gradient_reconstruction.h', 'linear_convection.h'])


def load_module(name, force_recompile=False):
    """
    Load the C/C++ module registered with the given name. Reload
    forces a cache-refresh, otherwise subsequent accesses are cached
    """
    return _MODULES.get_module(name, force_recompile)
