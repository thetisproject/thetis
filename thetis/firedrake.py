# Import firedrake stuff here so that adjoint mode works
from __future__ import absolute_import

import thetis_config

from firedrake import *  # NOQA

# https://github.com/firedrakeproject/tsfc/issues/103
parameters["coffee"]["optlevel"] = "O0"
parameters["pyop2_options"]["opt_level"] = "O0"
if thetis_config.adjoint:
    from firedrake_adjoint import *  # NOQA
