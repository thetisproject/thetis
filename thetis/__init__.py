from __future__ import absolute_import
from thetis.utility import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator  # NOQA
import thetis.solver as solver  # NOQA # hydrostatic version
import thetis.solver2d as solver2d  # NOQA # hydrostatic version

# non-hydrostatic (nh) modelling
import thetis.nh_extension.solver_nh as solver_nh # direct nh extension
import thetis.nh_extension.solver_sigma as solver_sigma # sigma nh extension
import thetis.nh_extension.solver_ml as solver_ml  # multi-layer nh extension (2d horizontal mesh)
import thetis.nh_extension.solver2d_ml as solver2d_ml  # multi-layer nh extension (1d horizontal mesh)

from thetis.callback import DiagnosticCallback, DetectorsCallback  # NOQA
import thetis.limiter as limiter      # NOQA
import thetis.interpolation as interpolation      # NOQA
import thetis.coordsys as coordsys      # NOQA
import thetis.timezone as timezone      # NOQA
from thetis._version import get_versions
from thetis.assembledschur import AssembledSchurPC  # NOQA

__version__ = get_versions()['version']
del get_versions

parameters['pyop2_options']['lazy_evaluation'] = False

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
