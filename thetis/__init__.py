from __future__ import absolute_import
from thetis.utility import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator  # NOQA
import thetis.solver as solver  # NOQA
import thetis.solver2d as solver2d  # NOQA
import thetis.solver_nh as solver_nh
import thetis.solver2d_nh as solver2d_nh  # NOQA
import thetis.solver1d_nh as solver1d_nh  # NOQA
import thetis.solver_sigma as solver_sigma # NOQA
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
