from __future__ import absolute_import
from thetis.utility import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator  # NOQA
import thetis.solver as solver  # NOQA
import thetis.solver2d as solver2d  # NOQA
from thetis.callback import DiagnosticCallback, DetectorsCallback  # NOQA
import thetis.limiter as limiter      # NOQA
import thetis.interpolation as interpolation      # NOQA
import thetis.coordsys as coordsys      # NOQA
import thetis.timezone as timezone      # NOQA
import thetis.turbines  # NOQA
import thetis.optimisation  # NOQA
from thetis._version import get_versions
from thetis.assembledschur import AssembledSchurPC  # NOQA
from thetis.options import TidalTurbineFarmOptions  # NOQA
__version__ = get_versions()['version']
del get_versions

parameters['pyop2_options']['lazy_evaluation'] = False

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
