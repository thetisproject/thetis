from thetis.utility import *
from thetis.utility3d import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator  # NOQA
import thetis.solver as solver  # NOQA
import thetis.solver2d as solver2d  # NOQA
from thetis.callback import DiagnosticCallback, DetectorsCallback  # NOQA
from thetis.callback import TimeSeriesCallback2D, TimeSeriesCallback3D  # NOQA
from thetis.callback import VerticalProfileCallback  # NOQA
import thetis.limiter as limiter      # NOQA
import thetis.interpolation as interpolation      # NOQA
import thetis.coordsys as coordsys      # NOQA
import thetis.timezone as timezone      # NOQA
import thetis.turbines  # NOQA
import thetis.optimisation  # NOQA
import thetis.diagnostics  # NOQA
from thetis._version import get_versions
from thetis.assembledschur import AssembledSchurPC  # NOQA
from thetis.options import TidalTurbineFarmOptions  # NOQA
import os  # NOQA
import datetime  # NOQA
import numpy  # NOQA

__version__ = get_versions()['version']
del get_versions

thetis_log_level(DEBUG)
set_thetis_loggers(comm=COMM_WORLD)
