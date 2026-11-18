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
from thetis.assembledschur import AssembledSchurPC  # NOQA
from thetis.options import TidalTurbineFarmOptions, DiscreteTidalTurbineFarmOptions  # NOQA
from importlib.metadata import PackageNotFoundError, version as _pkg_version
import os  # NOQA
import datetime  # NOQA
import numpy  # NOQA

try:
    __version__ = _pkg_version("thetis")
except PackageNotFoundError:
    __version__ = "0+unknown"
del _pkg_version, PackageNotFoundError

thetis_log_level(DEBUG)
set_thetis_loggers(comm=COMM_WORLD)
