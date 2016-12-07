from __future__ import absolute_import
from thetis.utility import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator  # NOQA
import thetis.solver as solver  # NOQA
import thetis.solver2d as solver2d  # NOQA
from thetis.callback import DiagnosticCallback  # NOQA
import thetis.limiter as limiter      # NOQA
from thetis.assembledschur import AssembledSchurPC

parameters['assembly_cache']['enabled'] = False
