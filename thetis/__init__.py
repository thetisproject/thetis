from __future__ import absolute_import
from thetis.utility import *
from thetis.log import *
import thetis.timeintegrator as timeintegrator
import thetis.solver as solver  # NOQA
import thetis.solver2d as solver2d  # NOQA
from thetis.callback import DiagnosticCallback  # NOQA
import thetis.limiter as limiter      # NOQA

op2.init(log_level=WARNING)
parameters['assembly_cache']['enabled'] = False
