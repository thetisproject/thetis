from utility import *
import timeintegrator  # NOQA
import solver  # NOQA
import solver2d  # NOQA
from callback import DiagnosticCallback  # NOQA
from log import *

op2.init(log_level=WARNING)
parameters['assembly_cache']['enabled'] = False
