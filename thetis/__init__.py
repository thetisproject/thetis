from utility import *
import timeintegrator  # NOQA
import solver  # NOQA
import solver2d  # NOQA
from callback import DiagnosticCallback  # NOQA

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
parameters['assembly_cache']['enabled'] = False
