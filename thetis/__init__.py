from utility import *
import timeintegrator  # NOQA
import solver  # NOQA
import solver2d  # NOQA

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
parameters['pyop2_options']['profiling'] = True
parameters['assembly_cache']['enabled'] = False
