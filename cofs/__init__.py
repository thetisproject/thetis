from utility import *
import timeIntegrator
import solver
import solver2d

parameters['coffee']={}  # omit COFFEE optimzations for now

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
parameters['pyop2_options']['profiling'] = True
