"""
Default values for common constants and parameters in a dictionary.

Tuomas Karna 2015-02-23
"""
from firedrake import *

physical_constants = \
    {'g_grav': Constant(9.81),      # gravitational acceleration
     'rho0': Constant(1000.0),      # reference water density
     'f0': Constant(1e-4),          # beta plane approximation
     'betas': Constant(2e-11),      # for coriolis: f=f0+beta*y
     'mu_manning': Constant(0.0),   # manning bottom friction coefficient
     'z0_friction': Constant(0.0),  # bot roughness length for 3D model
     'von_karman': Constant(0.4),   # von Karman constant for bottom log layer
     'viscosity': Constant(0.0),    # horizontal viscosity
     'wd_alpha': Constant(0.3),     # wetting-dryinh depth parameter
     }


#g_grav = 9.81
#rho = 1000

#f0 = 1e-4
#beta = 2e-11

## manning coefficient for bottom friction
##mu_manning = 0.02
#mu_manning = 0.0

## horizontal viscosity (background value)
#viscosity = 10.0

## wetting-drying parameters
#wd_alpha = 0.3
