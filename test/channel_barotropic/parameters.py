"""
Constants and parameters.

Tuomas Karna 2015-02-23
"""
from firedrake import *

# gravitational acceleration
g_grav = 9.81
# reference water density
rho = 1000

# beta plane approximation for coriolis: f=f0+beta*y
f0 = 1e-4
beta = 2e-11

# manning coefficient for bottom friction
#mu_manning = Constant(0.02)
mu_manning = Constant(0.0)

# horizontal viscosity (background value)
viscosity = Constant(10.0)

# wetting-drying parameters
wd_alpha = 0.3