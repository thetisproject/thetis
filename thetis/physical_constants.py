"""
Default values for physical constants and parameters
"""
from __future__ import absolute_import
from firedrake import Constant

# TODO this module should only contain scalar variables
# TODO all parameters that can be spatially varying should be in options

physical_constants = {
    'g_grav': Constant(9.81),      # gravitational acceleration
    'rho0': Constant(1000.0),      # reference water density
    'z0_friction': Constant(0.0),  # bot roughness length for 3D model
    'von_karman': Constant(0.4),   # von Karman constant for bottom log layer
    'rho_air': Constant(1.22),     # air density (kg/m3)
}

physical_constants['rho0_inv'] = \
    Constant(1.0/physical_constants['rho0'])
