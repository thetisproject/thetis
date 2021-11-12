"""
Default values for physical constants and parameters
"""
from firedrake import Constant

physical_constants = {
    'g_grav': Constant(9.81),      # gravitational acceleration
    'rho0': Constant(1000.0),      # reference water density
    'von_karman': Constant(0.4),   # von Karman constant for bottom log layer
    'rho_air': Constant(1.22),     # air density (kg/m3)
}

physical_constants['rho0_inv'] = \
    Constant(1.0/physical_constants['rho0'])
