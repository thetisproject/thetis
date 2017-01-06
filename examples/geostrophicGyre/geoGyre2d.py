# Geostrophic gyre test case in 2D
# ================================
#
# Stationary gyre test case according to [1].
# Initial condition for elevation is Gaussian bell funcition.
# initial velocity is obtained from analytical solution corresponding to
# geostrophic balance. The model should retain the initial solution
# indefinitely long time.
#
#
#
# Tuomas Karna 2015-04-28

from thetis import *

# set physical constants
physical_constants['z0_friction'].assign(0.0)

mesh2d = Mesh('stommel_square.msh')
nonlin = False
depth = 1000.0
elev_amp = 3.0
outputdir = 'outputs'
t_end = 75*12*2*3600
t_export = 3600*2

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
coriolis_2d = Function(P1_2d)
f0, beta = 1.0e-4, 0.0
coriolis_2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0))

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.nonlin = False
options.coriolis = coriolis_2d
options.t_export = t_export
options.t_end = t_end
options.dt = 20.0
options.outputdir = outputdir
options.u_advection = Constant(0.01)
options.check_vol_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.timestepper_type = 'CrankNicolson'

solver_obj.create_equations()
sigma = 160.0e3
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.project(Expression('eta_amp*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)', eta_amp=elev_amp, s=sigma))

# initial velocity: u = -g/f deta/dy, v = g/f deta/dx
uv_init = Function(solver_obj.function_spaces.U_2d)
uv_init.project(Expression(('g/f*eta_amp*2*x[1]/s/s*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)',
                            '-g/f*eta_amp*2*x[0]/s/s*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)'), eta_amp=elev_amp, s=sigma, g=9.81, f=f0))

solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)

solver_obj.iterate()
