# Stommel gyre test case in 3D
# ============================
#
# Wind-driven geostrophic gyre in larege basin.
# Setup is according to [1]. This version us for 3D equations. As the problem
# is purely baroclinic the solution is the same as in 2D.
#
# [1] Comblen, R., Lambrechts, J., Remacle, J.-F., and Legat, V. (2010).
#     Practical evaluation of five partly discontinuous finite element pairs
#     for the non-conservative shallow water equations. International Journal
#     for Numerical Methods in Fluids, 63(6):701-724.
#
# Tuomas Karna 2015-04-28

from thetis import *

mesh2d = Mesh('stommel_square.msh')
outputdir = 'outputs'
print_info('Loaded mesh '+mesh2d.name)
print_info('Exporting to '+outputdir)
depth = 1000.0
layers = 6
t_end = 75*12*2*3600.
t_export = 3600.*2

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
coriolis_2d = Function(P1_2d)
f0, beta = 1.0e-4, 2.0e-11
coriolis_2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0))

# Wind stress
wind_stress_2d = Function(P1v_2d, name='wind stress')
tau_max = 0.1
L_y = 1.0e6
wind_stress_2d.interpolate(Expression(('tau_max*sin(pi*x[1]/L)', '0'), tau_max=tau_max, L=L_y))

# linear dissipation: tau_bot/(h*rho) = -bf_gamma*u
lin_drag = Constant(1e-6)

# --- create solver ---
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.cfl_2d = 1.0
options.nonlin = False
options.solve_salt = False
options.solve_temp = False
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False
# options.use_mode_split = False
options.baroclinic = False
options.coriolis = coriolis_2d
options.wind_stress = wind_stress_2d
options.lin_drag = lin_drag
options.t_export = t_export
options.t_end = t_end
options.dt_2d = 20.0
options.dt = 450.0
options.outputdir = outputdir
options.u_advection = Constant(0.01)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'uv_dav_2d']
options.timer_labels = []

solver_obj.iterate()
