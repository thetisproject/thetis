# Rhine ROFI test case
# ====================
#
# Idealized Rhine river plume test case according to [2].
#
# Salinity ranges from 32 psu in ocean to 0 in river.
# Temperature is constant 10 deg Celcius. This corresponds to density
# values 1024.611 kg/m3 in the ocean and 999.702 in the river.
#
# [1] de Boer, G., Pietrzak, J., and Winterwerp, J. (2006). On the vertical
#     structure of the Rhine region of freshwater influence. Ocean Dynamics,
#     56(3):198-216.
# [2] Fischer, E., Burchard, H., and Hetland, R. (2009). Numerical
#     investigations of the turbulent kinetic energy dissipation rate in the
#     Rhine region of freshwater influence. Ocean Dynamics, 59:629-641.
#
# Tuomas Karna 2015-06-24

from cofs import *

outputdir = create_directory('outputs')
layers = 6
mesh2d = Mesh('mesh_rhine_rofi_coarse.msh')
print_info('Loaded mesh '+mesh2d.name)
print_info('Exporting to '+outputdir)

# Physical parameters
eta_amplitude = 1.00  # mean (Fisher et al. 2009 tidal range 2.00 )
eta_phase = 0
H = 20  # water depth
HInlet = 5  # water depth at river inlet
Lriver = 45e3
Qriver = 3.0e3  # 1.5e3 river discharge (Fisher et al. 2009)
Sriver = 0
Ssea = 32
density_river = 999.7
density_ocean = 1024.6
Ttide = 44714.0  # M2 tidal period (Fisher et al. 2009)
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaEarth = 2*np.pi/Tday
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H)  # [m/s] wave speed
lat_deg = 52.5  # latitude
phi = (np.pi/180)*lat_deg  # latitude in radians
coriolis_f = 2*OmegaEarth*sin(phi)  # [rad/s] Coriolis parameter ~ 1.1e-4
kelvin_k = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvin_m = (coriolis_f/c)  # [-] Cross-shore variation

dt = 8.0
t_end = 32*44714
t_export = 900.0  # 44714/12

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('(x[0] > 0.0) ? H*(1-x[0]/Lriver) + HInlet*(x[0]/Lriver) : H',
                                     H=H, HInlet=HInlet, Lriver=Lriver))

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.cfl_2d = 1.0
# options.nonlin = False
options.solve_salt = True
options.solve_vert_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False
# options.use_semi_implicit_2d = False
# options.use_mode_split = False
options.baroclinic = True
options.coriolis = Constant(coriolis_f)
options.use_supg = False
options.use_gjv = False
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
Re_h = 2.0
options.smagorinsky_factor = Constant(1.0/np.sqrt(Re_h))
options.salt_jump_diff_factor = Constant(1.0)
options.salt_range = Constant(25.0)
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
# options.h_viscosity = Constant(0.5*2000.0/refinement[reso_str]/Re_h)
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
# options.h_viscosity = Constant(100.0/refinement[reso_str])
# options.h_viscosity = Constant(10.0)
if options.use_mode_split:
    options.dt = dt
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = Constant(2.0)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_salt_conservation = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d', 'gjv_alpha_h_3d', 'gjv_alpha_v_3d',
                            'smag_visc_3d', 'salt_jump_diff']
options.timer_labels = []

bnd_elev = Function(P1_2d, name='Boundary elevation')
bnd_time = Constant(0)
xyz = solver_obj.mesh2d.coordinates
tri = TrialFunction(P1_2d)
test = TestFunction(P1_2d)
elev = eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*elev*dx
bnd_elev_prob = LinearVariationalProblem(a, L, bnd_elev)
bnd_elev_solver = LinearVariationalSolver(bnd_elev_prob)
bnd_elev_solver.solve()

fs = P1_2d
bnd_v = Function(fs, name='Boundary v velocity')
tri = TrialFunction(fs)
test = TestFunction(fs)
v = -(g*kelvin_k/OmegaTide)*eta_amplitude*exp(xyz[0]*kelvin_m)*cos(xyz[1]*kelvin_k - OmegaTide*bnd_time)
a = inner(test, tri)*dx
L = test*v*dx
bnd_v_prob = LinearVariationalProblem(a, L, bnd_v)
bnd_v_solver = LinearVariationalSolver(bnd_v_prob)
bnd_v_solver.solve()

river_discharge = Constant(-Qriver)
ocean_salt = Constant(density_ocean)
river_salt = Constant(density_river)
tide_elev_funcs = {'elev': bnd_elev}
tide_uv_funcs = {'un': bnd_v}
open_funcs = {'symm': None}
river_funcs = {'flux': river_discharge}
bnd_ocean_salt = {'value': ocean_salt}
bnd_river_salt = {'value': river_salt}
solver_obj.bnd_functions['shallow_water'] = {1: tide_elev_funcs, 2: tide_elev_funcs,
                                             3: tide_elev_funcs, 6: river_funcs}
# solver_obj.bnd_functions['momentum'] = {1: tide_funcs, 2: tide_funcs,
#                                         3: tide_funcs, 6: river_funcs}
solver_obj.bnd_functions['salt'] = {1: bnd_ocean_salt, 2: bnd_ocean_salt,
                                    3: bnd_ocean_salt, 6: bnd_river_salt}

solver_obj.create_equations()
bnd_elev_3d = Function(solver_obj.function_spaces.P1, name='Boundary elevation 3d')
copy_2d_field_to_3d(bnd_elev, bnd_elev_3d)
tide_elev_funcs_3d = {'elev': bnd_elev_3d}
solver_obj.eq_momentum.bnd_functions = {1: tide_elev_funcs_3d, 2: tide_elev_funcs_3d,
                                        3: tide_elev_funcs_3d, 6: river_funcs}

elev_init = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : amp*cos(x[1]*kelvin_k)',
                      amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
elev_init2 = Function(solver_obj.function_spaces.H_2d, name='initial elevation')
elev_init2.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : 0.0',
                       amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
uv_init = Function(solver_obj.function_spaces.U_2d, name='initial velocity')
# uv_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvin_m)*cos(x[1]*kelvin_k) : amp*cos(x[1]*kelvin_k)',
#                       amp=eta_amplitude, kelvin_m=kelvin_m, kelvin_k=kelvin_k))
tri = TrialFunction(solver_obj.function_spaces.U_2d)
test = TestFunction(solver_obj.function_spaces.U_2d)
a = inner(test, tri)*dx
uv = (g*kelvin_k/OmegaTide)*elev_init2
L = test[1]*uv*dx
solve(a == L, uv_init)
salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
salt_init3d.interpolate(Expression('d_ocean - (d_ocean - d_river)*(1 + tanh((x[0] - xoff)/sigma))/2',
                                   sigma=6000.0, d_ocean=density_ocean,
                                   d_river=density_river, xoff=20.0e3))


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_solver.solve()
    copy_2d_field_to_3d(bnd_elev, bnd_elev_3d)

solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d, uv_2d=uv_init)
solver_obj.iterate()
