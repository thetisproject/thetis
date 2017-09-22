"""
Columbia river plume simulation
===============================
"""
from thetis import *
from bathymetry import get_bathymetry, smooth_bathymetry, smooth_bathymetry_at_bnd
comm = COMM_WORLD

# TODO add time-dependent river discharge
# TODO add tidal elevation netdf reader
# TODO add wind stress formulations
# TODO add atm netcdf reader, time dependent wind stress
# TODO add intial condition from ROMS/HYCOM

rho0 = 1000.0
# set physical constants
physical_constants['rho0'].assign(rho0)
physical_constants['z0_friction'].assign(0.005)

reso_str = 'coarse'
nlayers = 9
outputdir = 'outputs_{:}'.format(reso_str)
mesh2d = Mesh('mesh_cre-plume002.msh')
print_output('Loaded mesh ' + mesh2d.name)

nnodes = comm.allreduce(mesh2d.topology.num_vertices(), MPI.SUM)
ntriangles = comm.allreduce(mesh2d.topology.num_cells(), MPI.SUM)
nprisms = ntriangles*nlayers

t_end = 10*24*3600.
t_export = 900.

# interpolate bathymetry and smooth it
bathymetry_2d = get_bathymetry('bathymetry_300m.npz', mesh2d, project=False)
bathymetry_2d = smooth_bathymetry(
    bathymetry_2d, delta_sigma=1.0, bg_diff=0,
    alpha=5e6, exponent=1,
    minimum_depth=5., niter=20)
bathymetry_2d = smooth_bathymetry_at_bnd(bathymetry_2d, [1, 3])

# 3d mesh vertical stretch factor
z_stretch_fact_2d = Function(bathymetry_2d.function_space(), name='z_stretch')
# 1.0 (sigma mesh) in shallow areas, 4.0 in deep ocean
max_z_stretch = 4.
max_bath = 1800.
surf_elem_height = 5.0
z_stretch_fact_2d.project(-ln(surf_elem_height/bathymetry_2d)/ln(nlayers))
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data < 1.0] = 1.0
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data > max_z_stretch] = max_z_stretch

coriolis_f, coriolis_beta = beta_plane_coriolis_params(46.25)
q_river = 5000.
salt_river = 0.0
salt_ocean_surface = 32.0
salt_ocean_bottom = 34.0
temp_river = 15.0
temp_ocean_surface = 13.0
temp_ocean_bottom = 8.0
reynolds_number = 160.0

eta_amplitude = 1.00
eta_phase = 0
H_ocean = 200  # ~mean water depth in coast
Ttide = 44714.0  # M2 tidal period [2]
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H_ocean)  # [m/s] wave speed
kelvin_k = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvin_m = (coriolis_f/c)  # [-] Cross-shore variation

u_scale = 3.0
w_scale = 1e-3
delta_x = 2e3
nu_scale = u_scale * delta_x / reynolds_number

simple_barotropic = False  # for debugging

# create solver
extrude_options = {
    'z_stretch_fact': z_stretch_fact_2d,
}

solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers,
                               extrude_options=extrude_options)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'ssprk22'
options.solve_salt = not simple_barotropic
options.solve_temp = not simple_barotropic
options.solve_vert_diffusion = True  # not simple_barotropic
options.use_bottom_friction = True  # not simple_barotropic
options.use_turbulence = True  # not simple_barotropic
options.use_turbulence_advection = False  # not simple_barotropic
options.use_smooth_eddy_viscosity = False
options.turbulence_model = 'pacanowski'
options.baroclinic = not simple_barotropic
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
options.v_viscosity = Constant(2e-5)
options.v_diffusivity = Constant(2e-5)
options.h_viscosity = Constant(2.0)
options.h_diffusivity = Constant(2.0)
options.use_quadratic_pressure = True
options.use_limiter_for_tracers = True
options.use_limiter_for_velocity = True
options.smagorinsky_factor = Constant(1.0/np.sqrt(reynolds_number))
options.coriolis = Constant(coriolis_f)
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = Constant(u_scale)
options.w_advection = Constant(w_scale)
options.nu_viscosity = Constant(nu_scale)
options.check_salt_overshoot = True
options.check_temp_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'density_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d',
                            'int_pg_3d', 'hcc_metric_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
options.equation_of_state = 'full'
# gls_options = options.gls_options
# gls_options.apply_defaults('k-omega')
# gls_options.stability_name = 'CB'
options.pacanowski_options['alpha'] = 10.
options.pacanowski_options['exponent'] = 2
options.pacanowski_options['max_viscosity'] = 0.05

solver_obj.create_function_spaces()

xyz = SpatialCoordinate(solver_obj.mesh)
# vertical stratification in the ocean
river_blend = (1 + tanh((xyz[0] - 350e3)/2000.))/2  # 1 in river, 0 in ocean
salt_stratif = salt_ocean_bottom + (salt_ocean_surface - salt_ocean_bottom)*(1 + tanh((xyz[2] + 1200.)/700.))/2
salt_expr = salt_river*river_blend + (1 - river_blend)*salt_stratif
salt_init_3d = Function(solver_obj.function_spaces.H, name='initial salinity')
salt_init_3d.interpolate(salt_expr)

temp_stratif = temp_ocean_bottom + (temp_ocean_surface - temp_ocean_bottom)*(1 + tanh((xyz[2] + 50.)/15.))/2
temp_expr = temp_river*river_blend + (1 - river_blend)*temp_stratif
temp_init_3d = Function(solver_obj.function_spaces.H, name='initial temperature')
temp_init_3d.interpolate(temp_expr)

elev_init = Function(solver_obj.function_spaces.H_2d, name='initial elevation')

x0 = 330000.
xy = SpatialCoordinate(mesh2d)
bnd_time = Constant(0)
elev_expr = eta_amplitude*conditional(le(xy[0], x0),
                                      exp((xy[0]-x0)*kelvin_m)*cos(xy[1]*kelvin_k - OmegaTide*bnd_time),
                                      cos(xy[1]*kelvin_k - OmegaTide*bnd_time))
elev_init.interpolate(elev_expr)

fs_2d = bathymetry_2d.function_space()
bnd_elev = Function(fs_2d, name='Boundary elevation')

xyz = solver_obj.mesh2d.coordinates
tri = TrialFunction(fs_2d)
test = TestFunction(fs_2d)
ramp_t = 12*3600.
elev_ramp = conditional(le(bnd_time, ramp_t), bnd_time/ramp_t, 1.0)
elev_bnd_expr = elev_ramp*elev_expr
a = inner(test, tri)*dx
L = test*elev_bnd_expr*dx
bnd_elev_prob = LinearVariationalProblem(a, L, bnd_elev)
bnd_elev_solver = LinearVariationalSolver(bnd_elev_prob)
bnd_elev_solver.solve()

north_bnd_id = 1
west_bnd_id = 2
south_bnd_id = 3
river_bnd_id = 4
river_swe_funcs = {'flux': Constant(-q_river)}
tide_elev_funcs = {'elev': bnd_elev}
zero_elev_funcs = {'elev': Constant(0)}
open_uv_funcs = {'symm': None}
zero_uv_funcs = {'uv': Constant((0, 0, 0))}
bnd_river_salt = {'value': Constant(salt_river)}
ocean_salt_funcs = {'value': salt_init_3d}
bnd_river_temp = {'value': Constant(temp_river)}
ocean_temp_funcs = {'value': temp_init_3d}
solver_obj.bnd_functions['shallow_water'] = {
    river_bnd_id: river_swe_funcs,
    south_bnd_id: tide_elev_funcs,
    north_bnd_id: tide_elev_funcs,
    west_bnd_id: tide_elev_funcs,
}
solver_obj.bnd_functions['momentum'] = {
    river_bnd_id: open_uv_funcs,
    south_bnd_id: zero_uv_funcs,
    north_bnd_id: zero_uv_funcs,
    west_bnd_id: zero_uv_funcs,
}
solver_obj.bnd_functions['salt'] = {
    river_bnd_id: bnd_river_salt,
    south_bnd_id: ocean_salt_funcs,
    north_bnd_id: ocean_salt_funcs,
    west_bnd_id: ocean_salt_funcs,
}
solver_obj.bnd_functions['temp'] = {
    river_bnd_id: bnd_river_temp,
    south_bnd_id: ocean_temp_funcs,
    north_bnd_id: ocean_temp_funcs,
    west_bnd_id: ocean_temp_funcs,
}

solver_obj.create_equations()


hcc_obj = Mesh3DConsistencyCalculator(solver_obj)
hcc_obj.solve()

print_output('Running CRE plume with options:')
print_output('Resolution: {:}'.format(reso_str))
print_output('Element family: {:}'.format(options.element_family))
print_output('Polynomial order: {:}'.format(options.order))
print_output('Reynolds number: {:}'.format(reynolds_number))
print_output('Horizontal viscosity: {:}'.format(nu_scale))
print_output('Number of cores: {:}'.format(comm.size))
print_output('Number of 2D nodes={:}, triangles={:}, prisms={:}'.format(nnodes, ntriangles, nprisms))
print_output('Tracer DOFs: {:}'.format(6*nprisms))
print_output('Tracer DOFs per core: {:}'.format(float(6*nprisms)/comm.size))
print_output('Exporting to {:}'.format(outputdir))

solver_obj.assign_initial_conditions(salt=salt_init_3d, temp=temp_init_3d)


def show_uv_mag():
    uv = solver_obj.fields.uv_3d.dat.data
    print_output('uv: {:9.2e} .. {:9.2e}'.format(uv.min(), uv.max()))
    ipg = solver_obj.fields.int_pg_3d.dat.data
    print_output('int pg: {:9.2e} .. {:9.2e}'.format(ipg.min(), ipg.max()))


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_solver.solve()


solver_obj.iterate(update_forcings=update_forcings, export_func=show_uv_mag)
