"""
Baroclinic channel test case following [1]

The domain extends from latitude 30 N to 45 N and is 20 deg long in zonal
direction. Periodic boundaries are used in eastern and western boundaries.
Depth is constant 1600 m. Mesh resolution is 1/6 deg in zonal and 1/7 deg
in meridional directions. 24 evenly-spaced vertical levels are used.

Here the domain is approximated with a 1600-by-1600 km box in Cartesian
coordinates with 16 km resolution.

Initial temperature is defined by gradients dT/dy=-5e-6 degC/m and
dT/dz=8.2e-3 degC/m. Maximum surface temperature is 25 degC.
Salinity is constant 35 psu.

Vertical viscosity and diffusivity are set to 1e-3 and 1e-5 m2/s, respectively.
Pacanowski-Philander vertical mixing scheme is used for temperature with max.
diffusivity 0.01 m2/s. Horizontal diffusivity 30 m2/s is used. Horizontal
viscosity depends on the advection scheme (typ. 100 m2/s). Bottom drag
coefficient is Cd=0.0025.

[1] Danilov, S. (2012). Two finite-volume unstructured mesh models for
large-scale ocean modeling. Ocean Modelling, 47:14-25.
"""

from thetis import *

physical_constants['rho0'] = 1020.

refinement = 1  # normal = 4
lx = ly = 1600e3
delta_x = delta_y = 64e3/refinement
nx = int(np.ceil(lx/delta_x))
ny = int(np.ceil(ly/delta_y))
mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x')
depth = 1600.
nlayers = 6

t_end = 365*24*3600.
t_export = 1*24*3600.

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

coriolis_f, coriolis_beta = beta_plane_coriolis_params(37.5)
xy = SpatialCoordinate(mesh2d)
coriolis_expr = coriolis_f + coriolis_beta*(xy[1] - ly/2)
coriolis_2d = Function(P1_2d, name='coriolis').interpolate(coriolis_expr)

# temperature and salinity, results in 2.0 kg/m3 density difference
salt_const = 35.0
temp_max = 25.0
temp_ddy = -5e-6
temp_ddz = 8.2e-3

u_scale = 1.0
w_scale = 1e-3

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'ssprk22'
options.solve_salt = False
options.constant_salt = Constant(salt_const)
options.solve_temp = True
options.solve_vert_diffusion = False
options.use_bottom_friction = False
#options.solve_vert_diffusion = True
#options.use_bottom_friction = True
#options.use_turbulence = True
#options.turbulence_model = 'pacanowski'
options.use_ale_moving_mesh = True
options.baroclinic = True
options.coriolis = coriolis_2d
options.uv_lax_friedrichs = Constant(1.0)
options.tracer_lax_friedrichs = Constant(1.0)
options.use_limiter_for_tracers = True
options.quadratic_drag = Constant(0.0025)
options.h_viscosity = Constant(100.)
options.h_diffusivity = Constant(30.)
options.v_viscosity = Constant(1.0e-3)
options.v_diffusivity = Constant(1.0e-5)
options.t_export = t_export
options.t_end = t_end
#options.dt = 3600.
options.u_advection = Constant(u_scale)
options.w_advection = Constant(w_scale)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_temp_conservation = True
options.check_temp_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'temp_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d']

solver_obj.create_function_spaces()
solver_obj.create_fields()

xyz = SpatialCoordinate(solver_obj.mesh)
temp_init_3d = temp_max + xyz[2]*temp_ddz + xyz[1]*temp_ddy
temp_pertubation = 0.2*sin(6*pi*xyz[0]/lx)*exp(-(xyz[1]-ly/2)**2/(ly/4)**2)

# add a relaxation term for temperature:
y_arr = Function(solver_obj.function_spaces.H).interpolate(xyz[1]).dat.data[:]
temp_relax = temp_init_3d
t_temp_relax = Constant(3.*24.*3600.)  # time scale
mask_temp_relax_3d = Function(solver_obj.function_spaces.H, name='mask_temp_relax_3d')
ly_relax = 160e3  # approx 1.5 deg
mask_numpy_y0 = (ly_relax - y_arr)/ly_relax
mask_numpy_y1 = (1 + (y_arr - ly)/ly_relax)
mask_temp_relax_3d.dat.data[:] = np.maximum(mask_numpy_y0, mask_numpy_y1)
ix = mask_temp_relax_3d.dat.data < 0
mask_temp_relax_3d.dat.data[ix] = 0.0
# File('mask.pvd').write(mask_temp_relax_3d)
options.temp_source_3d = mask_temp_relax_3d/t_temp_relax*(temp_relax - solver_obj.fields.temp_3d)

solver_obj.create_equations()
solver_obj.assign_initial_conditions(temp=temp_init_3d)

# initialize flow in geostrophic balance
uv_init_expr = as_vector((-solver_obj.fields.int_pg_3d[1]/solver_obj.fields.coriolis_3d, 0, 0))
solver_obj.assign_initial_conditions(temp=temp_init_3d + temp_pertubation, uv_3d=uv_init_expr)

solver_obj.iterate()
