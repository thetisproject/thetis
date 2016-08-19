"""
Baroclinic channel test case following [1] and [2]

Domain is 160 by 500 km long channel, periodic in the x direction.
Depth is 1000 m.

Vertical temperature in the northern section is

T(z) = T_b + (T_b - T_s)*(z_b - z)/z_b

where T_b = 10.1, T_s = 13.1 degC are the bottom and surface temperatures, and z_b = -975 m is the bottom z coordinate.

Density is computed using a linear equation of state that does not depend on salinity:
rho = rho_0 - alpha*(T - T_ref)
with rho_0=1000 kg/m3, alpha=0.2 kg/m3/degC, T_ref=5.0 degC.

Horizontal mesh resolution is 10, 4, or 1 km. 20 vertical levels are used.

Coriolis parameter is 1.2e-4 1/s.
Bottom drag coefficient is 0.01.

Horizontal viscosity varies between 1.0 and 200 m2/s. Vertical viscosity is
set to constant 1e-4 m2/s. Tracer diffusion is set to zero.

[1] Ilicak et al. (2012). Spurious dianeutral mixing and the role
    of momentum closure. Ocean Modelling, 45-46(0):37-58.
[2] Petersen et al. (2015). Evaluation of the arbitrary Lagrangian-Eulerian
    vertical coordinate method in the MPAS-Ocean model. Ocean Modelling, 86:93-113.
"""

from thetis import *
comm = COMM_WORLD

delta_x = 10*1.e3
lx = 160e3
ly = 500e3
nx = int(lx/delta_x)
ny = int(ly/delta_x)
delta_x = lx/nx
mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x')
depth = 1000.
nlayers = 20

# compute horizontal viscosity
reynolds_number = 20
uscale = 0.1
nu_scale = uscale * delta_x / reynolds_number

f_cori = -1.2e-4
bottom_drag = 0.01
t_end = 200*24*3600.  # 365*24*3600.
t_export = 3*3600.

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# temperature and salinity, results in 2.0 kg/m3 density difference
salt_const = 35.0
temp_bot = 10.1
temp_surf = 13.1
rho_0 = 1000.0
physical_constants['rho0'].assign(rho_0)

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'leapfrog'
options.solve_salt = False
options.constant_salt = Constant(salt_const)
options.solve_temp = True
options.solve_vert_diffusion = True
options.use_bottom_friction = True
options.quadratic_drag = Constant(bottom_drag)
options.baroclinic = True
options.coriolis = Constant(f_cori)
options.uv_lax_friedrichs = None
options.tracer_lax_friedrichs = None
# options.smagorinsky_factor = Constant(1.0/np.sqrt(Re_h))
options.use_limiter_for_tracers = True
options.v_viscosity = Constant(1.0e-4)
options.h_viscosity = Constant(nu_scale)
options.nu_viscosity = Constant(nu_scale)
options.h_diffusivity = None
options.t_export = t_export
options.t_end = t_end
options.dt = 400.
options.u_advection = Constant(0.5)
options.w_advection = Constant(1e-2)
options.check_vol_conservation_2d = True
options.check_vol_conservation_3d = True
options.check_temp_conservation = True
options.check_temp_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'temp_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d',
                            'smag_visc_3d', 'salt_jump_diff']
options.equation_of_state = 'linear'
options.lin_equation_of_state_params = {
    'rho_ref': rho_0,
    's_ref': salt_const,
    'th_ref': 5.0,
    'alpha': 0.2,
    'beta': 0.0,
}

solver_obj.create_equations()

print_output('Running eddy test case with options:')
print_output('Number of cores: {:}'.format(comm.size))
print_output('Mesh resolution dx={:} nlayers={:}'.format(delta_x, nlayers))
print_output('Reynolds number: {:}'.format(reynolds_number))
print_output('Horizontal viscosity: {:}'.format(nu_scale))

xyz = SpatialCoordinate(solver_obj.mesh)
# vertical background stratification
temp_vert = temp_bot + (temp_surf - temp_bot)*(-depth - xyz[2])/-depth
# sinusoidal temperature anomaly
temp_delta = -1.2
y0 = 250.e3
ya = 40.e3
k = 3
yd = 40.e3
yw = y0 - ya*sin(2*pi*k*xyz[0]/lx)
fy = (1. - (xyz[1] - yw)/yd)
s_lo = 0.5*(sign(fy) + 1.)
s_hi = 0.5*(sign(1. - fy) + 1.)
temp_wave = temp_delta*(fy*s_lo*s_hi + (1.0-s_hi))
# perturbation of one crest
temp_delta2 = -0.3
x2 = 110.e3
x3 = 130.e3
yw2 = y0 - ya/2*sin(pi*(xyz[0] - x2)/(x3 - x2))
fy = (1. - (xyz[1] - yw2)/(yd/2))
s_lo = 0.5*(sign(fy) + 1.)
s_hi = 0.5*(sign(2. - fy) + 1.)
temp_wave2 = temp_delta2*(fy*s_lo*s_hi + (1.0-s_hi))
s_wave2 = 0.5*(sign(xyz[0] - x2)*(-1)*sign(xyz[0] - x3) + 1.)*s_hi
temp_expr = temp_vert + s_wave2*temp_wave2 + (1.0 - s_wave2)*temp_wave
temp_init3d = Function(solver_obj.function_spaces.H)
temp_init3d.interpolate(temp_expr)
solver_obj.assign_initial_conditions(temp=temp_init3d)

# compute 2D baroclinic head and use it to initialize elevation field
# to remove fast 2D gravity wave caused by the initial density difference
compute_baroclinic_head(solver_obj)
elev_init = Function(solver_obj.function_spaces.H_2d)
elev_init.assign(-solver_obj.fields.baroc_head_2d*depth)
mean_elev = assemble(elev_init*dx)/lx/ly
elev_init += -mean_elev
solver_obj.assign_initial_conditions(temp=temp_init3d, elev=elev_init)

solver_obj.iterate()
