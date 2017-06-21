# Overflow Test case
# =======================
#
# Overflow test case from Haidvogel and Beckmann (1999)
#
# 200 km long channel with sloping bathymetry from 200 m to 4 km depth.
# Initially dense water is located on top of the slope.
#
# Horizontal resolution: 1 km
# Vertical layers: 40, 66, or 100 (Ilicak, 2012)
# Baroclinic/barotropic time steps: 10.0 s / 1.0 s
#
# Dianeutral mixing depends on mesh Reynolds number (Ilicak et al. 2012)
# Re_h = U dx / nu
# U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(g_h drho/rho_0)
# dx = horizontal mesh size
# nu = background viscosity
#
# For coarse mesh:
# Re_h = 0.5 2000 / 100 = 10
#
# TODO run medium for Re_h = 250
# => nu = 0.5 500 / 250 = 1.0
#
# Smagorinsky factor should be C_s = 1/sqrt(Re_h)
#
# Tuomas Karna 2015-06-10

from thetis import *

physical_constants['rho0'] = 999.7

reso_str = 'medium'
refinement = {'medium': 1}
layers = int(round(50*refinement[reso_str]))/2
mesh2d = Mesh('mesh_{0:s}.msh'.format(reso_str))
print_output('Loaded mesh '+mesh2d.name)
dt = 5.0/refinement[reso_str]
t_end = 25 * 3600
t_export = 15*60.0
depth = 20.0
Re_h = 10.0
outputdir = 'outputs_' + reso_str + '_Re' + str(int(Re_h))

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('hmin + 0.5*(hmax - hmin)*(1 + tanh((x[0] - x0)/Ls))',
                          hmin=500.0, hmax=2000.0, Ls=10.0e3, x0=40.0e3))

# temperature and salinity, results in 2.0 kg/m3 density difference
salt_left = 2.5489
salt_right = 0.0
temp_const = 10.0

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'leapfrog'
# options.timestepper_type = 'ssprk33'
options.solve_salinity = True
options.solve_temperature = False
options.constant_temperature = Constant(temp_const)
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.baroclinic = True
options.use_lax_friedrichs_velocity = True
options.use_lax_friedrichs_tracer = True
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(Re_h))
options.use_limiter_for_tracers = True
options.vertical_viscosity = Constant(1.0e-4)
options.horizontal_viscosity = None
options.horizontal_diffusivity = None
options.t_export = t_export
options.t_end = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(6.0)
options.vertical_velocity_scale = Constant(3.0)
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.check_salinity_conservation = True
options.check_salinity_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'density_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'smag_visc_3d']

solver_obj.create_equations()
salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression('(x[0] > 20.0e3) ? s_r : s_l',
#                                    s_l=salt_left, s_r=salt_right))
# smooth condition
salt_init3d.interpolate(Expression('s_l + (s_r - s_l)*0.5*(1.0 + tanh((x[0] - x0)/sigma))',
                                   s_l=salt_left, s_r=salt_right, x0=20.0e3, sigma=1000.0))

solver_obj.assign_initial_conditions(salt=salt_init3d)
solver_obj.iterate()
