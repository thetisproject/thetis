# surface wave generation by a three-dimensional submarine rigid landslide
# ===================
# Wei Pan 08/Jan/2018

from thetis import *
import math

lx = 6.
ly = 1.8
nx = 150
ny = 36
mesh2d = RectangleMesh(nx, ny, lx, ly)
n_layers = 3
outputdir = 'outputs_landslide_rigid_sigma'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
slope = 15.
dip = slope*pi/180.
xy = SpatialCoordinate(mesh2d)
class BathyExpression(Expression):
    def eval(self, value, xy):
         value[:] = (xy[0]-0.1)*tan(dip)
         if xy[0] <= 0.2:
             value[:] = 0.1*tan(dip)
bathymetry_2d.interpolate(BathyExpression())

# set time step, export interval and run duration
dt = 0.005
t_export = 0.01
t_end = 4.

# --- create solver ---
solver_obj = nhsolver_sigma.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'
options.timestepper_options.use_automatic_timestep = False
# free surface elevation
options.update_free_surface = True
options.solve_separate_elevation_gradient = True
# tracer
options.solve_salinity = False
options.solve_temperature = False
# limiter
options.use_limiter_for_velocity = False
options.use_limiter_for_tracers = False
# mesh update
options.use_ale_moving_mesh = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
# time
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.no_exports = False
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']
# landslide
options.landslide = True
options.slide_is_rigid = True
# wetting and drying
options.use_wetting_and_drying = True
options.depth_wd_interface = 0.01

# need to call creator to create the function spaces
solver_obj.create_equations()

# set slide shape function to track position
def slide_shape(t):
    b = 0.395
    w = 0.680
    hmax = 0.082
    phi = 0.
    x0 = 0.651 # used by nhwave
    d = 0.061 # initial depth of slide
    y0 = 0#ly/2.

    eps = 0.717
    C = math.acosh(1./eps)
    kw = 2.*C/w
    kb = 2.*C/b
    a0 = 1.12
    ut = 1.70
    t0 = ut/a0
    s0 = (ut*ut)/a0

    s = s0*math.log(math.cosh(t/t0))*cos(dip)

    xs = x0 + s*cos(phi)
    ys = y0 + s*sin(phi)

    # calculate slide shape below
    hs = Function(P1_2d)
    mesh2d = hs.ufl_domain()
    xy_vector = mesh2d.coordinates.dat.data
    hs_vector = hs.dat.data
    assert xy_vector.shape[0] == hs_vector.shape[0]
    for i, xy in enumerate(xy_vector):
        xl = xs - 0.5*b*cos(dip)
        xr = xs + 0.5*b*cos(dip)
        yl = ys - 0.5*w
        yr = ys + 0.5*w
        hs_vector[i] = 0.
        if xy[0] >= xl and xy[0] <= xr and xy[1] >= yl and xy[1] <= yr:
            x = (xy[0] - xs)*cos(phi)/cos(dip) + (xy[1] - ys)*sin(phi)
            y = -(xy[0] - xs)*sin(phi)/cos(dip) + (xy[1] - ys)*cos(phi)
            hs_vector[i] = hmax/(1 - eps) * ((1/math.cosh(kb*x))*(1/math.cosh(kw*y)) - eps)
            if hs_vector[i] <= 0.:
                hs_vector[i] = 0.
    return hs

# update slide position and associated source term
def update_forcings(t):
    solver_obj.fields.h_ls.project(slide_shape(t))

solver_obj.assign_initial_conditions(h_ls=slide_shape(0))
solver_obj.iterate(update_forcings=update_forcings)

