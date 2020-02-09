# Landslide 2d validation case --- Storegga
# ===================
# Wei Pan 2018-April-09

from thetis import *
import math
print(1.992*sin(14.5*pi/180), 'sssssssssssssssssssss')
lx = 8.E3
ly = 8.E3/2.
dx = 25.
dy = 25.
nx = lx / dx
ny = ly / dy
mesh2d = RectangleMesh(nx, ny, lx, ly)
n_layers = 1
outputdir = 'outputs_landslide_nh'
print_output('Exporting to ' + outputdir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
depth_max = 400.
depth_min = 12.
slope = 4.
dip = slope*pi/180.
xy = SpatialCoordinate(mesh2d)

class MyExpression(Expression):
    def eval(self, value, xy):
         if xy[0] > 7000.:
              value[:] = depth_max
         else:
              value[:] = depth_max + (xy[0]-7000.)*tan(dip)
              if value[:] < depth_min:
                   value[:] = depth_min

bathymetry_2d.interpolate(MyExpression())

# set time step, export interval and run duration
dt = 2.
t_export = 10.
t_end = 100.

# --- create solver ---
solver_obj = solver_nh.FlowSolver(mesh2d, bathymetry_2d, n_layers)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.timestepper_type = 'SSPRK22'#'LeapFrog'#'SSPRK22'
options.use_nonlinear_equations = True
# for three-layer NH model, suggest to set alpha as 0.1, beta 0.45
# for coupled two-layer NH model, suggest to set alpha as 0.2
# for reduced model, alpha and beta depend on specific cases,
# as recommended by Cui et al. (2014), alpha = 0.15 and beta = 1.0
options.alpha_nh = Constant(0.5)
options.beta_nh = Constant(0.45)
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False#True
options.timestepper_options.use_automatic_timestep = False
options.output_directory = outputdir
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_ls']
# logical parameter for landslide solver
options.landslide = False#True
options.slide_is_rigid = True
options.t_landslide = 6732
options.rho_slide = 3000.
#options.volume_source_2d = Function(P1_2d)
# for horizontal 1D case
options.vertical_2d = False
# bottom drag
#options.manning_drag_coefficient = Constant(0.08)
# set sponge layer to absorb relected waves
#options.sponge_layer_length = [100.E3, 0.] # default is None, i.e. no absorption
#options.sponge_layer_xstart = [0., 0.] # default is 0.
##### --- wetting and drying --- #####
options.constant_mindep = False
# if True, the thin-film depth at wetting-drying interface is not varied and equals to wd_mindep
# if False, the thin-film depth at wetting-drying interface at each step is determine by wd_mindep,
# which here refers to the thin-film depth at the lowest depth, i.e. highest bathymetry point
#
### note: if options.thin_film is True, thin-film wd scheme will be used ###
options.thin_film = False
options.wd_mindep = 0.01#[0.08179199]

# need to call creator to create the function spaces
solver_obj.create_equations()

# set slide shape function to track position
def slide_shape(t):
    b = 686.
    w = 343.
    hmax = 24.
    phi = 0.
    #x0 = 2137.773473 + (w/2.)
    d = 60. # initial depth
    x0 = 7000. - (depth_max - (hmax*cos(dip) + d))/tan(dip)
    y0 = 0#ly/2.

    eps = 0.717
    C = math.acosh(1/eps)
    kw = 2*C/w
    kb = 2*C/b
    a0 = 0.27
    ut = 21.09
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
        x = (xy[0] - xs)*cos(phi)/cos(dip) + (xy[1] - ys)*sin(phi)
        y = -(xy[0] - xs)*sin(phi)/cos(dip) + (xy[1] - ys)*cos(phi)
        hs_vector[i] = hmax/(1-eps) * ((1/math.cosh(kb*x))*(1/math.cosh(kw*y)) - eps)
        hs_vector[i] = hs_vector[i]/cos(dip)
        if (1/math.cosh(kb*x))*(1/math.cosh(kw*y)) <= eps:
            hs_vector[i] = 0.
    return hs

# update slide position and associated source term
def update_forcings(t):
    #print_output("Updating bottom boundary and landslide source at t={}".format(t))
    landslide_source = solver_obj.fields.get('slide_source')
    hs = (slide_shape(t + options.timestep) - slide_shape(t))/options.timestep
    landslide_source.project(hs)
    solver_obj.bathymetry_dg.project(solver_obj.fields.bathymetry_2d - slide_shape(t))
    solver_obj.fields.solution_ls.sub(1).assign(-solver_obj.bathymetry_dg)

solver_obj.iterate(update_forcings=update_forcings)

