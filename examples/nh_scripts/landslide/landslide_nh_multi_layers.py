# 1st test case of Thetis-NH solver
# ===================
#
# Solves a standing wave in a rectangular basin using 3D Non-hydrostatic equations
#
# Initial condition for elevation corresponds to a standing wave.
#
# This example tests dispersion of surface waves improved by non-hydrostatic pressure.
#
# Wei Pan 08/Jan/2018
from thetis import *
import math

lx = 6.
ly = 1.8
nx = 150 # i.e. dx = 0.05
ny = 36 # i.e. dy = 0.05
mesh2d = RectangleMesh(nx, ny, lx, ly)
n_layers = 3
outputdir = 'outputs_landslide_nh_small_scale_multi_layer'
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
dt = 0.01/10.
t_export = 0.01
t_end = 4.

# --- create solver ---
solver_obj = solver_ml.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 2
options.timestepper_type = 'CrankNicolson'
options.use_nonlinear_equations = True
# for three-layer NH model, suggest to set alpha as 0.1, beta 0.45
# for coupled two-layer NH model, suggest to set alpha as 0.2
# for reduced model, alpha and beta depend on specific cases,
# as recommended by Cui et al. (2014), alpha = 0.15 and beta = 1.0
# for multi-layer case,
# layer thickness accounting for total height defined by alpha_nh list
#options.use_nh_solver = True
options.n_layers = n_layers
options.alpha_nh = [] # [] means uniform layers
options.output_directory = outputdir
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.fields_to_export = ['uv_2d', 'elev_2d']
# logical parameter for landslide solver
options.landslide = True
options.slide_is_rigid = True
options.t_landslide = 6732
options.rho_slide = 3000.
# for horizontal 1D case
options.set_vertical_2d = False
# bottom drag
#options.manning_drag_coefficient = Constant(0.2)
#options.linear_drag_coefficient = Constant(10.)
# set sponge layer to absorb relected waves
#options.sponge_layer_length = [100.E3, 0.] # default is None, i.e. no absorption
#options.sponge_layer_xstart = [0., 0.] # default is 0.
##### --- wetting and drying --- #####
options.use_wetting_and_drying = True
options.constant_mindep = True
# if True, the thin-film depth at wetting-drying interface is not varied and equals to wd_mindep
# if False, the thin-film depth at wetting-drying interface at each step is determine by wd_mindep,
# which here refers to the thin-film depth at the lowest depth, i.e. highest bathymetry point
#
### note: if options.thin_film is True, thin-film wd scheme will be used ###
options.thin_film = False
options.wd_mindep = 0.01

# need to call creator to create the function spaces
solver_obj.create_equations()

# set slide shape function to track position
def slide_shape(t):
    b = 0.395#686.
    w = 0.680#343.
    hmax = 0.082#24.
    phi = 0.
    x0 = 0.651 # used by nhwave
    d = 0.061 # initial depth of slide
   # x0 = (hmax*cos(dip) + d)/tan(dip)+0.1 # 0.62326
    y0 = 0#ly/2.

    eps = 0.717
    C = math.acosh(1./eps)
    kw = 2.*C/w
    kb = 2.*C/b
    a0 = 1.12#0.27
    ut = 1.70#21.09
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
        # original one for Storegga
        x = (xy[0] - xs)*cos(phi)/cos(dip) + (xy[1] - ys)*sin(phi)
        y = -(xy[0] - xs)*sin(phi)/cos(dip) + (xy[1] - ys)*cos(phi)
        hs_vector[i] = hmax/(1 - eps) * ((1/math.cosh(kb*x))*(1/math.cosh(kw*y)) - eps)
        hs_vector[i] = hs_vector[i]/cos(dip)
        if (1/math.cosh(kb*x))*(1/math.cosh(kw*y)) <= eps:
            hs_vector[i] = 0.

        # same as nhwave below
        x = (xy[0] - xs)*cos(phi) + (xy[1] - ys)*sin(phi)
        y = -(xy[0] - xs)*sin(phi) + (xy[1] - ys)*cos(phi)
        hs_vector[i] = hmax/(1 - eps) * ((1/math.cosh(kb*x))*(1/math.cosh(kw*y)) - eps)
        if (1/math.cosh(kb*x))*(1/math.cosh(kw*y)) <= eps:
            hs_vector[i] = 0.

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
    #print_output("Updating bottom boundary and landslide source at t={}".format(t))
    landslide_source = solver_obj.fields.get('slide_source')
    hs_1 = slide_shape(t)
    hs_2 = slide_shape(t + options.timestep)
    hs = (hs_2 - hs_1)/(options.timestep)
    landslide_source.project(hs)
    solver_obj.bathymetry_dg.project(solver_obj.fields.bathymetry_2d - hs_2)
    solver_obj.fields.solution_ls.sub(1).assign(-solver_obj.bathymetry_dg)

solver_obj.assign_initial_conditions(uv=Constant((1E-13, 0.)))
solver_obj.iterate(update_forcings=update_forcings)

