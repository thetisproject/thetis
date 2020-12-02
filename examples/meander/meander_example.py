from thetis import *

import pandas as pd
import numpy as np
import pylab as plt

import time
import datetime

def update_forcings_bnd(t_new):

    gradient_flux = (-0.053 + 0.02)/6000
    gradient_flux2 = (-0.02+0.053)/(18000-6000)
    gradient_elev = (10.04414- 9.9955)/6000
    gradient_elev2 = (9.9955-10.04414)/(18000-6000)
    elev_init_const = (-max(initial_bathymetry_2d.dat.data[:]) + 0.05436)

    if t_new != t_old.dat.data[:]:
        # update boundary condtions
        if t_new*morfac <= 6000:
            elev_constant.assign(gradient_elev*t_new*morfac + elev_init_const)
            flux_constant.assign((gradient_flux*t_new*morfac) - 0.02)
        else:
            flux_constant.assign((gradient_flux2*(t_new*morfac-6000)) - 0.053)
            elev_constant.assign(gradient_elev2*(t_new*morfac-18000) + elev_init_const)
        t_old.assign(t_new)

t_old = Constant(0.0)

# define mesh
mesh2d = Mesh("meander.msh")
x,y = SpatialCoordinate(mesh2d)

# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)
DG_2d = FunctionSpace(mesh2d, 'DG', 1)
vector_dg = VectorFunctionSpace(mesh2d, 'DG', 1)

# define underlying bathymetry

bathymetry_2d = Function(V, name='Bathymetry')

gradient = Constant(0.0035)

L_function= Function(V).interpolate(conditional(x > 5, pi*4*((pi/2)-acos((x-5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi, pi*4*((pi/2)-acos((-x+5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi))
bathymetry_2d1 = Function(V).interpolate(conditional(y > 2.5, conditional(x < 5, (L_function*gradient) + 9.97072, -(L_function*gradient) + 9.97072), 9.97072))

init = max(bathymetry_2d1.dat.data[:])
final = min(bathymetry_2d1.dat.data[:])

bathymetry_2d2 = Function(V).interpolate(conditional(x <= 5, conditional(y<=2.5, -9.97072 + gradient*abs(y - 2.5) + init, 0), conditional(y<=2.5, -9.97072 -gradient*abs(y - 2.5) + final, 0)))
bathymetry_2d = Function(V).interpolate(-bathymetry_2d1 - bathymetry_2d2)

initial_bathymetry_2d = Function(V).interpolate(bathymetry_2d)
diff_bathy = Function(V).interpolate(Constant(0.0))

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs'+ st

diff_bathy_file = File(outputdir + "/diff_bathy.pvd")
diff_bathy_file.write(diff_bathy)

# initialise velocity and elevation
chk = DumbCheckpoint("hydrodynamics_meander/elevation", mode=FILE_READ)
elev = Function(DG_2d, name="elevation")
chk.load(elev)
chk.close()

chk = DumbCheckpoint('hydrodynamics_meander/velocity', mode=FILE_READ)
uv = Function(vector_dg, name="velocity")
chk.load(uv)
chk.close()

morfac = 50
dt = 2
end_time = 18000

diffusivity = 0.15
viscosity_hydro = Constant(5*10**(-2))

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options

options.sediment_model_options.solve_suspended_sediment = False
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True
options.sediment_model_options.use_angle_correction = True
options.sediment_model_options.use_slope_mag_correction = True
options.sediment_model_options.use_secondary_current = True
options.sediment_model_options.use_advective_velocity_correction = False
options.sediment_model_options.morphological_viscosity = Constant(1e-6)

options.sediment_model_options.average_sediment_size = Constant(10**(-3))
options.sediment_model_options.bed_reference_height = Constant(0.003)
options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

options.simulation_end_time = end_time/morfac
options.simulation_export_time = options.simulation_end_time/90

options.output_directory = outputdir
options.check_volume_conservation_2d = True

options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']

# using nikuradse friction
options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)

# set horizontal diffusivity parameter
options.horizontal_diffusivity = Constant(diffusivity)
options.horizontal_viscosity = Constant(viscosity_hydro)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

left_bnd_id = 1
right_bnd_id = 2

# set boundary conditions
gradient_flux = (-0.053 + 0.02)/6000
gradient_flux2 = (-0.02+0.053)/(18000-6000)
gradient_elev = (10.04414- 9.9955)/6000
gradient_elev2 = (9.9955-10.04414)/(18000-6000)
elev_init_const = (-max(bathymetry_2d.dat.data[:]) + 0.05436)
swe_bnd = {}
swe_bnd[3] = {'un': Constant(0.0)}

flux_constant = Constant(-0.02)
elev_constant = Constant(elev_init_const)

swe_bnd[left_bnd_id] = {'flux': flux_constant}
swe_bnd[right_bnd_id] = {'elev': elev_constant}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

solver_obj.assign_initial_conditions(uv=uv, elev=elev)

# run model
solver_obj.iterate(update_forcings = update_forcings_bnd)
