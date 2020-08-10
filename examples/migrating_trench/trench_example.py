"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench.

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *
from sediment_callback import SedimentTotalMassConservation2DCallback

import numpy as np
import pandas as pd
import time

conservative = False

# Note it is necessary to run trench_hydro first to get the hydrodynamics simulation

# define mesh
lx = 16
ly = 1.1
nx = lx*5
ny = 5
mesh2d = RectangleMesh(nx, ny, lx, ly)

x, y = SpatialCoordinate(mesh2d)

# define function spaces
V = get_functionspace(mesh2d, "CG", 1)
P1_2d = get_functionspace(mesh2d, "DG", 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='bathymetry_2d')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                     conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv, depth_riv))))
bathymetry_2d.interpolate(-trench)

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

print_output('Exporting to '+outputdir)

# define bathymetry_file
bathy_file = File(outputdir + "/bathy.pvd")

morfac = 100
dt = 0.3
end_time = 15*3600

diffusivity = 0.15
viscosity_hydro = Constant(1e-6)

# initialise velocity and elevation
chk = DumbCheckpoint("hydrodynamics_trench/elevation", mode=FILE_READ)
elev = Function(DG_2d, name="elevation")
chk.load(elev)
chk.close()

chk = DumbCheckpoint('hydrodynamics_trench/velocity', mode=FILE_READ)
uv = Function(vector_dg, name="velocity")
chk.load(uv)
chk.close()

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options

options.sediment_model_options.solve_suspended_sediment = True
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True

options.sediment_model_options.use_sediment_conservative_form = conservative
options.sediment_model_options.average_sediment_size = 160*(10**(-6))
options.sediment_model_options.bed_reference_height = 0.025
options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

options.simulation_end_time = end_time/morfac
options.simulation_export_time = options.simulation_end_time/45

options.output_directory = outputdir
options.check_volume_conservation_2d = True

if options.sediment_model_options.solve_suspended_sediment:
    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d', 'bathymetry_2d']
    options.sediment_model_options.check_sediment_conservation = True
else:
    options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']

# using nikuradse friction
options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)

# set horizontal diffusivity parameter
options.horizontal_diffusivity = Constant(diffusivity)
options.horizontal_viscosity = Constant(viscosity_hydro)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.norm_smoother = Constant(0.1)

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

c = SedimentTotalMassConservation2DCallback('sediment_2d',
                                            solver_obj, export_to_hdf5=True, append_to_log=False)
solver_obj.add_callback(c, eval_interval='timestep')

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

if options.sediment_model_options.solve_suspended_sediment:
    # setting an equilibrium boundary conditions results in the sediment value at the boundary
    # being chosen so that erosion and deposition are equal here (ie. in equilibrium) and the bed is immobile at this boundary
    solver_obj.bnd_functions['sediment'] = {
        left_bnd_id: {'flux': Constant(-0.22), 'equilibrium': None},
        right_bnd_id: {'elev': Constant(0.397)}}

    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)

else:
    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)

# run model
solver_obj.iterate()

# record final sediment and final bathymetry
xaxisthetis1 = []
sedimentthetis1 = []
baththetis1 = []

for i in np.linspace(0, 15.8, 80):
    xaxisthetis1.append(i)
    if conservative:
        d = solver_obj.fields.bathymetry_2d.at([i, 0.55]) + solver_obj.fields.elev_2d.at([i, 0.55])
        sedimentthetis1.append(solver_obj.fields.sediment_2d.at([i, 0.55])/d)
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))
    else:
        sedimentthetis1.append(solver_obj.fields.sediment_2d.at([i, 0.55]))
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))

# check sediment conservation
sediment_mass_int, sediment_mass_int_rerr = solver_obj.callbacks['timestep']['sediment_2d total mass']()
print("Sediment total mass error: %11.4e" % (sediment_mass_int_rerr))

# check sediment and bathymetry values using previous runs
sediment_solution = pd.read_csv('sediment.csv')
bed_solution = pd.read_csv('bed.csv')


assert max([abs((sediment_solution['Sediment'][i] - sedimentthetis1[i])/sediment_solution['Sediment'][i]) for i in range(len(sedimentthetis1))]) < 0.15, "error in sediment"

assert max([abs((bed_solution['Bathymetry'][i] - baththetis1[i])) for i in range(len(baththetis1))]) < 0.01, "error in bed level"
