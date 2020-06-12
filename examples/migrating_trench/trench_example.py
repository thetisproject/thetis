"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench.

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *
import callback_cons_tracer as call
from thetis.sediments import SedimentModel

import numpy as np
import pandas as pd
import time

conservative = False

def initialise_fields(mesh2d, inputdir, outputdir,):
    """
    Initialise simulation with results from a previous simulation
    """
    DG_2d = get_functionspace(mesh2d, "DG", 1)
    # elevation
    with timed_stage('initialising elevation'):
        chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_READ)
        elev_init = Function(DG_2d, name="elevation")
        chk.load(elev_init)
        File(outputdir + "/elevation_imported.pvd").write(elev_init)
        chk.close()
    # velocity
    with timed_stage('initialising velocity'):
        chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_READ)
        V = VectorFunctionSpace(mesh2d, "DG", 1)
        uv_init = Function(V, name="velocity")
        chk.load(uv_init)
        File(outputdir + "/velocity_imported.pvd").write(uv_init)
        chk.close()
        return elev_init, uv_init,  

## Note it is necessary to run trench_hydro first to get the hydrodynamics simulation

# exporting bathymetry
def export_bath_func():
    bathy_file.write(sed_mod.bathymetry_2d)

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
bathymetry_2d = Function(V, name = 'bathymetry_2d')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                                                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv,
                                                                                                                          depth_riv))))
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

# initialise velocity, elevation and depth
elev, uv = initialise_fields(mesh2d, 'hydrodynamics_trench', outputdir)

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options

sed_mod = SedimentModel(options, suspendedload=True, convectivevel=True,
                        bedload=True, angle_correction=True, slope_eff=True, seccurrent=False,
                        mesh2d=mesh2d, bathymetry_2d=solver_obj.fields.bathymetry_2d, 
                        uv_init = uv, elev_init = elev, ks=0.025, average_size=160 * (10**(-6)), 
                        cons_tracer = conservative, wetting_and_drying = False, wetting_alpha = 0.1)

solver_obj.sediment_model = sed_mod

options.update(sed_mod.options)

options.simulation_end_time = end_time/morfac
options.simulation_export_time = options.simulation_end_time/45

options.output_directory = outputdir
options.check_volume_conservation_2d = True

if sed_mod.suspendedload:
    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d'] #note exporting bathymetry must be done through export func
    options.tracer_source_2d = sed_mod.ero_term
    options.tracer_sink_2d = sed_mod.depo_term
    #options.tracer_depth_integ_source = sed_mod.ero
    #options.tracer_depth_integ_sink = sed_mod.depo_term
    options.check_tracer_conservation = False
else:
    options.fields_to_export = ['uv_2d', 'elev_2d'] #note exporting bathymetry must be done through export func   

options.solve_exner = True
options.morphological_acceleration_factor = Constant(morfac)  
# using nikuradse friction
options.nikuradse_bed_roughness = sed_mod.ksp

# set horizontal diffusivity parameter
options.horizontal_diffusivity = Constant(diffusivity)
options.horizontal_viscosity = Constant(viscosity_hydro)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.norm_smoother = Constant(sed_mod.wetting_alpha)

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = dt

#c = call.TracerTotalMassConservation2DCallback('tracer_2d',
#                                               solver_obj, export_to_hdf5=True, append_to_log=False)
#solver_obj.add_callback(c, eval_interval='timestep') #FIXME

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}    

solver_obj.bnd_functions['shallow_water'] = swe_bnd

if sed_mod.suspendedload:
    solver_obj.bnd_functions['sediment'] = {left_bnd_id: {'value': sed_mod.sediment_rate, 'flux': Constant(-0.22)}, right_bnd_id: {'elev': Constant(0.397)} }

    # set initial conditions
    solver_obj.assign_initial_conditions(uv=sed_mod.uv_init, elev=sed_mod.elev_init, sediment=sed_mod.testtracer)

else:
    # set initial conditions
    solver_obj.assign_initial_conditions(uv=sed_mod.uv_init, elev=sed_mod.elev_init)

# run model
solver_obj.iterate(export_func = export_bath_func)

# record final tracer and final bathymetry
xaxisthetis1 = []
tracerthetis1 = []
baththetis1 = []

for i in np.linspace(0, 15.8, 80):
    xaxisthetis1.append(i)
    if conservative:
        d = solver_obj.fields.bathymetry_2d.at([i, 0.55]) + solver_obj.fields.elev_2d.at([i, 0.55])
        tracerthetis1.append(solver_obj.fields.tracer_2d.at([i, 0.55])/d)
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))
    else:
        tracerthetis1.append(solver_obj.fields.tracer_2d.at([i, 0.55]))
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))

    # check tracer conservation
tracer_mass_int, tracer_mass_int_rerr = solver_obj.callbacks['timestep']['tracer_2d total mass']()
print("Tracer total mass error: %11.4e" % (tracer_mass_int_rerr))

# check tracer and bathymetry values using previous runs
tracer_solution = pd.read_csv('tracer.csv')
bed_solution = pd.read_csv('bed.csv')


assert max([abs((tracer_solution['Tracer'][i] - tracerthetis1[i])/tracer_solution['Tracer'][i]) for i in range(len(tracerthetis1))]) < 0.12, "error in tracer"

assert max([abs((bed_solution['Bathymetry'][i] - baththetis1[i])) for i in range(len(baththetis1))]) < 0.007, "error in bed level"
