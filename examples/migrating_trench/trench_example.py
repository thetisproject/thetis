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

conservative = True

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

def update_forcings_tracer(t_new):
    s.update(t_new, solver_obj)


## Note it is necessary to run trench_hydro first to get the hydrodynamics simulation

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

mor_fac = 100
end_time = 15*3600

diffusivity = 0.15
viscosity_hydro = Constant(1e-6)

# initialise velocity, elevation and depth
elev, uv = initialise_fields(mesh2d, 'hydrodynamics_trench', outputdir)

s = SedimentModel(morfac=mor_fac, suspendedload=True, convectivevel=True,
                  bedload=True, angle_correction=True, slope_eff=True, seccurrent=False,
                  mesh2d=mesh2d, bathymetry_2d=bathymetry_2d, uv_init = uv, elev_init = elev,
                  outputdir=outputdir, ks=0.025, average_size=160 * (10**(-6)), dt=0.3, final_time=end_time, cons_tracer = conservative, wetting_and_drying = False, wetting_alpha = 0.1)

# final time of simulation
t_end = end_time/mor_fac

# export interval in seconds
t_export = t_end/45

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, s.bathymetry_2d)

options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True

if s.suspendedload:
    # switch on tracer calculation if using sediment transport component
    options.solve_tracer = True
    options.solve_sediment = True
    options.use_tracer_conservative_form = s.cons_tracer
    options.fields_to_export = ['tracer_2d', 'uv_2d', 'elev_2d']
    options.tracer_advective_velocity_factor = s.corr_factor_model.corr_vel_factor
    options.tracer_source_2d = s.ero_term
    options.tracer_sink_2d = s.depo_term
    #options.tracer_depth_integ_source = s.ero
    #options.tracer_depth_integ_sink = s.depo_term
    options.check_tracer_conservation = True
    options.use_lax_friedrichs_tracer = False
else:
    options.solve_tracer = False
    options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']

options.morphological_acceleration_factor = Constant(mor_fac)
# using nikuradse friction
options.nikuradse_bed_roughness = s.ksp

# set horizontal diffusivity parameter
options.horizontal_diffusivity = Constant(diffusivity)
options.horizontal_viscosity = Constant(viscosity_hydro)
# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.use_wetting_and_drying = s.wetting_and_drying
options.wetting_and_drying_alpha = Constant(s.wetting_alpha)
options.norm_smoother = Constant(s.wetting_alpha)

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = s.dt

c = call.TracerTotalMassConservation2DCallback('tracer_2d',
                                               solver_obj, export_to_hdf5=True, append_to_log=False)
solver_obj.add_callback(c, eval_interval='timestep')

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

if s.suspendedload:
    solver_obj.bnd_functions['tracer'] = {left_bnd_id: {'value': s.sediment_rate, 'flux': Constant(-0.22)}, right_bnd_id: {'elev': Constant(0.397)} }

    # set initial conditions
    solver_obj.assign_initial_conditions(uv=s.uv_init, elev=s.elev_init, tracer=s.testtracer)

else:
    # set initial conditions
    solver_obj.assign_initial_conditions(uv=s.uv_init, elev=s.elev_init)

# run model
solver_obj.iterate(update_forcings=update_forcings_tracer)

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
