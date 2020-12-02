"""
Meander Test case
=======================

Solves the initial hydrodynamics simulation of a meander.

Note this is not the main run-file and is just used to create an initial checkpoint for
the morphodynamic simulation.

For more details of the test case set-up see
[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *

import numpy as np
import time

# define mesh
mesh2d = Mesh("meander.msh")
x, y = SpatialCoordinate(mesh2d)

# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)
P1_2d = FunctionSpace(mesh2d, 'DG', 1)
vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)

# define underlying bathymetry

bathymetry_2d = Function(V, name='Bathymetry')

gradient = Constant(0.0035)

L_function = Function(V).interpolate(conditional(x > 5, pi*4*((pi/2)-acos((x-5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi, pi*4*((pi/2)-acos((-x+5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi))

bathymetry_2d1 = Function(V).interpolate(conditional(y > 2.5, conditional(x < 5, (L_function*gradient) + 9.97072, -(L_function*gradient) + 9.97072), 9.97072))

init = max(bathymetry_2d1.dat.data[:])
final = min(bathymetry_2d1.dat.data[:])

bathymetry_2d2 = Function(V).interpolate(conditional(x <= 5, conditional(y <= 2.5, -9.97072 + gradient*abs(y - 2.5) + init, 0), conditional(y <= 2.5, -9.97072 - gradient*abs(y - 2.5) + final, 0)))
bathymetry_2d = Function(V).interpolate(-bathymetry_2d1 - bathymetry_2d2)

# simulate initial hydrodynamics
# define initial elevation
elev_init = Function(P1_2d).interpolate(0.0544 - bathymetry_2d)
# define initial velocity
uv_init = Function(vectorP1_2d).interpolate(as_vector((0.001, 0.001)))

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

print_output('Exporting to '+outputdir)

t_end = 200
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # run as tests, not sufficient for proper spin up
    # but we simply want a run-through-without-error test
    t_end = 50

# export interval in seconds
t_export = np.round(t_end/40, 0)

# define parameters
average_size = 10**(-3)
ksp = Constant(3*average_size)

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir

options.check_volume_conservation_2d = True

options.fields_to_export = ['uv_2d', 'elev_2d']
options.solve_tracer = False
options.use_lax_friedrichs_tracer = False

# using nikuradse friction
options.nikuradse_bed_roughness = ksp
# setting viscosity
options.horizontal_viscosity = Constant(5*10**(-2))

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 1

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

elev_init_const = (-max(bathymetry_2d.dat.data[:]) + 0.05436)

swe_bnd[3] = {'un': Constant(0.0)}
swe_bnd[1] = {'flux': Constant(-0.02)}
swe_bnd[2] = {'elev': Constant(elev_init_const), 'flux': Constant(0.02)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

# run model
solver_obj.iterate()

# store hydrodynamics for next simulation
uv, elev = solver_obj.fields.solution_2d.split()

checkpoint_dir = "hydrodynamics_meander"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
chk = DumbCheckpoint(checkpoint_dir + "/velocity", mode=FILE_CREATE)
chk.store(uv, name="velocity")
chk.close()
chk = DumbCheckpoint(checkpoint_dir + "/elevation", mode=FILE_CREATE)
chk.store(elev, name="elevation")
chk.close()
