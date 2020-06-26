"""
Migrating Trench Test case
=======================

Solves the initial hydrodynamics simulation of a migrating trench.

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *

import numpy as np
import time

def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")
    chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_CREATE)
    chk.store(uv, name="velocity")
    File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_CREATE)
    chk.store(elev, name="elevation")
    File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()


# define mesh
lx = 16
ly = 1.1
nx = lx*5
ny = 5
mesh2d = RectangleMesh(nx, ny, lx, ly)

x, y = SpatialCoordinate(mesh2d)

# define function spaces
V = get_function_space(mesh2d, "CG", 1)
P1_2d = get_function_space(mesh2d, "DG", 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                                                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv,
                                                                                                                          depth_riv))))
bathymetry_2d.interpolate(-trench)


# simulate initial hydrodynamics
# define initial elevation
elev_init = Function(P1_2d).interpolate(Constant(0.4))
uv_init = as_vector((0.51, 0.0))

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

print_output('Exporting to '+outputdir)

t_end = 500

# export interval in seconds
t_export = np.round(t_end/40, 0)

# define parameters
average_size=160 * (10**(-6))
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
options.horizontal_viscosity = Constant(1e-6)

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.norm_smoother = Constant(0.1)

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 0.25

# set boundary conditions

left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}

swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}    

solver_obj.bnd_functions['shallow_water'] = swe_bnd

solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

# run model
solver_obj.iterate()

uv, elev = solver_obj.fields.solution_2d.split()
export_final_state("hydrodynamics_trench", uv, elev)