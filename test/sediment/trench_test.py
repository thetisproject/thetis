"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench with suspended and bedload transport

Tests the implementation of the sediment model and corrective_velocity_factor

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *

import numpy as np
import os

def run_migrating_trench(conservative):
    
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

    morfac = 300
    dt = 0.3
    end_time = 1.5*3600
    
    diffusivity = 0.15
    viscosity_hydro = Constant(1e-6)
    
    # initialise velocity, elevation and depth
    elev = Constant(0.4)
    uv = as_vector((0.51, 0.0))
    
    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    
    options.sediment_model_options.use_sediment_conservative_form = conservative
    options.sediment_model_options.average_sediment_size = 160*(10**(-6))
    options.sediment_model_options.ks = 0.025
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)
    
    options.simulation_end_time = end_time/morfac
    options.no_exports = True
    
    options.check_volume_conservation_2d = True
    
    options.fields_to_export = ['sediment_2d', 'uv_2d', 'elev_2d']  # note exporting bathymetry must be done through export func
    options.sediment_model_options.check_sediment_conservation = True

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
    
    # make sure set all hydrodynamic and sediment flags before creating model
    solver_obj.create_sediment_model(uv_init=uv, elev_init=elev,
                                     erosion='model_def', deposition='model_def')
    
    # set boundary conditions
    
    left_bnd_id = 1
    right_bnd_id = 2
    
    options.sediment_model_options.equilibrium_sediment_bd_ids = {left_bnd_id}
    
    swe_bnd = {}
    
    swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
    swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}
    
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    
    solver_obj.bnd_functions['sediment'] = {left_bnd_id: {'flux': Constant(-0.22)}, right_bnd_id: {'elev': Constant(0.397)}}
    
    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment=solver_obj.sediment_model.equiltracer)

    
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
    sediment_mass_int, sediment_mass_int_rerr = solver_obj.callbacks['export']['sediment_2d mass']()
    print_output("Sediment total mass error: %11.4e" % (sediment_mass_int_rerr))

    assert abs(sediment_mass_int_rerr) < 7e-1, 'sediment is not conserved'

    test_root = os.path.abspath(os.path.dirname(__file__))  # abs path to current file
    sediment_csv_file = os.path.join(test_root, 'sediment.csv')
    bed_csv_file = os.path.join(test_root, 'bed.csv')

    # check sediment and bathymetry values using previous runs
    sediment_solution = np.loadtxt(sediment_csv_file, delimiter=",", skiprows=1)
    bed_solution = np.loadtxt(bed_csv_file, delimiter=",", skiprows=1)
    
    assert max([abs((sediment_solution[i][1] - sedimentthetis1[i])/sediment_solution[i][1]) for i in range(len(sedimentthetis1))]) < 0.15, "error in sediment"
    assert max([abs((bed_solution[i][1] - baththetis1[i])) for i in range(len(baththetis1))]) < 0.003, "error in bed level"


def test_conservative():
    run_migrating_trench(True)


def test_non_conservative():
    run_migrating_trench(False)


if __name__ == '__main__':
    test_non_conservative()
