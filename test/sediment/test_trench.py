"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench with suspended sediment transport only.

Tests the implementation of the source term, conservative tracer and corrective_velocity_factor

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous 
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *
import morphological_hydro_fns_comb as morph

import numpy as np
import os

def boundary_conditions_fn_trench(morfac = 1, t_new = 0, state = 'initial'):

    """
    Define boundary conditions for problem to be used in morphological section.
    
    Inputs:
    morfac - morphological scale factor used when calculating time dependent boundary conditions
    t_new - timestep model currently at used when calculating time dependent boundary conditions
    state - when 'initial' this is the initial boundary condition set; when 'update' these are the boundary
            conditions set during update forcings (ie. if fluc_bcs = True, this will be called)
    """
    left_bnd_id = 1
    right_bnd_id = 2
    left_string = ['flux']
    right_string = ['elev']
    
    
    # set boundary conditions

    swe_bnd = {}
    

    flux_constant = -0.22
    elev_constant2 = 0.397
        
    inflow_constant = [flux_constant]
    outflow_constant = [elev_constant2]#, -flux_constant]
    return swe_bnd, left_bnd_id, right_bnd_id, inflow_constant, outflow_constant, left_string, right_string

def run_migrating_trench(conservative):
    # define mesh
    lx = 16
    ly = 1.1
    nx = lx*5
    ny = 5
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    x,y = SpatialCoordinate(mesh2d)

    # define function spaces
    V = FunctionSpace(mesh2d, 'CG', 1)
    P1_2d = FunctionSpace(mesh2d, 'DG', 1)
    vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)

    # define underlying bathymetry
    bathymetry_2d = Function(V, name='Bathymetry')
    initialdepth = Constant(0.397)
    depth_riv = Constant(initialdepth - 0.397)
    depth_trench = Constant(depth_riv - 0.15)
    depth_diff = depth_trench - depth_riv

    trench = conditional(x<3, depth_riv, conditional(x<7.6, ((depth_diff/2)/(tanh(7.6-5.3)))*tanh((x-5.3))+(depth_diff/2),\
             conditional(x < 8.4, depth_trench, conditional(x<13, ((depth_diff/2)/(tanh(8.4-10.7)))*tanh((x-10.7))+ (depth_diff/2),depth_riv))))
    
    bathymetry_2d.interpolate(-trench)

    wd_fn = Constant(0.015)

    solver_obj, update_forcings_tracer = morph.morphological(boundary_conditions_fn = boundary_conditions_fn_trench, morfac = 300, morfac_transport = True, convectivevel = True,\
                    mesh2d = mesh2d, bathymetry_2d = bathymetry_2d, ks = 0.025, average_size = 160 * (10**(-6)), dt = 0.2, final_time = 1.5*3600, 
                    elev_init = Constant(0.4), uv_init = as_vector((0.51, 0.0)), cons_tracer = conservative, wetting_alpha = wd_fn)

    # run model
    solver_obj.iterate(update_forcings = update_forcings_tracer)

    # record final tracer and final bathymetry
    xaxisthetis1 = []
    tracerthetis1 = []
    baththetis1 = []

    for i in np.linspace(0,15.8, 80):
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
    print_output("Tracer total mass error: %11.4e" %(tracer_mass_int_rerr))

    if conservative:
        assert abs(tracer_mass_int_rerr) < 1e-3, 'tracer is not conserved'
    else:
        assert abs(tracer_mass_int_rerr) < 7e-3, 'tracer is not conserved'

    test_root = os.path.abspath(os.path.dirname(__file__))  # abs path to current file
    tracer_csv_file = os.path.join(test_root, 'tracer_test.csv')
    bed_csv_file = os.path.join(test_root, 'bed_test.csv')

    # check tracer and bathymetry values using previous runs    
    tracer_solution = np.loadtxt(tracer_csv_file, delimiter = ",", skiprows = 1)
    bed_solution = np.loadtxt(bed_csv_file, delimiter = ",", skiprows = 1)
    
    assert max([abs((tracer_solution[i][1] - tracerthetis1[i])/tracer_solution[i][1]) for i in range(len(tracerthetis1))]) < 0.8, "error in tracer"
    assert max([abs((bed_solution[i][1] - baththetis1[i])) for i in range(len(baththetis1))]) < 0.002, "error in bed level"
    
def test_conservative():
    run_migrating_trench(True)
    
def test_non_conservative():
    run_migrating_trench(False)

if __name__ == '__main__':
    test_non_conservative()
