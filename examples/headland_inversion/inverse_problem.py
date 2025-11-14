from thetis import *
import thetis.inversion_tools as inversion_tools
from firedrake import *
from firedrake.adjoint import *
from model_config import construct_solver
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import h5py
import argparse
from mpi4py import MPI
from checkpoint_schedules import SingleMemoryStorageSchedule
import logging

# ---------------------------------------- Step 1: set up mesh and ground truth ----------------------------------------

# parse user input
parser = argparse.ArgumentParser(
    description='Channel inversion problem',
    # includes default values in help entries
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--case', nargs='+',
                    help='Method of Manning field representation',
                    choices=['Uniform', 'Regions', 'IndependentPointsScheme', 'GradientReg', 'HessianReg'],
                    default=['IndependentPointsScheme'],
                    )
parser.add_argument('--no-consistency-test', action='store_true',
                    help='Skip consistency test')
parser.add_argument('--no-taylor-test', action='store_true',
                    help='Skip Taylor test')
args = parser.parse_args()
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test
no_exports = os.getenv('THETIS_REGRESSION_TEST') is not None

case_to_output_dir = {
    'Uniform': 'uniform_friction',
    'Regions': 'region_based',
    'IndependentPointsScheme': 'independent_points_scheme',
    'GradientReg': 'gradient_regularised',
    'HessianReg': 'hessian_regularised'
}

selected_case = args.case[0]
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir_forward = os.path.join(pwd, 'outputs', 'outputs_forward')
output_dir_invert = os.path.join(pwd, 'outputs', 'outputs_inverse', case_to_output_dir[selected_case])

continue_annotation()

tape = get_working_tape()
output_logger.handlers = []
output_logger.propagate = False
tape.enable_checkpointing(
    SingleMemoryStorageSchedule(),
    gc_timestep_frequency=28,  # every 28 time steps we will do garbage collection
    gc_generation=2
)
if COMM_WORLD.rank == 0:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    output_logger.addHandler(handler)
else:
    output_logger.addHandler(logging.NullHandler())

solver_obj, update_forcings = construct_solver(
    output_directory=output_dir_invert,
    store_station_time_series=False,
    no_exports=True,
)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d
manning_2d = solver_obj.fields.manning_2d

coordinates = mesh2d.coordinates.dat.data[:]
x, y = coordinates[:, 0], coordinates[:, 1]
lx = mesh2d.comm.allreduce(np.max(x), MPI.MAX)
ly = mesh2d.comm.allreduce(np.max(y), MPI.MAX)

local_N = coordinates.shape[0]
N = mesh2d.comm.allreduce(local_N, MPI.SUM)  # allreduce sums the local numbers to get the total number of coordinates
masks, M, m_true = None, 0, []

# Create a FunctionSpace on the mesh (corresponds to Manning)
V = get_functionspace(mesh2d, 'CG', 1)

if selected_case == 'Uniform':
    manning_2d.assign(domain_constant(0.02, mesh2d, name='Manning'))
elif selected_case == 'Regions':
    # Define our values for n
    mask_values = np.array([
        ((x < lx / 2) & (y < ly / 6)),
        ((x >= lx / 2) & (y < ly / 6)),
        ((x < lx / 2) & (y >= ly / 6) & (y < 8 * ly / 15)),
        ((x >= lx / 2) & (y >= ly / 6) & (y < 8 * ly / 15)),
        ((x < 3 * lx / 8) & (y >= 8 * ly / 15) & (y < 5 * ly / 6)),
        ((x >= 3 * lx / 8) & (x < 0.5 * lx) & (y >= 8 * ly / 15)),
        ((x >= 0.5 * lx) & (x < 5 * lx / 8) & (y >= 8 * ly / 15)),
        ((x >= 5 * lx / 8) & (y >= 8 * ly / 15) & (y < 5 * ly / 6)),
        ((x < 3 * lx / 8) & (y >= 5 * ly / 6)),
        ((x >= 5 * lx / 8) & (y >= 5 * ly / 6))
    ], dtype=float)

    m_true = [domain_constant(0.03 - 0.0005 * i, mesh2d) for i in range(len(mask_values))]
    masks = [Function(V) for _ in range(len(mask_values))]
    for mask, values in zip(masks, mask_values):
        mask.dat.data[:] = values

    M = len(m_true)

    manning_2d.assign(0)
    for m_, mask_ in zip(m_true, masks):
        manning_2d += m_ * mask_
elif selected_case == 'IndependentPointsScheme':
    # Define our values for n
    points = np.array([
        [0, 0], [0, 0.5], [0, 1], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5],
        [0.1, 0.9], [0.3, 0.9], [0.7, 0.9], [0.9, 0.9],
        [0.1, 0.7], [0.3, 0.7], [0.7, 0.7], [0.9, 0.7],
        [0.3, 0.75], [0.7, 0.75], [0.5, 0.4], [0.5, 0.1],
        [0.1, 0.25], [0.3, 0.25], [0.7, 0.25], [0.9, 0.25],
        [0.1, 0.05], [0.3, 0.05], [0.7, 0.05], [0.9, 0.05]
    ]) * np.array([lx, ly])
    m_true = [domain_constant(0.03 - 0.0005 * i, mesh2d) for i in range(len(points))]
    M = len(m_true)

    linear_interpolator = LinearNDInterpolator(points, np.eye(len(points)))
    nearest_interpolator = NearestNDInterpolator(points, np.eye(len(points)))

    # Apply the interpolators to the mesh coordinates
    linear_coefficients = linear_interpolator(coordinates)
    nan_mask = np.isnan(linear_coefficients).any(axis=1)
    linear_coefficients[nan_mask] = nearest_interpolator(coordinates[nan_mask])

    # Create Function objects to store the coefficients
    masks = [Function(V) for _ in range(len(points))]

    # Assign the linear coefficients to the masks
    for i, mask in enumerate(masks):
        mask.dat.data[:] = linear_coefficients[:, i]

    manning_2d.assign(0)
    for m_, mask_ in zip(m_true, masks):
        manning_2d += m_ * mask_
elif selected_case in ('GradientReg', 'HessianReg'):
    pass
else:
    raise ValueError("Unknown optimisation case")

# Assign initial conditions
print_output('Exporting to ' + options.output_directory)

# Create station manager
observation_data_dir = output_dir_forward
variable = 'uv'
stations = [
    ('stationA', (lx/10, ly/2)),
    ('stationB', (lx/2, ly/2)),
    ('stationC', (3*lx/4, ly/4)),
    ('stationD', (3*lx/4, 3*ly/4)),
    ('stationE', (9*lx/10, ly/2)),
]
sta_manager = inversion_tools.StationObservationManager(mesh2d, output_directory=options.output_directory)
if selected_case in ('GradientReg', 'HessianReg'):
    cfs_scalar = 1e2 * solver_obj.dt / options.simulation_end_time
else:
    cfs_scalar = 1e4 * solver_obj.dt / options.simulation_end_time
cost_function_scaling = domain_constant(cfs_scalar, mesh2d)
sta_manager.cost_function_scaling = cost_function_scaling
print_output('Station Manager instantiated.')
station_names, observation_coords, observation_time, observation_u, observation_v = [], [], [], [], []
for name, (sta_x, sta_y) in stations:
    file = os.path.join(output_dir_forward, f'diagnostic_timeseries_{name}_{variable}.hdf5')
    with h5py.File(file) as h5file:
        t = h5file['time'][:].flatten()
        var = h5file[name][:]
        station_names.append(name)
        observation_coords.append((sta_x, sta_y))
        observation_time.append(t)
        observation_u.append(var[:, 0])
        observation_v.append(var[:, 1])
observation_x, observation_y = numpy.array(observation_coords).T
sta_manager.register_observation_data(station_names, variable, observation_time, observation_x, observation_y,
                                      data=(observation_u, observation_v), start_times=None, end_times=None)
print_output('Data registered.')
sta_manager.construct_evaluator()
sta_manager.set_model_field(solver_obj.fields.uv_2d)
print_output('Station Manager set-up complete.')

# -------------------------------------- Step 3: define the optimisation problem ---------------------------------------

# Define the scaling for the cost function so that dJ/dm ~ O(1)
J_scalar = sta_manager.cost_function_scaling

# Setup controls and regularization parameters (regularisation only needed where lots of values are being changed)
gamma_penalty_list = []
control_bounds_list = []
gamma_penalty = None
bounds = [0.01, 0.06]
if selected_case in ('GradientReg', 'HessianReg'):
    gamma_factor = 1e6
    gamma_penalty = Constant(gamma_factor * cfs_scalar)  # scaling is significant due to unit normalisation
    print_output(f'Manning regularization params: gamma={float(gamma_penalty):.3g}')
    gamma_penalty_list.append(gamma_penalty)
control_bounds_list.append(bounds)
# reshape to [[lo1, lo2, ...], [hi1, hi2, ...]]
control_bounds = numpy.array(control_bounds_list).T

# Create inversion manager and add controls
inv_manager = inversion_tools.InversionManager(
    sta_manager, output_dir=options.output_directory, no_exports=False, penalty_parameters=gamma_penalty_list,
    cost_function_scaling=J_scalar, test_consistency=do_consistency_test, test_gradient=do_taylor_test)

print_output('Inversion Manager instantiated.')

if selected_case == 'Uniform':
    manning_const = domain_constant(0.03, mesh2d, name='Manning')
    manning_2d.project(manning_const)
    inv_manager.add_control(manning_const)
elif selected_case in ('Regions', 'IndependentPointsScheme'):
    m_values = [domain_constant(0.03 - 0.0005 * i, mesh2d) for i in range(M)]
    manning_2d.assign(0)
    for m_, mask_ in zip(m_values, masks):
        manning_2d += m_ * mask_
    inv_manager.add_control(m_values, mappings=masks)
else:
    manning_2d.assign(0.04)
    inv_manager.add_control(manning_2d)

if not no_exports:
    VTKFile(os.path.join(output_dir_invert, 'manning_init.pvd')).write(manning_2d)

# Extract the regularized cost function
if selected_case == 'GradientReg':
    cost_function = inv_manager.get_cost_function(solver_obj,
                                                  weight_by_variance=True, regularisation_manager='Gradient')
else:
    cost_function = inv_manager.get_cost_function(solver_obj, weight_by_variance=True)
cost_function_callback = inversion_tools.CostFunctionCallback(solver_obj, cost_function)
solver_obj.add_callback(cost_function_callback, 'timestep')

# Solve and setup reduced functional
num_steps = int(np.ceil(solver_obj.options.simulation_end_time / solver_obj.dt))
adj_timestepper = tape.timestepper(iter(range(num_steps)))
solver_obj.iterate(update_forcings=update_forcings, adj_timestepper=adj_timestepper)
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1  # scipy diagnostics -1, 0, 1, 99, 100, 101
opt_options = {
    'maxiter': 10,  # NOTE increase to run iteration longer
    'ftol': 1e-5,
    'disp': opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    opt_options['maxiter'] = 1
control_opt_list = inv_manager.minimize(
    opt_method='L-BFGS-B', bounds=control_bounds, **opt_options)
if isinstance(control_opt_list, Function):
    control_opt_list = [control_opt_list]
for oc, cc in zip(control_opt_list, inv_manager.control_coeff_list):
    print_function_value_range(oc, prefix='Optimal')
    # NOTE: This would need to be updated to match the controls if bathymetry was also added!
    if selected_case == 'Uniform':
        P1_2d = get_functionspace(mesh2d, 'CG', 1)
        manning_2d = Function(P1_2d, name='Manning coefficient')
        manning_2d.assign(domain_constant(inv_manager.m_list[-1], mesh2d))
        if not no_exports:
            VTKFile(os.path.join(options.output_directory, 'manning_optimised.pvd')).write(manning_2d)
    elif selected_case == 'Regions' or selected_case == 'IndependentPointsScheme':
        P1_2d = get_functionspace(mesh2d, 'CG', 1)
        manning_2d = Function(P1_2d, name='manning2d')
        manning_2d.assign(0)
        for m_, mask_ in zip(inv_manager.m_list, masks):
            manning_2d += m_ * mask_
        if not no_exports:
            VTKFile(os.path.join(options.output_directory, 'manning_optimised.pvd')).write(manning_2d)
    else:
        name = cc.name()
        oc.rename(name)
        print_function_value_range(oc, prefix='Optimal')
        if not no_exports:
            VTKFile(os.path.join(options.output_directory, f'{name}_optimised.pvd')).write(oc)

if selected_case == 'Regions' or selected_case == 'IndependentPointsScheme':
    print_output("Optimised vector m:\n" + str([np.round(control_opt_list[i].dat.data[0], 4) for i in range(len(control_opt_list))]))
