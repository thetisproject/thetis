from thetis import *
from firedrake_adjoint import *
import numpy
import thetis.inversion_tools as inversion_tools
from model_config import *
import argparse
import os

# parse user input
parser = argparse.ArgumentParser(
    description='North Sea inversion problem',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--controls', nargs='+',
                    help='Control variable(s) to optimize',
                    choices=['Bathymetry', 'Manning'],
                    default=['Manning'],
                    )
parser.add_argument('--no-consistency-test', action='store_true',
                    help='Skip consistency test')
parser.add_argument('--no-taylor-test', action='store_true',
                    help='Skip Taylor test')
args = parser.parse_args()
controls = sorted(args.controls)
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test
no_exports = os.getenv('THETIS_REGRESSION_TEST') is not None

# Create the solver object
pwd = os.path.abspath(os.path.dirname(__file__))
start_date = datetime.datetime(2022, 1, 15, tzinfo=sim_tz)
end_date = datetime.datetime(2022, 1, 18, tzinfo=sim_tz)
solver_obj, start_time, update_forcings = construct_solver(
    start_date=start_date,
    end_date=end_date,
    output_directory=f'{pwd}/outputs',
    store_station_time_series=not no_exports,
    no_exports=no_exports,
)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d
manning_2d = solver_obj.fields.manning_2d

# Set output directory
output_dir_suffix = '_' + '-'.join(controls) + '-opt'
options.output_directory += output_dir_suffix

# Setup controls and regularization parameters
gamma_hessian_list = []
control_bounds_list = []
for control_name in controls:
    if control_name == 'Bathymetry':
        bounds = [1.0, 1e4]
        gamma_hessian = Constant(1e-3)
    elif control_name == 'Manning':
        bounds = [1e-4, 1e-1]
        gamma_hessian = Constant(1.0)
    else:
        raise ValueError(f'Unsupported control variable {control_name}')
    print_output(f'{control_name} regularization params: hess={float(gamma_hessian):.3g}')
    gamma_hessian_list.append(gamma_hessian)
    control_bounds_list.append(bounds)
# reshape to [[lo1, lo2, ...], [hi1, hi2, ...]]
control_bounds = numpy.array(control_bounds_list).T

# Assign initial conditions
print_output('Exporting to ' + options.output_directory)
solver_obj.load_state(14, outputdir="outputs_spinup", t=0, iteration=0)
u, eta = solver_obj.fields.solution_2d.copy(deepcopy=True).split()
solver_obj.assign_initial_conditions(uv=u, elev=eta)
update_forcings(0.0)

# Create station manager
observation_data_dir = f'{pwd}/observations'
variable = 'elev'
station_names = list(read_station_data().keys())
sta_manager = inversion_tools.StationObservationManager(
    mesh2d, output_directory=options.output_directory)
sta_manager.load_observation_data(observation_data_dir, station_names, variable)
sta_manager.set_model_field(solver_obj.fields.elev_2d)

# Define the scaling for the cost function so that J ~ O(1)
t_end = (end_date - start_date).total_seconds()
J_scalar = Constant(solver_obj.dt / t_end)

# Create inversion manager and add controls
inv_manager = inversion_tools.InversionManager(
    sta_manager, output_dir=options.output_directory, no_exports=no_exports,
    penalty_parameters=gamma_hessian_list, cost_function_scaling=J_scalar,
    test_consistency=do_consistency_test, test_gradient=do_taylor_test)
for control_name in controls:
    if control_name == 'Bathymetry':
        inv_manager.add_control(bathymetry_2d)
    elif control_name == 'Manning':
        inv_manager.add_control(manning_2d)

# Extract the regularized cost function
cost_function = inv_manager.get_cost_function(solver_obj, weight_by_variance=True)

# Solve and setup reduced functional
solver_obj.iterate(update_forcings=cost_function)
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1  # scipy diagnostics -1, 0, 1, 99, 100, 101
opt_options = {
    'maxiter': 100,
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
    name = cc.name()
    oc.rename(name)
    print_function_value_range(oc, prefix='Optimal')
    if not no_exports:
        File(f'{options.output_directory}/{name}_optimised.pvd').write(oc)
