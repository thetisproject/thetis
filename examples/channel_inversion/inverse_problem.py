from thetis import *
from firedrake_adjoint import *
import numpy
import thetis.inversion_tools as inversion_tools
from model_config import construct_solver
import argparse
import os

# parse user input
parser = argparse.ArgumentParser(
    description='Channel inversion problem',
    # includes default values in help entries
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--controls', nargs='+',
                    help='Control variable(s) to optimize',
                    choices=['Bathymetry', 'Manning', 'InitialElev'],
                    default=['Bathymetry'],
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

pwd = os.path.abspath(os.path.dirname(__file__))
solver_obj = construct_solver(
    output_directory=f'{pwd}/outputs',
    store_station_time_series=not no_exports,
    no_exports=no_exports,
)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d
manning_2d = solver_obj.fields.manning_2d
elev_init_2d = solver_obj.fields.elev_init_2d

output_dir_suffix = '_' + '-'.join(controls) + '-opt'
options.output_directory += output_dir_suffix

gamma_hessian_list = []
control_bounds_list = []
op = inversion_tools.OptimisationProgress(options.output_directory, no_exports=no_exports)

for control_name in controls:
    if control_name == 'Bathymetry':
        bathymetry_2d.assign(5.0)
        bounds = [1.0, 50.]
        op.add_control(bathymetry_2d)
        gamma_hessian = Constant(1e-3)  # regularization parameter
    elif control_name == 'Manning':
        manning_2d.assign(1.0e-03)
        bounds = [1e-4, 1e-1]
        op.add_control(manning_2d)
        gamma_hessian = Constant(1.0)
    elif control_name == 'InitialElev':
        elev_init_2d.assign(0.5)
        bounds = [-10., 10.]
        op.add_control(elev_init_2d)
        gamma_hessian = Constant(0.1)
    else:
        raise ValueError(f'Unsupported control variable {control_name}')
    print_output(f'{control_name} regularization params: hess={float(gamma_hessian):.3g}')
    gamma_hessian_list.append(gamma_hessian)
    control_bounds_list.append(bounds)
# reshape to [[lo1, lo2, ...], [hi1, hi2, ...]]
control_bounds = numpy.array(control_bounds_list).T

print_output('Exporting to ' + options.output_directory)
solver_obj.assign_initial_conditions(elev=elev_init_2d, uv=Constant((1e-5, 0)))

# define the cost function
# scale cost function so that J ~ O(1)
dt_const = solver_obj.dt
total_time_const = options.simulation_end_time
J_scalar = dt_const/total_time_const

observation_data_dir = f'{pwd}/outputs_forward'
variable = 'elev'
station_names = [
    'stationA',
    'stationB',
    'stationC',
    'stationD',
    'stationE',
]
stationmanager = inversion_tools.StationObservationManager(
    mesh2d, J_scalar=J_scalar, output_directory=options.output_directory)
stationmanager.load_observation_data(observation_data_dir, station_names, variable)
stationmanager.set_model_field(solver_obj.fields.elev_2d)

# regularization for each control field
reg_manager = inversion_tools.ControlRegularizationManager(
    op.control_coeff_list, gamma_hessian_list, J_scalar=J_scalar)


def cost_function():
    """
    Compute square misfit between data and observations.
    """
    t = solver_obj.simulation_time

    J_misfit = stationmanager.eval_cost_function(t)
    op.J += J_misfit


def gradient_eval_callback(j, djdm, m):
    """
    Stash optimisation state.
    """
    op.set_control_state(j, djdm, m)
    op.nb_grad_evals += 1


# compute regularization term
op.J = reg_manager.eval_cost_function()

# Solve and setup reduced functional
solver_obj.iterate(export_func=cost_function)
Jhat = ReducedFunctional(op.J, op.control_list, derivative_cb_post=gradient_eval_callback)
stop_annotating()

# Consistency test
if do_consistency_test:
    print_output('Running consistency test')
    J = Jhat(op.control_coeff_list)
    assert numpy.isclose(J, op.J)
    print_output('Consistency test passed!')

# Taylor test
if do_taylor_test:
    func_list = []
    for f in op.control_coeff_list:
        dc = Function(f.function_space()).assign(f)
        func_list.append(dc)
    minconv = taylor_test(Jhat, op.control_coeff_list, func_list)
    assert minconv > 1.9
    print_output('Taylor test passed!')


def optimization_callback(m):
    """
    Stash optimisation progress after successful line search.
    """
    op.update_progress()
    stationmanager.dump_time_series()


# Run inversion
opt_method = 'L-BFGS-B'
opt_verbose = -1  # scipy diagnostics -1, 0, 1, 99, 100, 101
opt_options = {
    'maxiter': 6,  # NOTE increase to run iteration longer
    'ftol': 1e-5,
    'disp': opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    opt_options['maxiter'] = 1

print_output(f'Running {opt_method} optimization')
op.reset_counters()
op.start_clock()
J = float(Jhat(op.control_coeff_list))
op.set_initial_state(J, Jhat.derivative(), op.control_coeff_list)
stationmanager.dump_time_series()
control_opt_list = minimize(
    Jhat, method=opt_method, bounds=control_bounds,
    callback=optimization_callback, options=opt_options)
if isinstance(control_opt_list, Function):
    control_opt_list = [control_opt_list]
for oc, cc in zip(control_opt_list, op.control_coeff_list):
    name = cc.name()
    oc.rename(name)
    print_function_value_range(oc, prefix='Optimal')
    if not no_exports:
        File(f'{options.output_directory}/{name}_optimised.pvd').write(oc)
