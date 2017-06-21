"""
Tests diagnostic callbacks and hdf file output.
"""
from thetis import *
from thetis.callback import VolumeConservation3DCallback
import h5py
import pytest


@pytest.fixture(scope='session')
def tmp_outputdir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('outputs')
    return str(fn)


def test_callbacks(tmp_outputdir):

    lx = 45000.0
    ly = 3000.0
    nx = 25
    ny = 2
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    depth = 50.0
    elev_amp = 1.0
    n_layers = 6
    # estimate of max advective velocity used to estimate time step
    u_mag = Constant(0.5)

    outputdir = tmp_outputdir
    print_output('Exporting to ' + outputdir)

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # set time step, export interval and run duration
    c_wave = float(np.sqrt(9.81*depth))
    t_cycle = lx/c_wave
    n_steps = 20
    dt = round(float(t_cycle/n_steps))
    t_export = dt
    t_end = 10*t_export

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solver_obj.options
    options.use_nonlinear_equations = False
    options.solve_salinity = True
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = False
    options.dt = dt/40.0
    options.t_export = t_export
    options.t_end = t_end
    options.u_advection = u_mag
    options.check_salinity_conservation = True
    options.check_salinity_overshoot = True
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.fields_to_export = []
    options.fields_to_export_hdf5 = []

    # need to call creator to create the function spaces
    solver_obj.create_equations()
    elev_init = Function(solver_obj.function_spaces.H_2d)
    elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/lx)', eta_amp=elev_amp,
                                 lx=lx))
    salt_init3d = Constant(3.45)

    class ConstCallback(DiagnosticCallback):
        """Simple callback example"""
        name = 'constintegral'
        variable_names = ['constant', 'integral']

        def __init__(self, const_val, solver_obj, outputdir=None, export_to_hdf5=False,
                     append_to_log=True):
            super(ConstCallback, self).__init__(solver_obj,
                                                outputdir,
                                                export_to_hdf5,
                                                append_to_log)
            self.const_val = const_val

        def __call__(self):
            value = self.const_val
            c = Constant(self.const_val,
                         domain=solver_obj.mesh)
            integral = assemble(c*dx)
            return value, integral

        def message_str(self, *args):
            line = 'Constant: {0:11.4e} Integral: {1:11.4e}'.format(*args)
            return line

    const_value = 4.5
    cb = ConstCallback(const_value,
                       solver_obj,
                       export_to_hdf5=True,
                       outputdir=solver_obj.options.output_directory)

    # test call interface
    val, integral = cb()
    assert np.allclose(val, const_value)
    assert np.allclose(integral, const_value*lx*ly*depth)
    msg = cb.message_str(val, integral)
    assert msg == 'Constant:  4.5000e+00 Integral:  3.0375e+10'

    solver_obj.add_callback(cb)
    vcb = VolumeConservation3DCallback(solver_obj)
    solver_obj.add_callback(vcb)
    solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
    solver_obj.iterate()

    # verify hdf file contents

    with h5py.File('outputs/diagnostic_constintegral.hdf5', 'r') as h5file:
        time = h5file['time'][:]
        value = h5file['constant'][:]
        integral = h5file['integral'][:]
        correct_time = np.arange(11, dtype=float)[:, np.newaxis]
        correct_time *= solver_obj.options.t_export
        correct_value = np.ones_like(correct_time)*const_value
        correct_integral = np.ones_like(correct_time)*const_value*lx*ly*depth
        assert np.allclose(time, correct_time)
        assert np.allclose(value, correct_value)
        assert np.allclose(integral, correct_integral)

    with h5py.File('outputs/diagnostic_volume2d.hdf5', 'r') as h5file:
        time = h5file['time'][:]
        reldiff = h5file['relative_difference'][:]
        integral = h5file['integral'][:]
        correct_time = np.arange(11, dtype=float)[:, np.newaxis]
        correct_time *= solver_obj.options.t_export
        correct_integral = np.ones_like(correct_time)*lx*ly*depth
        correct_reldiff = np.zeros_like(correct_time)
        assert np.allclose(time, correct_time)
        assert np.allclose(reldiff, correct_reldiff)
        assert np.allclose(integral, correct_integral)

    with h5py.File('outputs/diagnostic_salt_3d_mass.hdf5', 'r') as h5file:
        time = h5file['time'][:]
        reldiff = h5file['relative_difference'][:]
        integral = h5file['integral'][:]
        correct_time = np.arange(11, dtype=float)[:, np.newaxis]
        correct_time *= solver_obj.options.t_export
        correct_integral = np.ones_like(correct_time)*lx*ly*depth*3.45
        correct_reldiff = np.zeros_like(correct_time)
        assert np.allclose(time, correct_time)
        assert np.allclose(reldiff, correct_reldiff)
        assert np.allclose(integral, correct_integral)


if __name__ == '__main__':
    test_callbacks('outputs')
