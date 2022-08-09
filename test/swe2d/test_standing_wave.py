# Test for temporal convergence of CrankNicolson and pressureprojection picard timesteppers,
# tests convergence of a single period of a standing wave in a rectangular channel.
# This only tests against a linear solution, so does not really test whether the splitting
# in PressureProjectionPicard between nonlinear momentum and linearized wave equation terms is correct.
# PressureProjectionPicard does need two iterations to ensure 2nd order convergence
from thetis import *
import pytest
import math
import h5py


@pytest.mark.parametrize("timesteps,max_rel_err", [
    (10, 0.02), (20, 5e-3), (40, 1.25e-3)])
# with nonlin=True and nx=100 this converges for the series
#  (10,0.02), (20,5e-3), (40, 1.25e-3)
# with nonlin=False further converge is possible
@pytest.mark.parametrize("timestepper", [
    'CrankNicolson', 'PressureProjectionPicard', ])
def test_standing_wave_channel(timesteps, max_rel_err, timestepper, tmpdir, do_export=False):

    lx = 5e3
    ly = 1e3
    nx = 100
    mesh2d = RectangleMesh(nx, 1, lx, ly)

    n = timesteps
    depth = 100.
    g = float(physical_constants['g_grav'])
    c = math.sqrt(g*depth)
    period = 2*lx/c
    dt = period/n
    t_end = period-0.1*dt  # make sure we don't overshoot

    x = SpatialCoordinate(mesh2d)
    elev_init = cos(pi*x[0]/lx)

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(depth)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.timestep = dt
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.simulation_end_time = t_end
    solver_obj.options.no_exports = not do_export
    solver_obj.options.swe_timestepper_type = timestepper
    solver_obj.options.output_directory = str(tmpdir)

    if timestepper == 'CrankNicolson':
        solver_obj.options.element_family = 'dg-dg'
        # Crank Nicolson stops being 2nd order if we linearise
        # (this is not the case for PressureProjectionPicard, as we do 2 Picard iterations)
        solver_obj.options.swe_timestepper_options.use_semi_implicit_linearization = False
    elif timestepper == 'PressureProjectionPicard':
        # this approach currently only works well with dg-cg, because in dg-dg
        # the pressure gradient term puts an additional stabilisation term in the velocity block
        # (even without that term  this approach is not as fast, as the stencil for the assembled schur system
        # is a lot bigger for dg-dg than dg-cg)
        solver_obj.options.element_family = 'dg-cg'
        solver_obj.options.swe_timestepper_options.use_semi_implicit_linearization = True
        solver_obj.options.swe_timestepper_options.picard_iterations = 2
    if hasattr(solver_obj.options.swe_timestepper_options, 'use_automatic_timestep'):
        solver_obj.options.swe_timestepper_options.use_automatic_timestep = False

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(elev=elev_init)

    # first two detector locations are outside domain
    xy = [[-2*lx, ly/2.], [-lx/2, ly/2.], [lx/4., ly/2.], [3*lx/4., ly/2.]]
    # but second one can be moved with dist<lx
    xy = select_and_move_detectors(mesh2d, xy, maximum_distance=lx)
    # thus we should end up with only the first one removed
    assert len(xy) == 3
    numpy.testing.assert_almost_equal(xy[0][0], lx/nx/3.)
    # first set of detectors
    cb1 = DetectorsCallback(solver_obj, xy, ['elev_2d', 'uv_2d'], name='set1',
                            append_to_log=True)
    # same set in reverse order, now with named detectors and only elevations
    cb2 = DetectorsCallback(solver_obj, xy[::-1], ['elev_2d'], name='set2',
                            detector_names=['two', 'one', 'zero'],
                            append_to_log=True)
    solver_obj.add_callback(cb1)
    solver_obj.add_callback(cb2)

    solver_obj.iterate()

    uv, eta = solver_obj.fields.solution_2d.split()

    area = lx*ly
    rel_err = errornorm(elev_init, eta)/math.sqrt(area)
    print_output(rel_err)
    assert rel_err < max_rel_err
    print_output("PASSED")

    with h5py.File(str(tmpdir) + '/diagnostic_set1.hdf5', 'r') as df:
        assert all(df.attrs['field_dims'][:] == [1, 2])
        trange = numpy.arange(n+1)*dt
        numpy.testing.assert_almost_equal(df['time'][:, 0], trange)
        x = lx/4.  # location of detector1
        numpy.testing.assert_allclose(df['detector1'][:][:, 0],
                                      numpy.cos(pi*x/lx)*numpy.cos(2*pi*trange/period),
                                      atol=5e-2, rtol=0.5)
    with h5py.File(str(tmpdir) + '/diagnostic_set2.hdf5', 'r') as df:
        assert all(df.attrs['field_dims'][:] == [1, ])
        x = lx/4.  # location of detector1
        numpy.testing.assert_allclose(df['one'][:][:, 0],
                                      numpy.cos(pi*x/lx)*numpy.cos(2*pi*trange/period),
                                      atol=5e-2, rtol=0.5)


if __name__ == '__main__':
    test_standing_wave_channel(10, 1.6e-02, 'CrankNicolson', 'outputs', do_export=True)
