"""Tests for the Flather shallow-water boundary condition."""
from thetis import *
import math
import pytest


def make_flather_term():
    mesh2d = RectangleMesh(2, 1, 2.0, 1.0)
    p1 = get_functionspace(mesh2d, 'CG', 1)
    bathymetry = Function(p1).assign(4.0)
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    solver_obj.create_equations()
    return solver_obj.equations.sw.terms['HUDivTerm']


def test_flather_boundary_validation_and_velocity():
    term = make_flather_term()
    bnd_conditions = {1: {'flather': {'elev': Constant(1.0), 'un': Constant(0.1)}}}

    assert term.impose_dynamic_bnd(bnd_conditions[1], 1)
    _, h_ext, un_flather = term.get_flather_bnd_functions(Constant(1.2), Constant((0.0, 0.0)), 1,
                                                           bnd_conditions)
    expected = 0.1 + math.sqrt(float(physical_constants['g_grav'])/5.0)*0.2
    assert abs(assemble(h_ext*ds(1, domain=term.mesh))/term.boundary_len[1] - 5.0) < 1.0e-12
    assert abs(assemble(un_flather*ds(1, domain=term.mesh))/term.boundary_len[1] - expected) < 1.0e-12


@pytest.mark.parametrize('bnd_funcs, message', [
    ({'flather': {'elev': Constant(0.0)}}, 'exactly'),
    ({'flather': {'un': Constant(0.0)}}, 'exactly'),
    ({'flather': {'elev': Constant(0.0), 'un': Constant(0.0), 'flux': Constant(0.0)}}, 'exactly'),
    ({'flather': {'elev': Constant(0.0), 'un': Constant(0.0)}, 'elev': Constant(0.0)}, 'cannot'),
])
def test_flather_boundary_validation_errors(bnd_funcs, message):
    term = make_flather_term()
    with pytest.raises(Exception, match=message):
        term.impose_dynamic_bnd(bnd_funcs, 1)


def _run_wave(tmpdir, flather):
    length = 1000.0
    depth = 10.0
    mesh2d = RectangleMesh(50, 1, length, 100.0)
    p1 = get_functionspace(mesh2d, 'CG', 1)
    bathymetry = Function(p1).assign(depth)
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.use_nonlinear_equations = False
    options.swe_timestepper_type = 'CrankNicolson'
    options.swe_timestepper_options.use_semi_implicit_linearization = False
    options.timestep = 2.0
    options.simulation_end_time = 140.0
    options.simulation_export_time = 140.0
    options.output_directory = str(tmpdir)
    options.no_exports = True

    if flather:
        solver_obj.bnd_functions['shallow_water'] = {
            2: {'flather': {'elev': Constant(0.0), 'un': Constant(0.0)}}
        }
    else:
        solver_obj.bnd_functions['shallow_water'] = {}

    x = SpatialCoordinate(mesh2d)[0]
    elevation = 0.1*exp(-((x - 250.0)/60.0)**2)
    velocity = as_vector((sqrt(physical_constants['g_grav']/depth)*elevation, 0.0))
    solver_obj.assign_initial_conditions(elev=elevation, uv=velocity)
    solver_obj.iterate()
    _, eta = solver_obj.fields.solution_2d.subfunctions
    return norm(eta)


def test_flather_transmits_outgoing_wave(tmpdir):
    """An outgoing wave is absorbed more effectively than at a closed boundary."""
    flather_norm = _run_wave(tmpdir, flather=True)
    closed_norm = _run_wave(tmpdir, flather=False)
    assert flather_norm < 0.5*closed_norm
