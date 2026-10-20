"""Tests for the depth-averaged Smagorinsky viscosity closure."""
import numpy

from thetis import *


def test_smagorinsky_viscosity_2d_affine_velocity():
    """The direct-gradient calculator reproduces constant affine strain."""
    mesh = UnitSquareMesh(4, 4)
    velocity_space = VectorFunctionSpace(mesh, 'CG', 1)
    scalar_space = FunctionSpace(mesh, 'CG', 1)
    x, y = SpatialCoordinate(mesh)
    velocity = Function(velocity_space).interpolate(as_vector((x, 2*y)))
    elem_size = Function(scalar_space).assign(2.0)
    viscosity = Function(scalar_space)

    SmagorinskyViscosity2d(
        velocity, viscosity, Constant(0.1), elem_size, weak_form=False).solve()

    assert numpy.allclose(viscosity.dat.data_ro, 0.04)


def test_smagorinsky_viscosity_2d_weak_gradient_and_cap():
    """P0 velocity uses weak recovery and respects a supplied cap."""
    mesh = UnitSquareMesh(4, 4)
    velocity_space = VectorFunctionSpace(mesh, 'DG', 0)
    scalar_space = FunctionSpace(mesh, 'CG', 1)
    x, y = SpatialCoordinate(mesh)
    velocity = Function(velocity_space).interpolate(as_vector((x, y)))
    elem_size = Function(scalar_space).assign(1.0)
    viscosity = Function(scalar_space)
    cap = Function(scalar_space).assign(1.0e-3)

    SmagorinskyViscosity2d(
        velocity, viscosity, Constant(1.0), elem_size, max_val=cap,
        weak_form=True).solve()

    assert numpy.all(viscosity.dat.data_ro >= 1.0e-10)
    assert numpy.all(viscosity.dat.data_ro <= cap.dat.data_ro)


def _make_solver(use_smagorinsky):
    mesh = UnitSquareMesh(2, 2)
    bathymetry = Function(FunctionSpace(mesh, 'CG', 1)).assign(10.0)
    solver_obj = solver2d.FlowSolver2d(mesh, bathymetry)
    options = solver_obj.options
    options.no_exports = True
    options.timestep = 1.0
    options.use_smagorinsky_viscosity = use_smagorinsky
    options.horizontal_viscosity = Constant(0.5)
    return solver_obj


def test_flow_solver_2d_smagorinsky_wiring():
    solver_obj = _make_solver(True)
    solver_obj.create_timestepper()

    assert 'smag_visc_2d' in solver_obj.fields
    assert solver_obj.smagorinsky_viscosity_solver.max_val is not None
    solver_obj.assign_initial_conditions(uv=Constant((1.0, 0.0)))
    solver_obj.smagorinsky_viscosity_solver.solve()
    assert numpy.all(solver_obj.fields.smag_visc_2d.dat.data_ro >= 1.0e-10)


def test_flow_solver_2d_smagorinsky_disabled():
    solver_obj = _make_solver(False)
    solver_obj.create_timestepper()

    assert 'smag_visc_2d' not in solver_obj.fields
    assert solver_obj.horizontal_viscosity_2d is solver_obj.options.horizontal_viscosity


def test_flow_solver_2d_smagorinsky_cap_modes():
    manual_solver = _make_solver(True)
    manual_solver.options.use_automatic_smagorinsky_viscosity_cap = False
    manual_solver.options.smagorinsky_viscosity_max = 0.02
    manual_solver.create_timestepper()
    assert numpy.allclose(
        manual_solver.smagorinsky_viscosity_solver.max_val.dat.data_ro, 0.02)

    uncapped_solver = _make_solver(True)
    uncapped_solver.options.use_automatic_smagorinsky_viscosity_cap = False
    uncapped_solver.create_timestepper()
    assert uncapped_solver.smagorinsky_viscosity_solver.max_val is None
