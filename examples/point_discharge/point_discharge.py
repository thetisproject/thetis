"""
TELEMAC-2D `Point Discharge with Diffusion' test case
=====================================================

Solves tracer advection equation in a rectangular domain with
uniform fluid velocity, constant diffusivity and a constant
tracer source term. Neumann conditions are imposed on the
channel walls and a Dirichlet condition is imposed on the
inflow boundary, with the outflow boundary remaining open.

The two different functional quantities of interest considered
in [2] are evaluated on each mesh and convergence is assessed.
The point source was represented as a circular indicator
function of narrow radius in [2]. In the extended paper, [3],
a Gaussian parametrisation was adopted, with the radius
calibrated using gradient-based optimisation.

Further details for the test case can be found in [1].

[1] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system:
    2D hydrodynamics TELEMAC-2D software release 7.0 user
    manual." Paris:  R&D, Electricite de France, p. 134
    (2014).

[2] J.G. Wallwork, N. Barral, D.A. Ham, M.D. Piggott,
    "Anisotropic Goal-Oriented Mesh Adaptation in Firedrake",
    In: Proceedings of the 28th International Meshing
    Roundtable (2020), DOI:10.5281/zenodo.3653101,
    https://doi.org/10.5281/zenodo.3653101.

[3] J.G. Wallwork, N. Barral, D.A. Ham, M.D. Piggott,
    "Goal-Oriented Error Estimation and Mesh Adaptation for
    Tracer Transport Modelling", submitted to Computer
    Aided Design (2021).
"""
from thetis import *
import numpy as np


def bessi0(x):
    """
    Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'.
    """
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(
        0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = exp(ax)/sqrt(ax)*(0.39894228 + y2*(0.1328592e-1 + y2*(
        0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(
            -0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)


def bessk0(x):
    """
    Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = -ln(x/2.0)*bessi0(x) + (-0.57721566 + y1*(0.42278420 + y1*(
        0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = exp(-x)/sqrt(x)*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(
        -0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)


class PointDischargeParameters(object):
    """
    Problem parameter class, including point source representation.

    Delta functions are not in H1 and hence do not live in the FunctionSpaces we
    seek to use. Here we use a Gaussian approximation with a small radius. The
    small radius has been calibrated against the analytical solution. See [3]
    for details.
    """
    def __init__(self, offset):
        self.offset = offset

        # Physical parameters
        self.diffusivity = Constant(0.1)
        # self.viscosity = Constant(1.0e-08)
        self.viscosity = None
        self.drag = Constant(0.0025)
        self.uv = Constant(as_vector([1.0, 0.0]))
        self.elev = Constant(0.0)

        # Stabilisation
        self.use_lax_friedrichs_tracer = True

        # Parametrisation of point source
        self.source_x, self.source_y = 2.0, 5.0
        self.source_r = 0.05606298 if self.use_lax_friedrichs_tracer else 0.05606298
        self.source_value = 100.0

        # Boundary conditions
        self.boundary_conditions = {
            'tracer': {
                1: {'value': Constant(0.0)},      # inflow
                2: {'open': None},                # outflow
                3: {'diff_flux': Constant(0.0)},  # Neumann
                4: {'diff_flux': Constant(0.0)},  # Neumann
            },
            'shallow_water': {
                1: {
                    'uv': Constant(as_vector([1.0, 0.0])),
                    'elev': Constant(0.0)
                },                                # inflow
                2: {
                    'uv': Constant(as_vector([1.0, 0.0])),
                    'elev': Constant(0.0)
                },                                # outflow
                3: {'un': Constant(0.0)},         # free-slip
                4: {'un': Constant(0.0)},         # free-slip
            }
        }

    def ball(self, mesh, triple, scaling=1.0, eps=1.0e-10):
        x, y = SpatialCoordinate(mesh)
        expr = lt((x-triple[0])**2 + (y-triple[1])**2, triple[2]**2 + eps)
        return conditional(expr, scaling, 0.0)

    def gaussian(self, mesh, triple, scaling=1.0):
        x, y = SpatialCoordinate(mesh)
        expr = exp(-((x-triple[0])**2 + (y-triple[1])**2)/triple[2]**2)
        return scaling*expr

    def source(self, fs):
        triple = (self.source_x, self.source_y, self.source_r)
        return self.gaussian(fs.mesh(), triple, scaling=self.source_value)

    def bathymetry(self, fs):
        return Function(fs).assign(5.0)

    def quantity_of_interest_kernel(self, mesh):
        triple = (20.0, 7.5 if self.offset else 5.0, 0.5)
        area = assemble(self.ball(mesh, triple)*dx)
        area_analytical = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_analytical/area
        return self.ball(mesh, triple, scaling=scaling)

    def quantity_of_interest(self, sol):
        kernel = self.quantity_of_interest_kernel(sol.function_space().mesh())
        return assemble(inner(kernel, sol)*dx(degree=12))

    def analytical_quantity_of_interest(self, mesh):
        """
        The analytical solution can be found in [1]. Due to the modified
        Bessel function, it cannot be evaluated exactly and instead must
        be computed using quadrature.
        """
        x, y = SpatialCoordinate(mesh)
        x0, y0 = self.source_x, self.source_y
        u = self.uv[0]
        D = self.diffusivity
        Pe = 0.5*u/D  # Mesh Peclet number
        r = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0))
        r = max_value(r, self.source_r)  # (Bessel fn explodes at (x0, y0))
        sol = 0.5/(pi*D)*exp(Pe*(x-x0))*bessk0(Pe*r)
        kernel = self.quantity_of_interest_kernel(mesh)
        return assemble(kernel*sol*dx(degree=12))


def solve_tracer(n, offset, hydrodynamics=False):
    """
    Solve the tracer transport problem.

    :arg n: mesh resolution level.
    :arg offset: toggle between aligned and offset source/receiver.
    :kwarg hydrodynamics: solve shallow water equations?
    """
    mesh2d = RectangleMesh(100*2**n, 20*2**n, 50, 10)
    P1_2d = FunctionSpace(mesh2d, "CG", 1)

    # Set up parameter class
    params = PointDischargeParameters(offset)
    source = params.source(P1_2d)

    # Solve tracer transport problem
    solver_obj = solver2d.FlowSolver2d(mesh2d, params.bathymetry(P1_2d))
    options = solver_obj.options
    options.timestepper_type = 'SteadyState'
    options.timestep = 20.0
    options.simulation_end_time = 18.0
    options.simulation_export_time = 18.0
    ts_options = options.timestepper_options
    for sp in (ts_options.solver_parameters, ts_options.solver_parameters_tracer):
        sp['pc_factor_mat_solver_type'] = 'mumps'
        sp['snes_monitor'] = None
    options.fields_to_export = ['tracer_2d', 'uv_2d', 'elev_2d']

    # Hydrodynamics
    options.element_family = 'dg-dg'
    options.horizontal_diffusivity = params.diffusivity
    options.horizontal_viscosity = params.viscosity
    options.quadratic_drag_coefficient = params.drag
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)

    # Passive tracer
    options.solve_tracer = True
    options.tracer_only = not hydrodynamics
    options.use_lax_friedrichs_tracer = params.use_lax_friedrichs_tracer
    options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
    options.use_limiter_for_tracers = True
    options.tracer_source_2d = source

    # Initial and boundary conditions
    solver_obj.bnd_functions = params.boundary_conditions
    uv_init = Constant(as_vector([1.0e-08, 0.0])) if hydrodynamics else params.uv
    solver_obj.assign_initial_conditions(tracer=source, uv=uv_init, elev=params.elev)

    # Solve
    solver_obj.iterate()
    return solver_obj.fields.tracer_2d


def run_convergence(offset, num_levels=4, plot=False, **kwargs):
    """
    Assess convergence of the quantity of interest with increasing DoF count.

    :arg offset: toggle between aligned and offset source/receiver.
    :kwarg num_levels: number of uniform refinements to consider.
    :kwarg plot: toggle plotting of convergence curves.
    :kwargs: other kwargs are passed to `solve_tracer`.
    """
    J = []
    dof_count = []
    params = PointDischargeParameters(offset)

    # Run model on a sequence of uniform meshes and compute QoI error
    for refinement_level in range(num_levels):
        sol = solve_tracer(refinement_level, offset, **kwargs)
        J.append(params.quantity_of_interest(sol))
        dof_count.append(sol.function_space().dof_count)
    J_analytical = params.analytical_quantity_of_interest(sol.function_space().mesh())
    relative_error = np.abs((np.array(J) - J_analytical)/J_analytical)

    # Plot convergence curves
    if plot:
        import matplotlib.pyplot as plt
        from mpltools import annotation

        fig, axes = plt.subplots()
        axes.loglog(dof_count, relative_error, '--x')
        axes.set_xlabel("DoF count")
        axes.set_ylabel("QoI error")
        axes.grid(True)
        annotation.slope_marker((dof_count[1], 0.1), -1, invert=True, ax=axes, size_frac=0.2)
        fname = "outputs/convergence_{:s}.png".format('offset' if offset else 'aligned')
        plt.savefig(fname)

    # Check for linear convergence
    delta_y = np.log10(relative_error[-1]) - np.log10(relative_error[0])
    delta_x = np.log10(dof_count[-1]) - np.log10(dof_count[0])
    rate = abs(delta_y/delta_x)
    assert rate > 0.9, "Sublinear convergence rate {:.4f}".format(rate)


if __name__ == "__main__":
    hydrodynamics = False
    num_levels = 4
    kwargs = dict(num_levels=num_levels, hydrodynamics=hydrodynamics)
    for offset in (False, True):
        run_convergence(offset, plot=True, **kwargs)
