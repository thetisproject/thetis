"""
Module for 2D depth averaged solver
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import timeintegrator
from . import rungekutta
from . import implicitexplicit
from . import timestepper_nh
import weakref
from .log import *
from .solver2d import FlowSolver2d
from .solver import FlowSolver


class FlowSolverNH(FlowSolver2d, FlowSolver):
    """
    Main object for the non-hydrostatic solver with various approaches

    **Example**

    Create mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create solver object and set some options

    .. code-block:: python

        solver_obj = solver_nh.FlowSolverNH(mesh2d, bathymetry_2d, n_layers=1, use_2d_solver=True)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'CrankNicolson'
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0
        options_nh = options.nh_model_options
        options_nh.solve_nonhydrostatic_pressure = True

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 use_2d_solver=True, options=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :arg int n_layers: Number of layers in the vertical direction.
            Elements are distributed uniformly over the vertical.
        :arg bool use_2d_solver: `True` stands for the use of solver with
            the 2D mesh only.
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions2d` instance
        """
        if use_2d_solver:
            super(FlowSolverNH, self).__init__(mesh2d, bathymetry_2d, options=None)
        else:
            super(FlowSolver2d, self).__init__(mesh2d, bathymetry_2d, n_layers, options=None)

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        if self.options.nh_model_options.use_2d_solver:
            super(FlowSolverNH, self).create_function_spaces()
        else:
            super(FlowSolver2d, self).create_function_spaces()

    def set_sipg_parameter(self):
        r"""
        Compute a penalty parameter which ensures stability of the Interior Penalty method
        used for viscosity and diffusivity terms, from Epshteyn et al. 2007
        (http://dx.doi.org/10.1016/j.cam.2006.08.029).

        The scheme is stable if

        ..math::
            \alpha|_K > 3*X*p*(p+1)*\cot(\theta_K),

        for all elements :math:`K`, where

        ..math::
            X = \frac{\max_{x\in K}(\nu(x))}{\min_{x\in K}(\nu(x))},

        :math:`p` the degree, and :math:`\theta_K` is the minimum angle in the element.
        """
        if self.options.nh_model_options.use_2d_solver:
            super(FlowSolverNH, self).set_sipg_parameter()
        else:
            super(FlowSolver2d, self).set_sipg_parameter()

    def poisson_solver(self, q):
        """
        Solvers for Poisson equation and subsequently updating velocities

        Generic forms:
           2D: `div(grad(q)) + inner(A, grad(q)) + B*q = C`
           3D: `div(grad(q)) = rho_0/dt*div(uv_3d)`

        :arg A, B and C: Known functions, constants or expressions
        :type A: vector, B: scalar, C: scalar (3D: div terms). Valued :class:`Function`, `Constant`, or an expression
        :arg q: Non-hydrostatic pressure to be solved and output
        :type q: scalar function 3D or 2D :class:`Function`
        """
        rho_0 = physical_constants['rho0']
        fs_q = q.function_space()
        test_q = TestFunction(fs_q)
        normal = FacetNormal(fs_q.mesh())
        boundary_markers = fs_q.mesh().exterior_facets.unique_markers

        if self.options.nh_model_options.use_2d_solver:
            q_2d = q
            uv_2d, elev_2d = self.fields.solution_2d.split()
            w_2d = self.w_2d
            bath_2d = self.fields.bathymetry_2d
            h_star = self.depth.get_total_depth(elev_2d)
            w_b = -dot(uv_2d, grad(bath_2d))  # TODO account for bed movement

            A = grad(elev_2d - bath_2d)/h_star
            B = div(A) - 4./(h_star**2)
            C = 2.*rho_0/self.dt*(div(uv_2d) + (w_2d - w_b)/(0.5*h_star))

            # weak forms
            f = -dot(grad(q_2d), grad(test_q))*dx - q_2d*div(A*test_q)*dx + B*q_2d*test_q*dx - C*test_q*dx
            # boundary conditions
            bcs = []
            for bnd_marker in boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if func is not None:  # e.g. inlet flow, TODO be more precise
                    bc = DirichletBC(fs_q, 0., int(bnd_marker))
                    bcs.append(bc)
                else:
                    # Neumann boundary condition => inner(grad(q_2d), normal)=0.
                    f += (q_2d*inner(A, normal))*test_q*ds_bnd

            prob_q = NonlinearVariationalProblem(f, q_2d)
            solver_q = NonlinearVariationalSolver(
                prob_q,
                solver_parameters={'snes_type': 'ksponly',
                                   'ksp_type': 'preonly',
                                   'mat_type': 'aij',
                                   'pc_type': 'lu'},
                options_prefix='poisson_solver')

            # velocity updators
            tri_u = TrialFunction(self.function_spaces.U_2d)
            test_u = TestFunction(self.function_spaces.U_2d)
            a_u = inner(tri_u, test_u)*dx
            l_u = dot(uv_2d - 0.5*self.dt/rho_0*(grad(q_2d) + A*q_2d), test_u)*dx
            prob_u = LinearVariationalProblem(a_u, l_u, uv_2d)
            solver_u = LinearVariationalSolver(prob_u)

            tri_w = TrialFunction(self.function_spaces.H_2d)
            test_w = TestFunction(self.function_spaces.H_2d)
            a_w = inner(tri_w, test_w)*dx
            l_w = dot(w_2d + self.dt/rho_0*(q_2d/h_star), test_w)*dx
            prob_w = LinearVariationalProblem(a_w, l_w, w_2d)
            solver_w = LinearVariationalSolver(prob_w)

        else:
            # q_3d = q
            raise NotImplementedError("3D Poisson equation has not been implemented currently in master branch.")

        return solver_q, solver_u, solver_w

    def create_equations(self):
        """
        Creates equations for non-hydrostatic model
        """
        if self.options.nh_model_options.use_2d_solver:
            super(FlowSolverNH, self).create_equations()
            if self.options.nh_model_options.solve_nonhydrostatic_pressure:
                print_output('Using non-hydrostatic model with {:} vertical layer'.format(self.options.nh_model_options.n_layers))
                print_output('... using 2D mesh based solver')
                self._isfrozen = False
                fs_q = get_functionspace(self.mesh2d, 'CG', self.options.polynomial_degree)
                self.fields.q_2d = Function(fs_q)  # 2d non-hydrostatic pressure at bottom
                self.w_2d = Function(self.function_spaces.H_2d)  # depth-averaged vertical velocity
                # free surface equation
                self.eq_free_surface = shallowwater_eq.FreeSurfaceEquation(
                    TestFunction(self.function_spaces.H_2d), self.function_spaces.H_2d, self.function_spaces.U_2d,
                    self.depth, self.options)
                self.eq_free_surface.bnd_functions = self.bnd_functions['shallow_water']
                self._isfrozen = True
        else:
            super(FlowSolver2d, self).create_equations()
            print_output('... using 3D extruded mesh based solver ...')
            raise NotImplementedError("3D model has not been implemented currently in master branch.")

    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        super(FlowSolverNH, self).create_timestepper()
        if self.options.nh_model_options.solve_nonhydrostatic_pressure:
            assert self.options.nh_model_options.use_2d_solver
            self._isfrozen = False
            # TODO modift to no-copy of `steppers`
            steppers = {
                'SSPRK33': rungekutta.SSPRK33,
                'ForwardEuler': timeintegrator.ForwardEuler,
                'SteadyState': timeintegrator.SteadyState,
                'BackwardEuler': rungekutta.BackwardEulerUForm,
                'DIRK22': rungekutta.DIRK22UForm,
                'DIRK33': rungekutta.DIRK33UForm,
                'CrankNicolson': timeintegrator.CrankNicolson,
                'PressureProjectionPicard': timeintegrator.PressureProjectionPicard,
                'SSPIMEX': implicitexplicit.IMEXLPUM2,
            }
            self.timestepper = timestepper_nh.TimeStepper2d(
                weakref.proxy(self), steppers[self.options.timestepper_type],
            )

            # solvers for 2D Poisson equation and subsequent update of velocities
            self.solver_q, self.solver_u, self.solver_w = self.poisson_solver(self.fields.q_2d)

            self._isfrozen = True
