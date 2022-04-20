"""
Implements Runge-Kutta time integration methods.

The abstract class :class:`~.AbstractRKScheme` defines the Runge-Kutta
coefficients, and can be used to implement generic time integrators.
"""
from .timeintegrator import *
from abc import ABC, abstractproperty, abstractmethod
import operator
import numpy


def butcher_to_shuosher_form(a, b):
    r"""
    Converts Butcher tableau to Shu-Osher form.

    The Shu-Osher form of a s-stage scheme is defined by two s+1 by s+1 arrays
    :math:`\alpha` and :math:`\beta`:

    .. math::
        u^{0} &= u^n \\
        u^{(i)} &= \sum_{j=0}^s \alpha_{i,j} u^{(j)} + \sum_{j=0}^s \beta_{i,j} F(u^{(j)}) \\
        u^{n+1} &= u^{(s)}

    The Shu-Osher form is not unique. Here we construct the form where beta
    values are the diagonal entries (for DIRK schemes) or sub-diagonal entries
    (for explicit schemes) of the concatenated Butcher tableau [:math:`a`; :math:`b`].

    For more information see Ketchelson et al. (2009) http://dx.doi.org/10.1016/j.apnum.2008.03.034
    """
    import numpy.linalg as linalg

    butcher = numpy.vstack((a, b))

    implicit = numpy.diag(a).any()

    if implicit:
        # a is not singular
        # take diag entries of a to beta
        be_0 = numpy.diag(numpy.diag(a, k=0), k=0)
        be_1 = numpy.zeros_like(b)
        be_1[-1] = b[-1]
        be = numpy.vstack((be_0, be_1))

        n = a.shape[0]
        iden = numpy.eye(n)
        al_0 = iden - numpy.dot(be_0, linalg.inv(a))
        al_1 = numpy.dot((b - be_1), numpy.dot(linalg.inv(be_0), (iden - al_0)))
        al = numpy.vstack((al_0, al_1))

        # construct full shu-osher form
        alpha = numpy.zeros((n+1, n+1))
        alpha[:, 1:] = al
        # consistency
        alpha[:, 0] = 1.0 - numpy.sum(alpha, axis=1)
        beta = numpy.zeros((n+1, n+1))
        beta[:, 1:] = be
    else:
        # a is singular: solve for lower part of butcher tableau
        aa = butcher[1:, :]
        # take diag entries of aa to beta
        be_0 = numpy.diag(numpy.diag(aa, k=0), k=0)
        n = aa.shape[0]
        iden = numpy.eye(n)
        al_0 = iden - numpy.dot(be_0, linalg.inv(aa))

        # construct full shu-osher form
        alpha = numpy.zeros((n+1, n+1))
        alpha[1:, 1:] = al_0
        # consistency
        alpha[:, 0] = 1.0 - numpy.sum(alpha, axis=1)
        beta = numpy.zeros((n+1, n+1))
        beta[1:, :-1] = be_0

    # round off small entries
    alpha[numpy.abs(alpha) < 1e-13] = 0.0
    beta[numpy.abs(beta) < 1e-13] = 0.0

    # check sanity
    assert numpy.allclose(numpy.sum(alpha, axis=1), 1.0)
    if implicit:
        err = beta[:, 1:] - (butcher - numpy.dot(alpha[:, 1:], a))
    else:
        err = beta[:, :-1] - (butcher - numpy.dot(alpha[:, :-1], a))
    assert numpy.allclose(err, 0.0)

    return alpha, beta


class AbstractRKScheme(ABC):
    """
    Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """
    @abstractproperty
    def a(self):
        """Runge-Kutta matrix :math:`a_{i,j}` of the Butcher tableau"""
        pass

    @abstractproperty
    def b(self):
        """weights :math:`b_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def c(self):
        """nodes :math:`c_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def cfl_coeff(self):
        """
        CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.
        """
        pass

    def __init__(self):
        super(AbstractRKScheme, self).__init__()
        self.a = numpy.array(self.a)
        self.b = numpy.array(self.b)
        self.c = numpy.array(self.c)

        assert not numpy.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert numpy.allclose(numpy.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = numpy.vstack((self.a, self.b))

        self.is_implicit = numpy.diag(self.a).any()
        self.is_dirk = numpy.diag(self.a).all()

        if self.is_dirk or not self.is_implicit:
            self.alpha, self.beta = butcher_to_shuosher_form(self.a, self.b)


class ForwardEulerAbstract(AbstractRKScheme):
    """
    Forward Euler method
    """
    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class BackwardEulerAbstract(AbstractRKScheme):
    """
    Backward Euler method
    """
    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class ImplicitMidpointAbstract(AbstractRKScheme):
    r"""
    Implicit midpoint method, second order.

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|c}
        0.5 & 0.5 \\ \hline
            & 1.0
        \end{array}

    """
    a = [[0.5]]
    b = [1.0]
    c = [0.5]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class CrankNicolsonAbstract(AbstractRKScheme):
    """
    Crack-Nicolson scheme
    """
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22Abstract(AbstractRKScheme):
    r"""
    2-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
        \gamma &   \gamma &       0 \\
              1 & 1-\gamma & \gamma \\ \hline
                &       1/2 &     1/2
        \end{array}

    with :math:`\gamma = (2 - \sqrt{2})/2`.

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = (2.0 - numpy.sqrt(2.0))/2.0
    a = [[gamma, 0],
         [1-gamma, gamma]]
    b = [1-gamma, gamma]
    c = [gamma, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK23Abstract(AbstractRKScheme):
    r"""
    2-stage, 3rd order Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
          \gamma &    \gamma &       0 \\
        1-\gamma & 1-2\gamma & \gamma \\ \hline
                  &        1/2 &     1/2
        \end{array}

    with :math:`\gamma = (3 + \sqrt{3})/6`.

    From DIRK(2,3,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = (3 + numpy.sqrt(3))/6
    a = [[gamma, 0],
         [1-2*gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1-gamma]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK33Abstract(AbstractRKScheme):
    """
    3-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = 0.4358665215
    b1 = -3.0/2.0*gamma**2 + 4*gamma - 1.0/4.0
    b2 = 3.0/2.0*gamma**2 - 5*gamma + 5.0/4.0
    a = [[gamma, 0, 0],
         [(1-gamma)/2, gamma, 0],
         [b1, b2, gamma]]
    b = [b1, b2, gamma]
    c = [gamma, (1+gamma)/2, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK43Abstract(AbstractRKScheme):
    """
    4-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    a = [[0.5, 0, 0, 0],
         [1.0/6.0, 0.5, 0, 0],
         [-0.5, 0.5, 0.5, 0],
         [3.0/2.0, -3.0/2.0, 0.5, 0.5]]
    b = [3.0/2.0, -3.0/2.0, 0.5, 0.5]
    c = [0.5, 2.0/3.0, 0.5, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRKLSPUM2Abstract(AbstractRKScheme):
    """
    DIRKLSPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[2.0/11.0, 0, 0],
         [205.0/462.0, 2.0/11.0, 0],
         [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.82


class DIRKLPUM2Abstract(AbstractRKScheme):
    """
    DIRKLPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[2.0/11.0, 0, 0],
         [41.0/154.0, 2.0/11.0, 0],
         [289.0/847.0, 42.0/121.0, 2.0/11.0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [2.0/11.0, 69.0/154.0, 67.0/77.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.09


class SSPRK33Abstract(AbstractRKScheme):
    r"""
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau

    .. math::
        \begin{array}{c|ccc}
            0 &                 \\
            1 & 1               \\
          1/2 & 1/4 & 1/4 &     \\ \hline
              & 1/6 & 1/6 & 2/3
        \end{array}

    CFL coefficient is 1.0
    """
    a = [[0, 0, 0],
         [1.0, 0, 0],
         [0.25, 0.25, 0]]
    b = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class ERKLSPUM2Abstract(AbstractRKScheme):
    """
    ERKLSPUM2, 3-stage, 2nd order Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [5.0/6.0, 0, 0],
         [11.0/24.0, 11.0/24.0, 0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [0, 5.0/6.0, 11.0/12.0]
    cfl_coeff = 1.2


class ERKLPUM2Abstract(AbstractRKScheme):
    """
    ERKLPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [1.0/2.0, 0, 0],
         [1.0/2.0, 1.0/2.0, 0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [0, 1.0/2.0, 1.0]
    cfl_coeff = 2.0


class ERKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class ESDIRKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.0, 0.5]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class ESDIRKTrapezoidAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class RungeKuttaTimeIntegrator(TimeIntegrator, ABC):
    """Abstract base class for all Runge-Kutta time integrators"""
    @abstractmethod
    def get_final_solution(self, additive=False):
        """
        Evaluates the final solution
        """
        pass

    @abstractmethod
    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        pass

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if not self._initialized:
            self.initialize(self.solution)
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()


class DIRKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic implementation of Diagonally Implicit Runge Kutta schemes.

    All derived classes must define the Butcher tableau coefficients :attr:`a`,
    :attr:`b`, :attr:`c`.
    """
    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(DIRKGeneric, self).__init__(equation, solution, fields, dt, options)
        semi_implicit = False
        if hasattr(options, 'use_semi_implicit_linearization'):
            semi_implicit = options.use_semi_implicit_linearization
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')
        self._initialized = False

        fs = self.equation.function_space
        self.solution_old = Function(self.equation.function_space, name='old solution')

        test = self.equation.test
        mixed_space = len(fs) > 1

        # Allocate tendency fields
        self.k = []
        for i in range(self.n_stages):
            fname = f'{self.name}_k{i}'
            self.k.append(Function(fs, name=fname))

        u = self.solution
        u_old = self.solution_old
        if semi_implicit:
            # linearize around last timestep using the fact that all terms are
            # written in the form A(u_nl) u
            u_nl = u_old
        else:
            # solve the full nonlinear residual form
            u_nl = u

        # construct variational problems
        self.F = []
        if not mixed_space:
            for i in range(self.n_stages):
                for j in range(i+1):
                    if j == 0:
                        u = u_old + self.a[i][j]*self.dt_const*self.k[j]
                    else:
                        u += self.a[i][j]*self.dt_const*self.k[j]
                self.F.append(-inner(self.k[i], test)*dx
                              + self.equation.residual(terms_to_add, u, u_nl, fields, fields, bnd_conditions))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in range(self.n_stages):
                for j in range(i+1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(split(u_old), split(self.k[j])):
                            u.append(s + self.a[i][j]*self.dt_const*k)
                    else:
                        for l, k in enumerate(split(self.k[j])):
                            u[l] += self.a[i][j]*self.dt_const*k
                self.F.append(-inner(self.k[i], test)*dx
                              + self.equation.residual(terms_to_add, u, u_nl, fields, fields, bnd_conditions))
        self.update_solver()

        # construct expressions for stage solutions
        self.sol_expressions = []
        for i_stage in range(self.n_stages):
            sol_expr = sum(map(operator.mul, self.k[:i_stage+1], self.dt_const*self.a[i_stage][:i_stage+1]))
            self.sol_expressions.append(sol_expr)
        self.final_sol_expr = u_old + sum(map(operator.mul, self.k, self.dt_const*self.b))

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.update_solver")
    def update_solver(self):
        """Create solver objects"""
        self.solver = []
        for i in range(self.n_stages):
            p = NonlinearVariationalProblem(self.F[i], self.k[i])
            sname = f'{self.name}_stage{i}_'
            self.solver.append(
                NonlinearVariationalSolver(p,
                                           solver_parameters=self.solver_parameters,
                                           options_prefix=sname,
                                           ad_block_tag=self.ad_block_tag + f'_stage{i}'))

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.initialize")
    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.update_solution")
    def update_solution(self, i_stage):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        """
        self.solution.assign(self.solution_old + self.sol_expressions[i_stage])

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.solve_tendency")
    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage.
        """
        if i_stage == 0:
            # NOTE solution may have changed in coupled system
            self.solution_old.assign(self.solution)
        if not self._initialized:
            error('Time integrator {:} is not initialized'.format(self.name))
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.get_final_solution")
    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution`"""
        self.solution.assign(self.final_sol_expr)

    @PETSc.Log.EventDecorator("thetis.DIRKGeneric.solve_stage")
    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.solve_tendency(i_stage, t, update_forcings)
        self.update_solution(i_stage)


class DIRKGenericUForm(RungeKuttaTimeIntegrator):
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE

    @PETSc.Log.EventDecorator("thetis.DIRKGenericUForm.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super().__init__(equation, solution, fields, dt, options)
        semi_implicit = options.use_semi_implicit_linearization
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')
        self._initialized = False

        self.solution_old = Function(self.equation.function_space, name='solution_old')

        self.n_stages = len(self.b)

        # assume final stage is trivial
        assert numpy.array_equal(self.a[-1, :], self.b)

        fs = self.equation.function_space
        test = self.equation.test

        # Allocate tendency fields
        self.k = []
        for i in range(self.n_stages - 1):
            fname = f'{self.name}_k{i}'
            self.k.append(Function(fs, name=fname))

        u = self.solution
        u_old = self.solution_old
        if semi_implicit:
            # linearize around last timestep using the fact that all terms are
            # written in the form A(u_nl) u
            u_nl = u_old
        else:
            # solve the full nonlinear residual form
            u_nl = u
        bnd = bnd_conditions
        fields = self.fields

        # construct variational problems for each stage
        self.F = []
        for i in range(self.n_stages):
            mass = self.equation.mass_term(u) - self.equation.mass_term(u_old)
            rhs = self.dt_const*self.a[i][i]*self.equation.residual('all', u, u_nl, fields, fields, bnd)
            for j in range(i):
                rhs += self.dt_const*self.a[i][j]*inner(self.k[j], test)*dx
            self.F.append(mass - rhs)

        # construct variational problems to evaluate tendencies
        self.k_form = []
        for i in range(self.n_stages - 1):
            kf = self.dt_const*self.a[i][i]*inner(self.k[i], test)*dx - (self.equation.mass_term(u) - self.equation.mass_term(u_old))
            for j in range(i):
                kf += self.dt_const*self.a[i][j]*inner(self.k[j], test)*dx
            self.k_form.append(kf)

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.DIRKGenericUForm.update_solver")
    def update_solver(self):
        """Create solver objects"""
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        self.solver = []
        for i in range(self.n_stages):
            p = NonlinearVariationalProblem(self.F[i], self.solution)
            sname = f'{self.name}_stage{i}_'
            s = NonlinearVariationalSolver(
                p, solver_parameters=self.solver_parameters,
                options_prefix=sname,
                ad_block_tag=self.ad_block_tag + f'_stage{i}')
            self.solver.append(s)
        self.k_solver = []
        k_solver_parameters = {
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'ksp_rtol': 1e-8,
        }
        for i in range(self.n_stages - 1):
            p = NonlinearVariationalProblem(self.k_form[i], self.k[i])
            sname = f'{self.name}_k_stage{i}_'
            s = NonlinearVariationalSolver(
                p, solver_parameters=k_solver_parameters,
                options_prefix=sname,
                ad_block_tag=self.ad_block_tag + f'_k_stage{i}')
            self.k_solver.append(s)

    @PETSc.Log.EventDecorator("thetis.DIRKGenericUForm.initialize")
    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)
        self._initialized = True

    def get_final_solution(self, additive=False):
        """
        Evaluates the final solution
        """
        pass

    @PETSc.Log.EventDecorator("thetis.DIRKGenericUForm.solve_stage")
    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        if i_stage == 0:
            # NOTE solution may have changed in coupled system
            self.solution_old.assign(self.solution)
        if not self._initialized:
            error('Time integrator {:} is not initialized'.format(self.name))
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()
        if i_stage < self.n_stages - 1:
            self.k_solver[i_stage].solve()


class BackwardEuler(DIRKGeneric, BackwardEulerAbstract):
    pass


class ImplicitMidpoint(DIRKGeneric, ImplicitMidpointAbstract):
    pass


class CrankNicolsonRK(DIRKGeneric, CrankNicolsonAbstract):
    pass


class DIRK22(DIRKGeneric, DIRK22Abstract):
    pass


class DIRK23(DIRKGeneric, DIRK23Abstract):
    pass


class DIRK33(DIRKGeneric, DIRK33Abstract):
    pass


class DIRK43(DIRKGeneric, DIRK43Abstract):
    pass


class DIRKLSPUM2(DIRKGeneric, DIRKLSPUM2Abstract):
    pass


class DIRKLPUM2(DIRKGeneric, DIRKLPUM2Abstract):
    pass


class BackwardEulerUForm(DIRKGenericUForm, BackwardEulerAbstract):
    pass


class DIRK22UForm(DIRKGenericUForm, DIRK22Abstract):
    pass


class DIRK33UForm(DIRKGenericUForm, DIRK33Abstract):
    pass


class ERKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form. All terms in the equation are treated explicitly.
    """
    @PETSc.Log.EventDecorator("thetis.ERKGeneric.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(ERKGeneric, self).__init__(equation, solution, fields, dt, options)
        self._initialized = False
        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.tendency = []
        for i in range(self.n_stages):
            k = Function(self.equation.function_space, name='tendency{:}'.format(i))
            self.tendency.append(k)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual(terms_to_add, self.solution, self.solution, self.fields, self.fields, bnd_conditions)

        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        if self._nontrivial:
            self.sol_expressions = []
            for i_stage in range(self.n_stages):
                sol_expr = sum(map(operator.mul, self.tendency[:i_stage], self.a[i_stage][:i_stage]))
                self.sol_expressions.append(sol_expr)
            self.final_sol_expr = sum(map(operator.mul, self.tendency, self.b))

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.update_solver")
    def update_solver(self):
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency[i])
                solver = LinearVariationalSolver(prob, options_prefix=self.name + f'_k{i}',
                                                 solver_parameters=self.solver_parameters,
                                                 ad_block_tag=self.ad_block_tag + f'_k{i}')
                self.solver.append(solver)

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.update_solution")
    def update_solution(self, i_stage, additive=False):
        """
        Computes the solution of the i-th stage

        Tendencies must have been evaluated first.

        If additive=False, will overwrite :attr:`solution` function, otherwise
        will add to it.
        """
        if not additive:
            self.solution.assign(self.solution_old)
        if self._nontrivial and i_stage > 0:
            self.solution += self.sol_expressions[i_stage]

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.solve_tendency")
    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            self.solver[i_stage].solve()

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.get_final_solution")
    def get_final_solution(self, additive=False):
        """Assign final solution to :attr:`self.solution`

        If additive=False, will overwrite :attr:`solution` function, otherwise
        will add to it.
        """
        if not additive:
            self.solution.assign(self.solution_old)
        if self._nontrivial:
            self.solution += self.final_sol_expr
        self.solution_old.assign(self.solution)

    @PETSc.Log.EventDecorator("thetis.ERKGeneric.solve_stage")
    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)


class ERKGenericShuOsher(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Shu-Osher form.
    """
    @PETSc.Log.EventDecorator("thetis.ERKGenericShuOsher.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(ERKGenericShuOsher, self).__init__(equation, solution, fields, dt, options)

        self.tendency = Function(self.equation.function_space, name='tendency')
        self.stage_sol = []
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name=f'sol{i}')
            self.stage_sol.append(s)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual(terms_to_add,
                                                         self.solution, self.solution,
                                                         self.fields, self.fields,
                                                         bnd_conditions)
        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        if self._nontrivial:
            self.sol_expressions = []
            for i_stage in range(self.n_stages):
                sol_expr = self.tendency*self.beta[i_stage + 1][i_stage] + sum(
                    map(operator.mul, self.stage_sol[:i_stage + 1],
                        self.alpha[i_stage + 1][:i_stage + 1]))
                self.sol_expressions.append(sol_expr)

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.ERKGenericShuOsher.update_solver")
    def update_solver(self):
        if self._nontrivial:
            prob = LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency)
            self.solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k',
                                                  solver_parameters=self.solver_parameters,
                                                  ad_block_tag=self.ad_block_tag + '_k')

    def initialize(self, solution):
        pass

    @PETSc.Log.EventDecorator("thetis.ERKGenericShuOsher.solve_stage")
    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)

            if i_stage == 0:
                self.stage_sol[0].assign(self.solution)

            # solve tendency
            self.solver.solve()

            # solve the next internal solution
            self.solution.assign(self.sol_expressions[i_stage])

            if i_stage < self.n_stages - 1:
                self.stage_sol[i_stage + 1].assign(self.solution)

    @PETSc.Log.EventDecorator("thetis.ERKGenericShuOsher.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)


class SSPRK33(ERKGenericShuOsher, SSPRK33Abstract):
    pass


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ESDIRKMidpoint(DIRKGeneric, ESDIRKMidpointAbstract):
    pass


class ESDIRKTrapezoid(DIRKGeneric, ESDIRKTrapezoidAbstract):
    pass


class ERKEuler(ERKGeneric, ForwardEulerAbstract):
    pass
