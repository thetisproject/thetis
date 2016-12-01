"""
Implements Runge-Kutta time integration methods.

The abstract class :class:`~.AbstractRKScheme` defines the Runge-Kutta
coefficients, and can be used to implement generic time integrators.
"""
from __future__ import absolute_import
from .timeintegrator import *
from abc import ABCMeta, abstractproperty
import operator

CFL_UNCONDITIONALLY_STABLE = np.inf
# CFL coefficient for unconditionally stable methods


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

    butcher = np.vstack((a, b))

    implicit = np.diag(a).any()

    if implicit:
        # a is not singular
        # take diag entries of a to beta
        be_0 = np.diag(np.diag(a, k=0), k=0)
        be_1 = np.zeros_like(b)
        be_1[-1] = b[-1]
        be = np.vstack((be_0, be_1))

        n = a.shape[0]
        iden = np.eye(n)
        al_0 = iden - np.dot(be_0, linalg.inv(a))
        al_1 = np.dot((b - be_1), np.dot(linalg.inv(be_0), (iden - al_0)))
        al = np.vstack((al_0, al_1))

        # construct full shu-osher form
        alpha = np.zeros((n+1, n+1))
        alpha[:, 1:] = al
        # consistency
        alpha[:, 0] = 1.0 - np.sum(alpha, axis=1)
        beta = np.zeros((n+1, n+1))
        beta[:, 1:] = be
    else:
        # a is singular: solve for lower part of butcher tableau
        aa = butcher[1:, :]
        # take diag entries of aa to beta
        be_0 = np.diag(np.diag(aa, k=0), k=0)
        n = aa.shape[0]
        iden = np.eye(n)
        al_0 = iden - np.dot(be_0, linalg.inv(aa))

        # construct full shu-osher form
        alpha = np.zeros((n+1, n+1))
        alpha[1:, 1:] = al_0
        # consistency
        alpha[:, 0] = 1.0 - np.sum(alpha, axis=1)
        beta = np.zeros((n+1, n+1))
        beta[1:, :-1] = be_0

    # round off small entries
    alpha[np.abs(alpha) < 1e-13] = 0.0
    beta[np.abs(beta) < 1e-13] = 0.0

    # check sanity
    assert np.allclose(np.sum(alpha, axis=1), 1.0)
    if implicit:
        err = beta[:, 1:] - (butcher - np.dot(alpha[:, 1:], a))
    else:
        err = beta[:, :-1] - (butcher - np.dot(alpha[:, :-1], a))
    assert np.allclose(err, 0.0)

    return alpha, beta


class AbstractRKScheme(object):
    """
    Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """
    __metaclass__ = ABCMeta

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
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert np.allclose(np.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()

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


class ERKTrapezoidAbstract(AbstractRKScheme):
    r"""
    Explicit Trapezoid scheme

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
        0.0 & 0.0 & 0.0 \\
        1.0 & 1.0 & 0.0 \\ \hline
            & 0.5 & 0.5
        \end{array}
    """
    a = [[0.0, 0.0],
         [1.0, 0.0]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = 1.0


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

    with :math:`\gamma = (2 + \sqrt{2})/2`.

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    gamma = (2.0 + np.sqrt(2.0))/2.0
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
    gamma = (3 + np.sqrt(3))/6
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


class RungeKuttaTimeIntegrator(TimeIntegrator):
    """Abstract base class for all Runge-Kutta time integrators"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def get_final_solution(self, additive=False):
        """
        Evaluates the final solution
        """
        pass

    @abstractproperty
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
            error('Time integrator {:} is not initialized'.format(self.name))
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()


class DIRKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic implementation of Diagonally Implicit Runge Kutta schemes.

    All derived classes must define the Butcher tableau coefficients :attr:`a`,
    :attr:`b`, :attr:`c`.
    """
    def __init__(self, equation, solution, fields, dt,
                 bnd_conditions=None, solver_parameters={}, terms_to_add='all'):
        """
        :param equation: the equation to solve
        :type equation: :class:`Equation` object
        :param solution: :class:`Function` where solution will be stored
        :param fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :param dt: time step in seconds
        :type dt: float
        :param bnd_conditions: Dictionary of boundary conditions passed to the equation
        :type bnd_conditions: dict
        :param solver_parameters: PETSc solver options
        :type solver_parameters: dict
        :param terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(DIRKGeneric, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')
        self._initialized = False

        fs = self.equation.function_space
        self.solution_old = Function(self.equation.function_space, name='old solution')

        test = self.equation.test
        mixed_space = len(fs) > 1

        # Allocate tendency fields
        self.k = []
        for i in xrange(self.n_stages):
            fname = '{:}_k{:}'.format(self.name, i)
            self.k.append(Function(fs, name=fname))

        # construct variational problems
        self.F = []
        if not mixed_space:
            for i in xrange(self.n_stages):
                for j in xrange(i+1):
                    if j == 0:
                        u = self.solution_old + self.a[i][j]*self.dt_const*self.k[j]
                    else:
                        u += self.a[i][j]*self.dt_const*self.k[j]
                self.F.append(-inner(self.k[i], test)*dx +
                              self.equation.residual(terms_to_add, u, self.solution_old, fields, fields, bnd_conditions))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in xrange(self.n_stages):
                for j in xrange(i+1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(split(self.solution_old), split(self.k[j])):
                            u.append(s + self.a[i][j]*self.dt_const*k)
                    else:
                        for l, k in enumerate(split(self.k[j])):
                            u[l] += self.a[i][j]*self.dt_const*k
                self.F.append(-inner(self.k[i], test)*dx +
                              self.equation.residual(terms_to_add, u, self.solution_old, fields, fields, bnd_conditions))
        self.update_solver()

        # construct expressions for stage solutions
        self.sol_expressions = []
        for i_stage in range(self.n_stages):
            sol_expr = reduce(operator.add,
                              map(operator.mul, self.k[:i_stage+1], self.dt_const*self.a[i_stage][:i_stage+1]))
            self.sol_expressions.append(sol_expr)
        self.final_sol_expr = reduce(operator.add,
                                     map(operator.mul, self.k, self.dt_const*self.b),
                                     self.solution_old)

    def update_solver(self):
        """Create solver objects"""
        self.solver = []
        for i in xrange(self.n_stages):
            p = NonlinearVariationalProblem(self.F[i], self.k[i])
            sname = '{:}_stage{:}_'.format(self.name, i)
            self.solver.append(
                NonlinearVariationalSolver(p,
                                           solver_parameters=self.solver_parameters,
                                           options_prefix=sname))

    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)
        self._initialized = True

    def update_solution(self, i_stage):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        """
        self.solution += self.sol_expressions[i_stage]

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

    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution`"""
        self.solution.assign(self.final_sol_expr)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.solve_tendency(i_stage, t, update_forcings)
        self.update_solution(i_stage)


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


class ERKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form. All terms in the equation are treated explicitly.
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, terms_to_add='all'):
        """
        :param equation: the equation to solve
        :type equation: :class:`Equation` object
        :param solution: :class:`Function` where solution will be stored
        :param fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :param dt: time step in seconds
        :type dt: float
        :param bnd_conditions: Dictionary of boundary conditions passed to the equation
        :type bnd_conditions: dict
        :param solver_parameters: PETSc solver options
        :type solver_parameters: dict
        :param terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(ERKGeneric, self).__init__(equation, solution, fields, dt, solver_parameters)

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
                sol_expr = reduce(operator.add,
                                  map(operator.mul, self.tendency[:i_stage], self.a[i_stage][:i_stage]),
                                  0.0)
                self.sol_expressions.append(sol_expr)
            self.final_sol_expr = reduce(operator.add,
                                         map(operator.mul, self.tendency, self.b))

        self.update_solver()

    def update_solver(self):
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency[i])
                solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                                 solver_parameters=self.solver_parameters)
                self.solver.append(solver)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

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

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            self.solver[i_stage].solve()

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

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)


class ERKGenericShuOsher(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Shu-Osher form.
    """
    # TODO derive from RungeKuttaTimeIntegrator class?
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, terms_to_add='all'):
        """
        :param equation: the equation to solve
        :type equation: :class:`Equation` object
        :param solution: :class:`Function` where solution will be stored
        :param fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :param dt: time step in seconds
        :type dt: float
        :param bnd_conditions: Dictionary of boundary conditions passed to the equation
        :type bnd_conditions: dict
        :param solver_parameters: PETSc solver options
        :type solver_parameters: dict
        :param terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(ERKGenericShuOsher, self).__init__(equation, solution, fields, dt, solver_parameters)

        self.tendency = Function(self.equation.function_space, name='tendency')
        self.stage_sol = []
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name='sol{:}'.format(i))
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
                sol_expr = reduce(operator.add,
                                  map(operator.mul,
                                      self.stage_sol[:i_stage + 1],
                                      self.alpha[i_stage + 1][:i_stage + 1]),
                                  self.tendency*self.beta[i_stage + 1][i_stage])
                self.sol_expressions.append(sol_expr)

        self.update_solver()

    def update_solver(self):
        if self._nontrivial:
            prob = LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency)
            self.solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k',
                                                  solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        pass

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

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)


class ERKGenericALE2(RungeKuttaTimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator for conservative ALE schemes.

    Implements the Butcher tableau.
    """

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """
        :param equation: the equation to solve
        :type equation: :class:`Equation` object
        :param solution: :class:`Function` where solution will be stored
        :param fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :param dt: time step in seconds
        :type dt: float
        :param bnd_conditions: Dictionary of boundary conditions passed to the equation
        :type bnd_conditions: dict
        :param solver_parameters: PETSc solver options
        :type solver_parameters: dict
        :param terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(ERKGenericALE2, self).__init__(equation, solution, fields, dt, solver_parameters)

        self.l_form = Function(self.equation.function_space, name='linear form')
        self.msol_old = Function(self.equation.function_space, name='old dual solution')
        self.stage_mk = []  # mass_matrix*tendency
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name='dual tendency {:}'.format(i))
            self.stage_mk.append(s)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual('all', self.solution, self.solution, self.fields, self.fields, bnd_conditions)
        self.mass_term = self.equation.mass_term(self.solution)

        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        self.sol_expressions = []
        if self._nontrivial:
            for i_stage in range(self.n_stages):
                sol_expr = reduce(operator.add,
                                  map(operator.mul, self.stage_mk[:i_stage], self.a[i_stage][:i_stage]),
                                  self.msol_old)
                self.sol_expressions.append(sol_expr)
        self.final_sol_expr = reduce(operator.add,
                                     map(operator.mul, self.stage_mk, self.b),
                                     self.msol_old)

        self.update_solver()

    def update_solver(self):
        if self._nontrivial:
            self.A = assemble(self.a_rk)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        assemble(self.mass_term, self.msol_old)

    def update_solution(self, i_stage):
        """
        Computes the solution of the i-th stage

        Tendencies must have been evaluated first.
        """
        if self._nontrivial:
            # construct full form: L = c*dt*F + M_n*sol_n + ...
            self.l_form.assign(self.sol_expressions[i_stage])

            assemble(self.a_rk, self.A)
            solve(self.A, self.solution, self.l_form)

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            assemble(self.l_rk, self.stage_mk[i_stage])

    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution` """
        if self._nontrivial:
            self.l_form.assign(self.final_sol_expr)
            assemble(self.a_rk, self.A)
            solve(self.A, self.solution, self.l_form)
            assemble(self.mass_term, self.msol_old)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)
        if i_stage == self.n_stages - 1:
            self.get_final_solution()


class ERKSemiImplicitGeneric(RungeKuttaTimeIntegrator):
    """
    Generic implementation of semi-implicit RK schemes.

    If semi_implicit=True, this corresponds to a linearized semi-implicit
    scheme. The linearization must be defined in the equation using solution and
    solution_old functions: residual = residual(solution, solution_old)

    If semi_implicit=False, this corresponds to a fully non-linear scheme:
    residual = residual(solution, solution)
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, semi_implicit=False, theta=0.5):
        """
        :param equation: the equation to solve
        :type equation: :class:`Equation` object
        :param solution: :class:`Function` where solution will be stored
        :param fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :param dt: time step in seconds
        :type dt: float
        :param bnd_conditions: Dictionary of boundary conditions passed to the equation
        :type bnd_conditions: dict
        :param solver_parameters: PETSc solver options
        :type solver_parameters: dict
        :param semi_implicit: If True use a linearized semi-implicit scheme
        :type semi_implicit: bool
        :param theta: Implicitness parameter, default 0.5
        :type theta: float
        """
        super(ERKSemiImplicitGeneric, self).__init__(equation, solution, fields, dt, solver_parameters)

        assert self.n_stages == 3, 'This method supports only for 3 stages'

        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.theta = Constant(theta)

        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.stage_sol = []
        for i in range(self.n_stages - 1):
            s = Function(self.equation.function_space, name='solution stage {:}'.format(i))
            self.stage_sol.append(s)
        self.stage_sol.append(self.solution)

        sol_nl = [None]*self.n_stages
        if semi_implicit:
            # linearize around previous sub-timestep using the fact that all
            # terms are written in the form A(u_nl) u
            sol_nl[0] = self.solution_old
            sol_nl[1] = self.stage_sol[0]
            sol_nl[2] = self.stage_sol[1]
        else:
            # solve the full nonlinear residual form
            sol_nl[0] = self.stage_sol[0]
            sol_nl[1] = self.stage_sol[1]
            sol_nl[2] = self.solution

        args = (self.fields, self.fields, bnd_conditions)
        self.L = [None]*self.n_stages
        self.F = [None]*self.n_stages
        self.L[0] = (self.theta*self.equation.residual('implicit', self.stage_sol[0], sol_nl[0], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.solution_old, self.solution_old, *args) +
                     self.equation.residual('explicit', self.solution_old, self.solution_old, *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[0] = (self.equation.mass_term(self.stage_sol[0]) -
                     self.alpha[1][0]*self.equation.mass_term(self.solution_old) -
                     self.beta[1][0]*self.dt_const*self.L[0])
        self.L[1] = (self.theta*self.equation.residual('implicit', self.stage_sol[1], sol_nl[1], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.stage_sol[0], self.stage_sol[0], *args) +
                     self.equation.residual('explicit', self.stage_sol[0], self.stage_sol[0], *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[1] = (self.equation.mass_term(self.stage_sol[1]) -
                     self.alpha[2][0]*self.equation.mass_term(self.solution_old) -
                     self.alpha[2][1]*self.equation.mass_term(self.stage_sol[0]) -
                     self.beta[2][1]*self.dt_const*self.L[1])
        self.L[2] = (self.theta*self.equation.residual('implicit', self.stage_sol[2], sol_nl[2], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.stage_sol[1], self.stage_sol[1], *args) +
                     self.equation.residual('explicit', self.stage_sol[1], self.stage_sol[1], *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[2] = (self.equation.mass_term(self.stage_sol[2]) -
                     self.alpha[3][0]*self.equation.mass_term(self.solution_old) -
                     self.alpha[3][1]*self.equation.mass_term(self.stage_sol[0]) -
                     self.alpha[3][2]*self.equation.mass_term(self.stage_sol[1]) -
                     self.beta[3][2]*self.dt_const*self.L[2])
        self.update_solver()

    def update_solver(self):
        self.solver = []
        for i in range(self.n_stages):
            prob = NonlinearVariationalProblem(self.F[i], self.stage_sol[i])
            solv = NonlinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                              solver_parameters=self.solver_parameters)
            self.solver.append(solv)

    def initialize(self, solution):
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()
        if i_stage == self.n_stages - 1:
            self.solution_old.assign(self.solution)
        else:
            self.solution.assign(self.stage_sol[i_stage])

    def get_final_solution(self):
        pass


class SSPRK33SemiImplicit(ERKSemiImplicitGeneric, SSPRK33Abstract):
    pass


class SSPRK33(ERKGenericShuOsher, SSPRK33Abstract):
    pass


class ERKLSPUM2SemiImplicit(ERKSemiImplicitGeneric, ERKLSPUM2Abstract):
    pass


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLSPUM2ALE(ERKGenericALE2, ERKLSPUM2Abstract):
    pass


class ERKLPUM2SemiImplicit(ERKSemiImplicitGeneric, ERKLPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKLPUM2ALE(ERKGenericALE2, ERKLPUM2Abstract):
    pass


class ERKTrapezoidRK(ERKGeneric, ERKTrapezoidAbstract):
    pass


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ERKMidpointALE(ERKGenericALE2, ERKMidpointAbstract):
    pass


class ESDIRKMidpoint(DIRKGeneric, ESDIRKMidpointAbstract):
    pass


class ERKEuler(ERKGeneric, ForwardEulerAbstract):
    pass


class ERKEulerALE(ERKGenericALE2, ForwardEulerAbstract):
    pass


class DIRKEuler(DIRKGeneric, BackwardEulerAbstract):
    pass
