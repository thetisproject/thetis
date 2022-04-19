"""
Implicit-explicit time integrators

"""
from .rungekutta import *


class IMEXGeneric(TimeIntegrator, ABC):
    """
    Generic implementation of Runge-Kutta Implicit-Explicit schemes

    Derived classes must define the implicit :attr:`dirk_class` and explicit
    :attr:`erk_class` Runge-Kutta time integrator classes.

    This method solves the linearized equations: All implicit terms are fed to
    the implicit solver, while all the other terms are fed to the explicit
    solver. In case of non-linear terms proper linearization must defined in the
    equation using the two solution functions (solution, solution_old)
    """
    @abstractproperty
    def dirk_class(self):
        """Implicit DIRK class"""
        pass

    @abstractproperty
    def erk_class(self):
        """Explicit Runge-Kutta class"""
        pass

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions):
        """
        :arg equation: equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        """
        super(IMEXGeneric, self).__init__(equation, solution, fields, dt, options)
        # NOTE: The same solver parameters are currently used for both implicit and explicit schemes

        # implicit scheme
        self.dirk = self.dirk_class(equation, solution, fields, dt, options,
                                    bnd_conditions=bnd_conditions,
                                    terms_to_add=('implicit'))
        self.dirk.ad_block_tag += '_impl'
        # explicit scheme
        self.erk = self.erk_class(equation, solution, fields, dt, options,
                                  bnd_conditions=bnd_conditions,
                                  terms_to_add=('explicit', 'source'))
        self.erk.ad_block_tag += '_expl'
        assert self.erk.n_stages == self.dirk.n_stages
        self.n_stages = self.erk.n_stages
        # FIXME this assumes that we are limited by whatever ERK solves ...
        # FIXME may violate max DIRK SSP time step (if any)
        self.cfl_coeff = self.erk.cfl_coeff

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.update_solver")
    def update_solver(self):
        """Create solver objects"""
        self.dirk.update_solver()
        self.erk.update_solver()

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.solve_stage")
    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves i-th stage
        """
        # set solution to u_n + dt*sum(a*k_erk)
        self.erk.update_solution(i_stage, additive=False)
        # set correct reference solution in the DIRK solver
        self.dirk.solution_old.assign(self.erk.solution)
        # solve implicit tendency (this is implicit solve)
        self.dirk.solve_tendency(i_stage, t, update_forcings)
        # set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
        self.dirk.update_solution(i_stage)
        # evaluate explicit tendency
        self.erk.solve_tendency(i_stage, t, update_forcings)

    @PETSc.Log.EventDecorator("thetis.IMEXGeneric.get_final_solution")
    def get_final_solution(self):
        """
        Evaluates the final solution.
        """
        # set solution to u_n + sum(b*k_erk)
        self.erk.get_final_solution(additive=False)
        # set correct reference solution in the DIRK solver
        self.dirk.solution_old.assign(self.erk.solution)
        # set solution to u_n + sum(b*k_erk) + sum(b*k_dirk)
        self.dirk.get_final_solution()
        # update old solution
        self.erk.solution_old.assign(self.dirk.solution)

    def set_dt(self, dt):
        """
        Update time step

        :arg float dt: time step
        """
        self.erk.set_dt(dt)
        self.dirk.set_dt(dt)


class IMEXLPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (20) in Higureras et al. (2014)

    CFL coefficient is 2.0

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    erk_class = ERKLPUM2
    dirk_class = DIRKLPUM2


class IMEXLSPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (17) in Higureras et al. (2014)

    CFL coefficient is 2.0

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    erk_class = ERKLSPUM2
    dirk_class = DIRKLSPUM2


class IMEXMidpoint(IMEXGeneric):
    """
    Implicit-explicit midpoint scheme (1, 2, 2) from Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """
    erk_class = ERKMidpoint
    dirk_class = ESDIRKMidpoint


class IMEXEuler(IMEXGeneric):
    """
    Forward-Backward Euler
    """
    erk_class = ERKEuler
    dirk_class = BackwardEuler
