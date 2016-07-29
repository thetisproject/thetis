"""
Implicit-explicit time integrators

"""
from __future__ import absolute_import
from .rungekutta import *


# OBSOLETE
class SSPIMEX(TimeIntegrator):
    """
    SSP-IMEX time integration scheme based on [1], method (17).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}):
        super(SSPIMEX, self).__init__(equation, solution, fields, dt, solver_parameters)

        # implicit scheme
        self.dirk = DIRKLSPUM2(equation, solution, fields, dt, bnd_conditions,
                               solver_parameters=solver_parameters_dirk,
                               terms_to_add=('implicit'))
        # explicit scheme
        self.erk = ERKLSPUM2(equation, solution, fields, dt, bnd_conditions,
                             solver_parameters=solver_parameters,
                             terms_to_add=('explicit', 'source'))
        self.n_stages = len(self.erk.b)

    def update_solver(self):
        self.dirk.update_solver()
        self.erk.update_solver()

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()

    def solve_stage(self, i_stage, t, update_forcings=None):
        self.erk.solve_stage(i_stage, t, update_forcings)
        self.dirk.solve_stage(i_stage, t, update_forcings)

    def get_final_solution(self):
        self.erk.get_final_solution()
        self.dirk.get_final_solution()


class IMEXGeneric(TimeIntegrator):
    """
    Generic implementation of Runge-Kutta Impicit-Explicit schemes
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def dirk_class(self):
        pass

    @abstractproperty
    def erk_class(self):
        pass

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}):
        super(IMEXGeneric, self).__init__(equation, solution, fields, dt, solver_parameters)

        # implicit scheme
        self.dirk = self.dirk_class(equation, solution, fields, dt, bnd_conditions,
                                    solver_parameters=solver_parameters_dirk,
                                    terms_to_add=('implicit'))
        # explicit scheme
        self.erk = self.erk_class(equation, solution, fields, dt, bnd_conditions,
                                  solver_parameters=solver_parameters,
                                  terms_to_add=('explicit', 'source'))
        assert self.erk.n_stages == self.dirk.n_stages
        self.n_stages = self.erk.n_stages
        # FIXME this assumes that we are limited by whatever DIRK solves ...
        # FIXME this really depends on the DIRK/ERK processes ...
        self.cfl_coeff = self.dirk.cfl_coeff

    def update_solver(self):
        self.dirk.update_solver()
        self.erk.update_solver()

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves one sub-stage.
        """
        # set solution to u_n + dt*sum(a*k_erk)
        self.erk.update_solution(i_stage, additive=False)
        # solve implicit tendency (this is implicit solve)
        self.dirk.solve_tendency(i_stage, t, update_forcings)
        # set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
        self.dirk.update_solution(i_stage, additive=True)
        # evaluate explicit tendency
        self.erk.solve_tendency(i_stage, t, update_forcings)

    def get_final_solution(self):
        """
        Evaluates the final solution.
        """
        # set solution to u_n + sum(b*k_erk)
        self.erk.get_final_solution(additive=False)
        # set solution to u_n + sum(b*k_erk) + sum(b*k_dirk)
        self.dirk.get_final_solution(additive=True)
        # update old solution
        # TODO share old solution func between dirk and erk
        self.erk.solution_old.assign(self.dirk.solution)

    def set_dt(self, dt):
        self.erk.set_dt(dt)
        self.dirk.set_dt(dt)


class IMEXLPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (20) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    erk_class = ERKLPUM2
    dirk_class = DIRKLPUM2


class IMEXLSPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    erk_class = ERKLSPUM2
    dirk_class = DIRKLSPUM2


class IMEXMidpoint(IMEXGeneric):
    """
    Implicit-explicit midpoint scheme (1, 2, 2)

    From Ascher (1997)
    """
    erk_class = ERKMidpoint
    dirk_class = ESDIRKMidpoint


class IMEXEuler(IMEXGeneric):
    """
    Forward-Backward Euler
    """
    erk_class = ERKEuler
    dirk_class = DIRKEuler
