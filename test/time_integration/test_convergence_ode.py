"""
Convergence tests for time integrators
"""
from thetis import *
from thetis.equation import Term, Equation
from thetis.timeintegrator import *
from thetis.rungekutta import *
from thetis.implicitexplicit import *
from abc import ABCMeta, abstractproperty, abstractmethod
from scipy import stats
import pytest


class MixedODETerm(Term):
    """
    Abstract base class for ODE term in mixed space
    """
    __metaclass__ = ABCMeta
    def __init__(self, subspace, test_func, alpha):
        super(MixedODETerm, self).__init__(subspace)
        self.test_func = test_func
        self.alpha = alpha


class TermA(MixedODETerm):
    """
    Term :math:`\alpha b` for the ODE.
    """
    def residual(self, a, b, a_old, b_old, fields, fields_old, bnd_conditions):
        f = self.alpha*inner(b, self.test_func)*dx
        return f


class TermB(MixedODETerm):
    """
    Term :math:`-\alpha a` for the ODE.
    """
    def residual(self, a, b, a_old, b_old, fields, fields_old, bnd_conditions):
        f = -self.alpha*inner(a, self.test_func)*dx
        return f


class SimpleODEEquation(Equation):
    """
    A simple ODE for testing time integrators.

    Defines a linear, time-dependent, ODE

    .. math::
        \frac{\partial a}{\partial t} &= \alpha b \\
        \frac{\partial b}{\partial t} &= -\alpha a

    with :math:`\alpha=2\pi` and initial condition :math:`a(0)=0` and
    :math:`b(0)=1`. The analytical solution is a sinusoidal.

    .. math::
        a(t) &= \sin(\alpha t) \\
        b(t) &= \cos(\alpha t)

    We solve this system on a mixed function space, where the solution is (a, b).
    """
    def __init__(self, function_space, alpha, mode='explicit'):
        super(SimpleODEEquation, self).__init__(function_space)
        self.a_space, self.b_space = function_space.split()
        self.a_test, self.b_test = TestFunctions(function_space)
        self.alpha = alpha
        if mode=='imex':
            # solve one term implicitly, the other explicitly
            self.add_term_a('explicit')
            self.add_term_b('implicit')
        else:
            self.add_terms(mode)

    def add_terms(self, mode='explicit'):
        self.add_term_a(mode)
        self.add_term_b(mode)

    def add_term_a(self, mode='explicit'):
        self.add_term(TermA(self.a_space, self.a_test, self.alpha), mode)

    def add_term_b(self, mode='explicit'):
        self.add_term(TermB(self.b_space, self.b_test, self.alpha), mode)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            a, b = solution
        else:
            a, b = split(solution)
        a_old, b_old = split(solution_old)
        f = 0
        for term in self.select_terms(label):
            f += term.residual(a, b, a_old, b_old, fields, fields_old, bnd_conditions)
        return f


def run(timeintegrator_class, refinement=1):
    """
    Run test for the given time integrator
    """
    mesh = UnitSquareMesh(2, 2)
    # TODO create ThetisMesh and push the bnd def there
    bnd_len = compute_boundary_length(mesh)
    mesh.boundary_len = bnd_len

    p1 = get_functionspace_2d(mesh, 'CG', 1)
    fs = MixedFunctionSpace([p1, p1])

    alpha = 2*np.pi

    mode = 'explicit'
    cfl_coeff = timeintegrator_class.cfl_coeff if hasattr(timeintegrator_class, 'cfl_coeff') else None
    if (cfl_coeff==CFL_UNCONDITIONALLY_STABLE or
            isinstance(timeintegrator_class, DIRKGeneric)):
        mode = 'implicit'
    if (IMEXGeneric in timeintegrator_class.__bases__):
        mode = 'imex'
    equation = SimpleODEEquation(fs, alpha, mode=mode)

    init_solution = [0, 1]
    solution = Function(fs, name='solution').assign(Constant(init_solution))
    fields = {}

    end_time = 1.0
    base_dt = 0.01
    ntimesteps = int(np.round(end_time/base_dt*refinement))
    dt = end_time/ntimesteps
    print('Running refinement {:2d}, dt = {:.6f}'.format(refinement, dt))
    times = np.zeros((ntimesteps+1, ))
    values = np.zeros((ntimesteps+1, 2))

    ti = timeintegrator_class(equation, solution, fields, dt)
    ti.initialize(solution)
    simulation_time = 0
    sol_a, sol_b = solution.split()
    values[0, :] = sol_a.dat.data[0], sol_b.dat.data[0]
    for i in range(ntimesteps):
        simulation_time = (i+1)*dt
        ti.advance(simulation_time)
        #print('{:.4f} {:.4f} {:.4f}'.format(simulation_time, sol_a.dat.data[0], sol_b.dat.data[0]))
        times[i+1] = simulation_time
        values[i+1, :] = sol_a.dat.data[0], sol_b.dat.data[0]

    assert abs(times[-1] - end_time) < 1e-16

    exact_sol = np.vstack((np.sin(alpha*times), np.cos(alpha*times))).T
    l2_err = np.sqrt(np.mean((values - exact_sol)**2))
    return  l2_err


def run_convergence(timeintegrator_class, ref_list,
                    expected_slope, tolerance=0.05):
    print('Testing {:}'.format(timeintegrator_class.__name__))
    err_list = []
    for r in ref_list:
        l2_err = run(timeintegrator_class, r)
        err_list.append(l2_err)
    x_log = np.log10(np.array(ref_list, dtype=float)**-1)
    y_log = np.log10(np.array(err_list))

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    print('Convergence rate {:.4f}'.format(slope))
    msg = '{:}: Wrong convergence rate {:.3f}, expected {:.3f}'.format(
        timeintegrator_class.__name__, slope, expected_slope)
    assert abs(slope - expected_slope)/slope < tolerance, msg

    return slope


@pytest.mark.parametrize(('ti_class', 'convergence_rate'),
                         [
                             (ForwardEuler, 1.0),
                             (CrankNicolson, 2.0),
                             (SSPRK22ALE, 2.0),
                             (LeapFrogAM3, 2.0),
                             (BackwardEuler, 1.0),
                             (ImplicitMidpoint, 2.0),
                             (CrankNicolsonRK, 2.0),
                             (DIRK22, 2.0),
                             (DIRK23, 3.0),
                             (DIRK33, 3.0),
                             (DIRK43, 3.0),
                             (DIRKLSPUM2, 2.0),
                             (DIRKLPUM2, 2.0),
                             (ERKLSPUM2, 2.0),
                             (SSPRK33, 3.0),
                             (ERKLSPUM2, 2.0),
                             (ERKLPUM2, 2.0),
                             (ERKMidpoint, 2.0),
                             (ESDIRKMidpoint, 2.0),
                             (ESDIRKTrapezoid, 2.0),
                             (IMEXLPUM2, 2.0),
                             (IMEXLSPUM2, 2.0),
                             (IMEXMidpoint, 2.0),
                             (IMEXEuler, 1.0),
                         ])
def test_timeintegrator_convergence(ti_class, convergence_rate):
    run_convergence(ti_class, [1, 2, 3, 4], convergence_rate)
