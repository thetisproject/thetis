"""
Classes for computing diagnostics.
"""
from .utility import *


class VorticityCalculator2D(object):
    r"""
    Linear solver for recovering fluid vorticity.

    It is recommended that the vorticity is sought
    in :math:`\mathbb P1` space.

    :arg uv_2d: horizontal velocity :class:`Function`.
    :arg vorticity_2d: :class:`Function` to hold calculated vorticity.
    :kwargs: to be passed to the :class:`LinearVariationalSolver`.
    """
    @PETSc.Log.EventDecorator("thetis.VorticityCalculator2D.__init__")
    def __init__(self, uv_2d, vorticity_2d, **kwargs):
        fs = vorticity_2d.function_space()
        dim = fs.mesh().topological_dimension()
        if dim != 2:
            raise ValueError(f'Dimension {dim} not supported')
        if element_continuity(fs.ufl_element()).horizontal != 'cg':
            raise ValueError('Vorticity must be calculated in a continuous space')
        n = FacetNormal(fs.mesh())

        # Weak formulation
        test = TestFunction(fs)
        a = TrialFunction(fs)*test*dx
        L = -inner(perp(uv_2d), grad(test))*dx \
            + dot(perp(uv_2d), test*n)*ds \
            + dot(avg(perp(uv_2d)), jump(test, n))*dS

        # Setup vorticity solver
        prob = LinearVariationalProblem(a, L, vorticity_2d)
        kwargs.setdefault('solver_parameters', {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        })
        self.solver = LinearVariationalSolver(prob, **kwargs)

    @PETSc.Log.EventDecorator("thetis.VorticityCalculator2D.solve")
    def solve(self):
        self.solver.solve()
