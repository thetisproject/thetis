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


class HessianRecoverer2D(object):
    r"""
    Linear solver for recovering Hessians.

    Hessians are recoved using double :math:`L^2`
    projection, which is implemented using a
    mixed finite element method.

    It is recommended that gradients and Hessians
    are sought in :math:`\mathbb P1` space of
    appropriate dimension.

    :arg field_2d: :class:`Function` to recover the Hessian of.
    :arg hessian_2d: :class:`Function` to hold recovered Hessian.
    :kwarg gradient_2d: :class:`Function` to hold recovered gradient.
    :kwargs: to be passed to the :class:`LinearVariationalSolver`.
    """
    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.__init__")
    def __init__(self, field_2d, hessian_2d, gradient_2d=None, **kwargs):
        self.hessian_2d = hessian_2d
        self.gradient_2d = gradient_2d
        Sigma = hessian_2d.function_space()
        mesh = Sigma.mesh()
        dim = mesh.topological_dimension()
        if dim != 2:
            raise ValueError(f'Dimension {dim} not supported')
        n = FacetNormal(mesh)

        # Extract function spaces
        if element_continuity(Sigma.ufl_element()).horizontal != 'cg':
            raise ValueError('Hessians must be calculated in a continuous space')
        if Sigma.dof_dset.dim != (2, 2):
            raise ValueError('Expecting a square tensor function')
        if gradient_2d is None:
            V = get_functionspace(mesh, 'CG', 1, vector=True)
        else:
            V = gradient_2d.function_space()
            if element_continuity(V.ufl_element()).horizontal != 'cg':
                raise ValueError('Gradients must be calculated in a continuous space')
            if V.dof_dset.dim != (2,):
                raise ValueError('Expecting a 2D vector function')

        # Setup function spaces
        W = V*Sigma
        g, H = TrialFunctions(W)
        phi, tau = TestFunctions(W)
        sol = Function(W)
        self._gradient, self._hessian = sol.split()

        # The formulation is chosen such that f does not need to have any
        # finite element derivatives
        a = inner(tau, H)*dx \
            + inner(div(tau), g)*dx \
            + inner(phi, g)*dx \
            - dot(g, dot(tau, n))*ds \
            - dot(avg(g), jump(tau, n))*dS
        L = field_2d*dot(phi, n)*ds \
            + avg(field_2d)*jump(phi, n)*dS \
            - field_2d*div(phi)*dx

        # Apply stationary preconditioners in the Schur complement to get away
        # with applying GMRES to the whole mixed system
        sp = {
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_max_it": 20,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_0_fields": "1",
            "pc_fieldsplit_1_fields": "0",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "gamg",
            "fieldsplit_1_mg_levels_ksp_max_it": 5,
        }
        if COMM_WORLD.size == 1:
            sp["fieldsplit_0_pc_type"] = "ilu"
            sp["fieldsplit_1_mg_levels_pc_type"] = "ilu"
        else:
            sp["fieldsplit_0_pc_type"] = "bjacobi"
            sp["fieldsplit_0_sub_ksp_type"] = "preonly"
            sp["fieldsplit_0_sub_pc_type"] = "ilu"
            sp["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
            sp["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
            sp["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"

        # Setup solver
        prob = LinearVariationalProblem(a, L, sol)
        kwargs.setdefault('solver_parameters', sp)
        self.solver = LinearVariationalSolver(prob, **kwargs)

    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.solve")
    def solve(self):
        self.solver.solve()
        self.hessian_2d.assign(self._hessian)
        if self.gradient_2d is not None:
            self.gradient_2d.assign(self._gradient)
