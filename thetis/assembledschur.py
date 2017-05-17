from firedrake import *
from petsc4py import PETSc

class AssembledSchurPC(PCBase):
    """Preconditioner for the Schur complement, where the preconditioner
    matrix is assembled by explicitly matrix multiplying A10*Minv*A10. Here:
    A01, A10 are the assembled sub-blocks of the saddle point system. The form
        of this system needs to be supplied in the appctx to the solver as appctx['a']
    Minv is the inverse of the mass-matrix which is assembled as
        assemble(v*u*dx, inverse=True), i.e. the element-wise inverse, where
        v and u are the test and trial of the 00 block. This gives the exact
        inverse of the mass matrix for a DG discretisation."""
    def initialize(self, pc):
        _, P = pc.getOperators()
        ctx = P.getPythonContext()
        a = ctx.appctx['a']
        options_prefix = pc.getOptionsPrefix()

        test, trial = a.arguments()
        W = test.function_space()
        V, Q = W.split()
        v = TestFunction(V)
        u = TrialFunction(V)
        mass = Tensor(dot(v, u)*dx)

        fs = dict(formmanipulation.split_form(a))
        A01 = Tensor(fs[(0, 1)])
        A10 = Tensor(fs[(1, 0)])
        A11 = Tensor(fs[(1, 1)])
        self.S = A11 - A10*mass.inv*A01

        self.ksp = PETSc.KSP().create(comm=pc.comm)
        self.ksp.setOptionsPrefix(options_prefix + 'schur_')
        self.Smat = assemble(self.S, form_compiler_parameters=ctx.fc_params)
        #self.Smat_assembler = create_assembly_callable(self.S, tensor=self.Smat, form_compiler_parameters=ctx.fc_params)
        #self.Smat_assembler()
        self.Smat.force_evaluation()
        self.ksp.setOperators(self.Smat.petscmat, self.Smat.petscmat)
        self.ksp.setFromOptions()
        self.update(pc)

    def update(self, pc):
        assemble(self.S, tensor=self.Smat, form_compiler_parameters=ctx.fc_params)
        self.Smat.force_evaluation()

    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)
        r = self.ksp.getConvergedReason()
        if r < 0:
            raise RuntimeError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), solving_utils.KSPReasons[r])

    def view(self, pc, viewer=None):
        super(AssembledSchurPC, self).view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()

    def applyTranspose(self, pc, X, Y):
        raise NotImplemented("applyTranspose not implemented for AssembledSchurPC")
