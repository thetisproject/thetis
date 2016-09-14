from firedrake import *
from firedrake.petsc import PETSc

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
        print "Initializing AssembledSchurPC"
        _, P = pc.getOperators()
        ctx = P.getPythonContext()
        a = ctx.appctx['a']
        options_prefix = pc.getOptionsPrefix()

        test, trial = a.arguments()
        W = test.function_space()
        V, Q = W.split()
        v = TestFunction(V)
        u = TrialFunction(V)
        mass = dot(v, u)*dx
        self.A00 = assemble(mass).M.handle
        self.A00_inv = assemble(mass, inverse=True).M.handle
        self.A00_inv.convert(PETSc.Mat.Type.AIJ)
        self.A10_A00_inv = None
        self.schur = None
        self.schur_plus = None

        fs = dict(formmanipulation.split_form(a))
        self.a01 = fs[(0,1)]
        self.a10 = fs[(1,0)]
        self.a11 = fs[(1,1)]
        self.ksp = PETSc.KSP().create()
        self.ksp.setOptionsPrefix(options_prefix + 'schur_')
        self.ksp.setFromOptions()
        self.update(pc)

    def update(self, pc):
        print "Updating AssembledSchurPC"
        self.A01 = assemble(self.a01).M.handle
        self.A10 = assemble(self.a10).M.handle
        self.A11 = assemble(self.a11).M.handle

        self.A10_A00_inv = self.A10.matMult(self.A00_inv, self.A10_A00_inv, 2.0)
        self.schur = self.A10_A00_inv.matMult(self.A01, self.schur, 2.0)
        if self.schur_plus is None:
          self.schur_plus = self.schur.duplicate(copy=True)
        else:
          self.schur_plus = self.schur.copy(self.schur_plus, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.schur_plus.aypx(-1.0, self.A11, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.ksp.setOperators(self.schur_plus)

    def apply(self, pc, X, Y):
        print "Applying AssembledSchurPC"

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
        
