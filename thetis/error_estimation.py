"""
Implements ErrorEstimator classes
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation


class ErrorEstimatorTerm(Term):
    """
    Implements the component of an error estimator from a single term of the underlying equation.

    .. note::
        Sign convention as in :class:`Term`.
    """
    def __init__(self, function_space):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        super(ErrorEstimatorTerm, self).__init__(function_space)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)

    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        # TODO: doc
        raise NotImplementedError('Must be implemented in the derived class')

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        # TODO: doc
        raise NotImplementedError('Must be implemented in the derived class')

    # TODO: I do not want the Jacobian method


class ErrorEstimator(Equation):
    """
    Implements an error estimator, comprised of the corresponding terms from the underlying equation.
    """
    def __init__(self, function_space):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        super(ErrorEstimator, self).__init__(function_space)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)

    def mass_term(self, solution, arg):
        # TODO: doc
        return self.p0test*inner(solution, arg)*dx

    def residual(self, label, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        # TODO: doc
        cell_residual_terms = 0
        for term in self.select_terms(label):
            cell_residual_terms += term.residual(solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions)
        cell_residual = Function(self.P0, name="Cell residual")
        cell_residual.interpolate(assemble(cell_residual_terms))
        return cell_residual

    def flux(self, label, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        # TODO: doc
        mass_term = self.p0test*self.p0trial*dx
        flux_terms = 0
        for term in self.select_terms(label):
            flux_terms += term.flux(solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions)
        flux = Function(self.P0, name="Flux and boundary terms")
        solve(mass_term == flux_term, flux)  # TODO: Solver parameters?
        return flux

    # TODO: I do not want the Jacobian method
