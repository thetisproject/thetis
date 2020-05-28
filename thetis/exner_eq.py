r"""
Some maths about the Exner equation...
"""
from __future__ import absolute_import
from .equation import Term, Equation

__all__ = [
    'ExnerEquation',
    'ExnerTerm',
]


class ExnerTerm(Term):
    """
    Generic tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        """
        super(TracerTerm, self).__init__(function_space)

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)


class ExnerSourceTerm(ExnerTerm):
    r"""
    some maths
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        z1 = solution
        z0 = solution_old

        tracer = fields['tracer']


        f = 0#
        return f

class ExnerEquation(Equation):
    """
    """
    def __init__(self, function_space):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        """
        super().__init__(function_space)

        args = (function_space,)
        self.add_term(ExnerSourceTerm(*args), 'implicit')
